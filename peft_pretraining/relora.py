import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF

from transformers import AutoModelForCausalLM, AutoConfig

from loguru import logger # 使用loguru库来记录日志，这对于调试和监控模型训练过程非常有用

# 定义ReLoRaConfig类，ReLoRa的配置
@dataclass # python里的一个装饰器，用于简化类的定义 有了它不用写init self等 @用于定义装饰器
class ReLoRaConfig: # 这是一个数据类
    r: int
    lora_alpha: int 
    lora_dropout: float
    target_modules: List[str]
    keep_original_weights: bool
    lora_only: bool = False
    trainable_scaling: bool = False
    quantize: str = None
    use_double_quant: bool = False

# 单独又定义了一个merge_and_reinit函数
def merge_and_reinit_functional(module): # restart
    if not isinstance(module, ReLoRaLinear): # 如果module不是ReLoRaLinear return
        return

    if module.quantize is not None:
        # Look below in merge_and_reinint method for the inspiration on how to implement this
        raise NotImplementedError("merge_and_reinit_functional for quantized models is not implemented yet. Use non-functional implementation")
        # raise 抛出异常类 NotImplementedError 报错
    _delta = module.lora_B.weight @ module.lora_A.weight # _delta是一个新的矩阵 @是矩阵乘法
    _delta = _delta * module._post_lora_scale() # 对_delta进行scale *是element-wise multiplication
    module.weight.data += _delta # merge
    nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5)) # reinit A=kaiming

    nn.init.zeros_(module.lora_B.weight) # B=0
    if module.trainable_scaling: # 判断是否为真（非空，非零）
        nn.init.zeros_(module.scaling)

"""
定义ReLoRaModel类，继承自torch.nn.Module
包装一个pretrained model 并在指定的线性层中替换为ReLoRaLinear层
ReLoRaModel是一个包装器 负责管理整个model model里有多个modules 每个module可能代表一个层或子模块
"""
class ReLoRaModel(torch.nn.Module): 
    """
    继承torch.nn.Module吗？是的torch.nn.Module是PyTorch中的一个基类，即nn的module，
    所有nn模块都应该继承自这个类
    ReLoRaModel类包装(wrap)一个预训练模型，并在其中应用ReLoRaLinear层
    wrap 包装 是指在现有的模型基础上添加或替换某些组件来创建一个新的模型结构。
    例如，给一个语言模型添加一个新的输出层，或者用ReLoRa技术增强一个预训练模型的特定层
    "包装"是模型结构上的改动，而"微调"是训练过程
    包装好了再用数据train这个model 即微调
    """
    def __init__(
        self,
        model, # 位置参数，可以直接传递 如 kkk = ReLoRaModel(mymodel)
        *, # *后面的均为关键字参数，这意味着调用函数时，必须明确指定这些参数的名称。
        # 如 kkk = ReLoRaModel(mymodel, target_modules=, r=, lora_alpha=)
        target_modules,
        r=128, # rank
        lora_alpha=32, # lora alpha
        lora_dropout=0.1, # dropout随机丢弃神经元防止过拟合，train每个batch时（即每个step ）随机丢弃10个神经元
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False, # ??
        quantize=None,
        use_double_quant=False,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")
            # raise 抛出异常类 ValueError 报错
        super().__init__() # super()是一个内置函数 返回当前类的父类对象 这里是调用父类的__init__方法
        self.wrapped_model: nn.Module = model # 将model 赋值给当前实例(self)的属性wrapped_model，
        # 并且使用类型注解指定该属性的类型为 nn.Module; : nn.Module 是一种类型注解（type hint），用于指明 wrapped_model 的类型
        # wrap 包装，wrapped function 包装函数
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling

        self._config = ReLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
            quantize=quantize,
            use_double_quant=use_double_quant,
        )

        # patch methods
        self.forward = self.wrapped_model.forward # 使用model的forward方法

        target_modules_list = target_modules
        if isinstance(target_modules_list, str): # 检查target_modules_list是否是str型
            target_modules_list = [target_modules_list] # 如果是字符串则把他变成list

        for module_name, module in self.wrapped_model.named_modules(): # wrapped_model里有很多modules,
            # 遍历所有modules的键名和值
            if not isinstance(module, nn.Linear): # 如果不是linear层 继续
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            weight_data = module.weight.data if keep_original_weights else None
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data if keep_original_weights else None

            new_module = ReLoRaLinear( # 用ReLoRaLinear创建一个新module: new_module，ReLoRaLinear类的定义在下面
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_only=self.lora_only,
                trainable_scaling=self.trainable_scaling,
                quantize=quantize,
                weight_data=weight_data,
                bias_data=bias_data,
                bnb_4bit_use_double_quant=use_double_quant,
            )
            if self.keep_original_weights:
                # make lora'ed network to be exacty the same as the original network at initialization
                nn.init.zeros_(new_module.lora_A.weight) # new_module.lora_A.weight 置0
                assert new_module.lora_A.bias is None # 断言检查 确保new_module.lora_A.bias的值为None
                assert new_module.lora_B.bias is None # 断言检查 确保new_module.lora_B.bias的值为None

            if self.lora_only:
                assert not self.keep_original_weights # 断言检查 确保self.keep_original_weights 为False
                module.weight = None

            del module # 删除module 把老module删除

            parent = self._get_parent(module_name) # 获取每个 module_name 的父对象，并将其赋值给 parent
            module_suffix = module_name.split(".")[-1] # module_name 是包含路径的字符串如my_package.my_module，
            # 这条是以.分割并提取model_name的最后一条字符串
            # module_suffix是一个字符串 是属性名
            setattr(parent, module_suffix, new_module) # 把new_module存为module_suffix对应的属性 即用 new_module 替换之前的 module

        torch.cuda.empty_cache() # 清空未使用的 CUDA 内存

    def _get_parent(self, module_name): # 获取指定模块名称的父模块
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge_and_reinit(self): # 在ReLoRaModel也定义了一个merge_and_reinit。不同于ReLoRaLinear里定义的merge_and_reinit，
        # 这里的用于merge_and_reinit ReLoRaModel里的所有ReLoRaLinear层
        for module in self.modules():
            if isinstance(module, ReLoRaLinear): # 如果 module 是 ReLoRaLinear 则:
                module.merge_and_reinit() # 调用ReLoRaLinear里的merge_and_reinit()

    def save_pretrained(self, path): # 保存pretrained model及其配置
        self.wrapped_model.save_pretrained(path) # 保存pretrained model
        with open(os.path.join(path, "relora_config.json"), "w") as f: # 打开‘relora_config.json’ as f
            json.dump(self._config.__dict__, f, indent=4) # 将 self._config 对象的属性（以字典形式）写入f。
            # indent=4 指定了 JSON 文件的缩进格式，使其更易于阅读。

    @classmethod
    def from_pretrained(cls, path): # 从指定路径加载预训练模型及其配置
        with open(os.path.join(path, "relora_config.json"), "r") as f:
            relora_config = json.load(f) # 加载f

        config = AutoConfig.from_pretrained(path) # 从指定路径加载模型的配置，通常包含模型架构和超参数

        base_model = AutoModelForCausalLM.from_config(config) # 使用加载的配置创建基础的因果语言模型 名为base_model
        if "keep_original" in relora_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {relora_config['keep_original']}")
            relora_config["lora_only"] = not relora_config.pop("keep_original")
            relora_config["keep_original_weights"] = not relora_config["lora_only"]

        if "trainable_scaling" not in relora_config:
            relora_config["trainable_scaling"] = False

        model = cls(base_model, **relora_config) # 使用基础模型和配置字典创建当前类的实例 名为model
        # **是灵活可选参数，可以随便输入
        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f: # 打开保存模型权重的文件
            state_dict = torch.load(f, map_location="cpu") # 加载权重文件到 state_dict 中

        model.wrapped_model.load_state_dict(state_dict, strict=True) # 将加载的权重应用于模型，strict=True 确保权重严格匹配模型
        return model


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
torch.nn.Module和nn.Module基本一样，但是它还包含了一个属性wrapped_model，用于保存被封装的模型。所以ReLoRaModel要用torch.nn.Module
"""
class ReLoRaLinear(nn.Module): # 定义一个ReLoRa线性层
    def __init__(
        self,
        in_features: int, # size
        out_features: int, # size
        r: int, # rank
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        lora_only: bool = False,
        weight_data=None,
        bias_data=None,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
        quantize=False,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """ # 这也是python注释
        nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        if lora_only:
            self.weight = None # 如果是lora_only，则没有传统权重 全靠lora
            self.bias = None
        else:
            # if full model weight + lora weight
            if bias_data is None:
                bias_data = torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True) if bias else None
            self.bias = nn.Parameter(bias_data) if bias else None

            if weight_data is None:
                # note that our trainable weight are W_a and W_b
                weight_data = torch.zeros(out_features, in_features, device=device, dtype=dtype, requires_grad=False)

            if quantize is None: # 支持权重量化，这有助于减少模型的大小和计算需求，特别是在部署到资源受限的设备上时
                self.weight = nn.Parameter(weight_data, requires_grad=False)
                # 如果 quantize 为 None，则使用标准的 PyTorch 权重参数，nn.Parameter 创建一个不可训练(Frozen)的权重（requires_grad=False）
            elif quantize == "4bit":
                self.weight = bnb.nn.Params4bit(
                    weight_data,
                    requires_grad=False,
                    compress_statistics=bnb_4bit_use_double_quant,
                    quant_type=bnb_4bit_quant_type,
                ) # 使用 bnb.nn.Params4bit 来创建一个 4 位量化的权重参数
            elif quantize == "8bit":
                logger.warning("Int8 currently does not support merge_and_reinit! It will fail")
                self.weight = bnb.nn.Int8Params(
                    weight_data,
                    requires_grad=False,
                ) # 发出警告，说明当前的 Int8 不支持 merge_and_reinit 操作
            else:
                raise ValueError(f"Unknown quantize type: {quantize}")
                # 如果没定义权重量化类型 则报错

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.quantize = quantize

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False) # 定义一个线性层，输入维度为 in_features，输出维度为 r
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_B = nn.Linear(r, out_features, bias=False) # 定义一个线性层，输入维度为 r，输出维度为 out_features
            nn.init.zeros_(self.lora_B.weight)
            if trainable_scaling: # ruguo scaling 也是可以train的
                self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
            else:
                self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            if not self.lora_only:
                self.weight.requires_grad = False 
    # __init__结束

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    @torch.no_grad() # 装饰器 和 with torch.no_grad(): 一样
    # 在反向传播时 要计算梯度来更新模型的参数，但是在某些情况下，我们可能不需要计算梯度，例如在评估模型(evaluation)或进行推理(inference)时。
    # 在这种情况下，我们可以使用 torch.no_grad() 上下文管理器来禁用梯度计算。
    # 推理指的是使用训练好的模型来预测新的数据，而评估则是在训练过程中，通过验证集或测试集来评估模型的性能。
    # 在这些阶段我们只关心模型的输出，而不需要计算梯度来更新模型的参数。
    def merge_and_reinit(self): # 定义 merge_and_reinit 函数
        if self.lora_only:
            print("WARNING: Skipping merge and reinit, because only lora parameters are used")
            return

        if not self.quantize:
            self.weight.data += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale() # merge
        elif self.quantize == "4bit":
            self.weight: bnb.nn.Params4bit
            _weight_fp = torch.empty(self.weight.data.shape, dtype=self.lora_B.weight.dtype, device=self.weight.data.device)
            bnbF.dequantize_4bit(self.weight.data, self.weight.quant_state, out=_weight_fp)
            _weight_fp += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale() # merge
            self.weight.data, self.weight.quant_state = bnbF.quantize_4bit(
                _weight_fp,
                quant_type=self.weight.quant_type,
                compress_statistics=self.weight.compress_statistics,
            )
            del _weight_fp
        elif self.quantize == "8bit":
            self.weight: bnb.nn.Int8Params
            _weight_fp = torch.empty(self.weight.data.shape, dtype=torch.bfloat16, device=self.weight.data.device)
            # !out assigned inplace
            bnbF.dequantize_blockwise(self.weight.data, self.self.lora_B.weight.dtype, out=_weight_fp)
            _weight_fp += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale() # merge
            self.weight.data, self.weight.quant_state = bnbF.quantize_blockwise(
                _weight_fp,
                self.weight.quant_state,
                out=self.weight.data,
            )
            del _weight_fp
        else:
            raise ValueError(f"Unknown quantize type: {self.quantize}")

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) # reinit A

        nn.init.zeros_(self.lora_B.weight) # reinit B
        if self.trainable_scaling:
            nn.init.zeros_(self.scaling)

    def forward(self, x: torch.Tensor): # feedforward
        if self.lora_only:
            # just lora
            return self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale() 

        #以下三个result是传统线性层的输出
        if self.quantize == "4bit":
            result = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        elif self.quantize == "8bit":
            result = bnb.matmul(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state) # matmul 是 PyTorch 中的一个函数，用于执行矩阵乘法。
            # x乘以self.weight.t()，然后加上偏置self.bias
        else:
            result = F.linear(x, self.weight, bias=self.bias) # F.linear 是 PyTorch 中的一个函数，用于执行线性变换，
            # 即 y = xA^T + b，其中 x 是输入张量，A 是权重张量，b 是偏置张量。

        # 如果rank>0则把x经过loraA线性层和loraB线性层的结果加到result(传统线性层输出)上
        if self.r > 0:
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale() # add lora
        return result
