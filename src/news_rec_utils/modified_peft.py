# Copyright (c) 2004 The Huggingface Team
# Modified by Ahmed Fahim in 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from peft import LoraConfig
import math
import operator
import warnings
from typing import Union, Any, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from peft.tuners.lora import LoraModel, LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge, BaseTunerLayer
from peft.tuners.lora.layer import dispatch_default
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils.peft_types import PeftType
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import get_pattern_key, get_quantization_config

from peft.tuners.lora.aqlm import dispatch_aqlm
from peft.tuners.lora.awq import dispatch_awq
from peft.tuners.lora.eetq import dispatch_eetq
from peft.tuners.lora.gptq import dispatch_gptq
from peft.tuners.lora.hqq import dispatch_hqq
from peft.tuners.lora.torchao import dispatch_torchao
from peft.tuners.lora.tp_layer import dispatch_megatron
from dataclasses import dataclass
from typing import Optional
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING

__all__ = ["ModifiedLoraConfig"]


@dataclass
class ModifiedLoraConfig(LoraConfig):
    enable_lora: Optional[list[bool]] = None

    def __init__(
        self, enable_lora: list[bool] = [False, False, False], *args, **kwargs
    ):
        self.enable_lora = enable_lora
        super().__init__(*args, **kwargs)


class BaseMergedLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_bias: bool = False,
        r: int = 0,
        enable_lora: list[bool] = [False],
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.enable_lora = enable_lora
        self.lora_bias = lora_bias

        self.weight_A = torch.nn.Parameter(
            torch.empty((sum(enable_lora) * r, in_features), **factory_kwargs)
        )
        self.weight_B = torch.nn.Parameter(
            torch.empty(
                (out_features // len(enable_lora) * sum(enable_lora), r),
                **factory_kwargs,
            )
        )
        if lora_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(
                    out_features // len(enable_lora) * sum(enable_lora),
                    **factory_kwargs,
                )
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask_ind",
            torch.zeros((self.out_features,), dtype=torch.bool).view(
                len(self.enable_lora), -1
            ),
        )
        self.mask_ind[self.enable_lora, :] = True
        self.mask_ind = self.mask_ind.view(-1)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.mask_ind), *x.shape[1:]))
        result[self.mask_ind] = x
        return result

    @property
    def padded_bias(self):
        return self.zero_pad(self.bias) if self.lora_bias else None

    def merge_AB(self):
        delta_w = F.conv1d(
            self.weight_A.unsqueeze(0),
            self.weight_B.unsqueeze(-1),
            groups=sum(self.enable_lora),
        ).squeeze(0)
        return self.zero_pad(delta_w)

    def forward(self, x):
        return F.linear(x, self.merge_AB(), self.padded_bias)


class MergedLinear(nn.Module, LoraLayer):
    adapter_layer_names = (
        "lora_merged",
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
    )

    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.lora_merged = nn.ModuleDict({})
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.enable_lora = {}
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            enable_lora=kwargs["enable_lora"],
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        enable_lora: list[bool],
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}"
            )

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.enable_lora[adapter_name] = enable_lora
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        # self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        # self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias
        self.lora_merged[adapter_name] = BaseMergedLinear(
            self.in_features, self.out_features, lora_bias, r, enable_lora
        )
        # self.lora_merged = nn.ModuleDict(
        #     {
        #         adapter_name: BaseMergedLinear(
        #             self.in_features, self.out_features, lora_bias, r, enable_lora
        #         )
        #     }
        # )
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        # if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
        #     with gather_params_ctx(self.get_base_layer().weight):
        #         self.pissa_init(adapter_name, init_lora_weights)
        # elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
        #     with gather_params_ctx(self.get_base_layer().weight):
        #         self.corda_init(adapter_name, init_lora_weights)
        # elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
        #     with gather_params_ctx(self.get_base_layer().weight):
        #         self.olora_init(adapter_name)
        # elif init_lora_weights == "loftq":
        #     with gather_params_ctx(self.get_base_layer().weight):
        #         self.loftq_init(adapter_name)
        # elif init_lora_weights == "eva":
        #     nn.init.zeros_(self.lora_B[adapter_name].weight)
        # elif init_lora_weights:
        self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # if use_dora:
        #     self.dora_init(adapter_name)
        #     self.use_dora[adapter_name] = True
        # else:
        self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        # if adapter_name in self.lora.keys():
        if init_lora_weights is True:
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(
                self.lora_merged[adapter_name].weight_A, a=math.sqrt(5)
            )
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(
                self.lora_merged[adapter_name].weight_A, std=1 / self.r[adapter_name]
            )
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights}")
        nn.init.zeros_(self.lora_merged[adapter_name].weight_B)
        if self.lora_bias[adapter_name]:
            nn.init.zeros_(self.lora_merged[adapter_name].bias)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_merged.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    # else:
                    #     # handle dora
                    #     # since delta_weight already includes scaling, set it to 1 here
                    #     weight_norm = (
                    #         self.lora_magnitude_vector[active_adapter]
                    #         .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                    #         .detach()
                    #     )
                    #     # We need to cache weight_norm because it has to be based on the original weights. We
                    #     # cannot calculate it on the fly based on the merged weights when unmerging because its a
                    #     # different value
                    #     self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    #     dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    #     dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                    #     orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = (
                            base_layer.bias + self.lora_merged[active_adapter].bias
                        )
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    # else:
                    #     # handle dora
                    #     # since delta_weight already includes scaling, set it to 1 here
                    #     weight_norm = (
                    #         self.lora_magnitude_vector[active_adapter]
                    #         .get_weight_norm(
                    #             base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                    #         )
                    #         .detach()
                    #     )
                    #     # We need to cache weight_norm because it has to be based on the original weights. We
                    #     # cannot calculate it on the fly based on the merged weights when unmerging because its a
                    #     # different value
                    #     self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    #     dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    #     dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                    #     new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                    #     base_layer.weight.data = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_merged[active_adapter].bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_merged.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = (
                        self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    )
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_merged[adapter].weight_B.device
        dtype = self.lora_merged[adapter].weight_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (
            dtype == torch.float16 or dtype == torch.bfloat16
        )

        weight_A = self.lora_merged[adapter].weight_A
        weight_B = self.lora_merged[adapter].weight_B

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (
            self.lora_merged[adapter].merge_AB().float() * self.scaling[adapter]
        )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_merged[adapter].weight_A.data = weight_A.to(dtype)
            self.lora_merged[adapter].weight_B.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_merged.keys():
                    continue
                # lora_A = self.lora_A[active_adapter]
                # lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                lora = self.lora_merged[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora.weight_A.dtype)

                if not self.use_dora[active_adapter]:
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                    result += lora(dropout(x)) * scaling
                # else:
                #     if isinstance(dropout, nn.Identity) or not self.training:
                #         base_result = result
                #     else:
                #         x = dropout(x)
                #         base_result = None

                #     result = result + self.lora_magnitude_vector[active_adapter](
                #         x,
                #         lora_A=lora_A,
                #         lora_B=lora_B,
                #         scaling=scaling,
                #         base_layer=self.get_base_layer(),
                #         base_result=base_result,
                #     )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_qkv_proj(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    enable_lora = getattr(lora_config, "enable_lora", [False, False, False])

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if (
        isinstance(target, torch.nn.Linear)
        and kwargs["target_name"] == "qkv_proj"
        and sum(enable_lora) > 0
    ):
        qkv_kwargs = kwargs.copy()
        qkv_kwargs["enable_lora"] = enable_lora
        new_module = MergedLinear(target, adapter_name, **qkv_kwargs)
    return new_module


class ModifiedLoraModel(LoraModel):
    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": lora_config.lora_bias,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "target_name": target_name,
        }
        # for torchao merging, we need the get_apply_tensor_subclass from the quantization config
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(
                self.model, method=quant_method
            )
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if (
            isinstance(target, LoraLayer)
            and not isinstance(target, AdaLoraLayer)
            and not isinstance(target, MergedLinear)
            and not hasattr(target, "enable_lora")
        ):
            if not hasattr(target, "enable_lora"):
                target.update_layer(
                    adapter_name,
                    r,
                    lora_alpha=alpha,
                    lora_dropout=lora_config.lora_dropout,
                    init_lora_weights=lora_config.init_lora_weights,
                    use_rslora=lora_config.use_rslora,
                    use_dora=lora_config.use_dora,
                    lora_bias=lora_config.lora_bias,
                )
            else:
                target.update_layer(
                    adapter_name,
                    r,
                    lora_alpha=alpha,
                    lora_dropout=lora_config.lora_dropout,
                    init_lora_weights=lora_config.init_lora_weights,
                    use_rslora=lora_config.use_rslora,
                    use_dora=lora_config.use_dora,
                    lora_bias=lora_config.lora_bias,
                    enable_lora=getattr(
                        lora_config, "enable_lora", [False, False, False]
                    ),
                )
        else:
            new_module = self._create_new_module(
                lora_config, adapter_name, target, **kwargs
            )
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        if lora_config._custom_modules:
            # Experimental custom LoRA module support. Allows users to pass a custom mapping for unsupported layer
            # types by impelementing their own LoRA layers.
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None

                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target

                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break

                return new_module

            dispatchers.append(dynamic_dispatch_func)

        dispatchers.append(dispatch_qkv_proj)

        # avoid eager bnb import
        if is_bnb_available():
            from peft.tuners.lora.bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from peft.tuners.lora.bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_eetq,
                dispatch_aqlm,
                dispatch_awq,
                dispatch_gptq,
                dispatch_hqq,
                dispatch_torchao,
                dispatch_megatron,
                dispatch_default,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target, adapter_name, lora_config=lora_config, **kwargs
            )
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`."
            )

        return new_module


PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = ModifiedLoraModel
PEFT_TYPE_TO_CONFIG_MAPPING["LORA"] = ModifiedLoraConfig
PEFT_TYPE_TO_TUNER_MAPPING["LORA"] = ModifiedLoraModel
