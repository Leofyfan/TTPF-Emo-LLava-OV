# from collections import defaultdict
# from dataclasses import dataclass
# from typing import Any, Dict, Optional, Tuple, Union
# import numpy as np

# import torch
# import torch.nn.functional as F
# import torch.utils.checkpoint
# from torch import nn
# from torch.nn import CrossEntropyLoss
# from torch.utils._pytree import tree_map

# from transformers import PretrainedConfig
# from transformers.activations import ACT2FN
# from transformers.modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
# )
# from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import ModelOutput, logging
# import time


# class TTTConfig(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`TTTModel`]. It is used to instantiate an TTT
#     model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
#     defaults will yield a similar configuration to that of the TTT-1B.

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.


#     Args:
#         vocab_size (`int`, *optional*, defaults to 32000):
#             Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
#             `inputs_ids` passed when calling [`LlamaModel`]
#         hidden_size (`int`, *optional*, defaults to 4096):
#             Dimension of the hidden representations.
#         intermediate_size (`int`, *optional*, defaults to 11008):
#             Dimension of the MLP representations.
#         num_hidden_layers (`int`, *optional*, defaults to 32):
#             Number of hidden layers in the Transformer decoder.
#         num_attention_heads (`int`, *optional*, defaults to 32):
#             Number of attention heads for each attention layer in the Transformer decoder.
#         num_key_value_heads (`int`, *optional*):
#             This is the number of key_value heads that should be used to implement Grouped Query Attention. If
#             `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
#             `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
#             converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
#             by meanpooling all the original heads within that group. For more details checkout [this
#             paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
#             `num_attention_heads`.
#         hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
#             The non-linear activation function (function or string) in the decoder.
#         max_position_embeddings (`int`, *optional*, defaults to 2048):
#             The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
#             Llama 2 up to 4096, CodeLlama up to 16384.
#         initializer_range (`float`, *optional*, defaults to 0.02):
#             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#         rms_norm_eps (`float`, *optional*, defaults to 1e-06):
#             The epsilon used by the rms normalization layers.
#         use_cache (`bool`, *optional*, defaults to `True`):
#             Whether or not the model should return the last key/values attentions (not used by all models). Only
#             relevant if `config.is_decoder=True`.
#         pad_token_id (`int`, *optional*):
#             Padding token id.
#         bos_token_id (`int`, *optional*, defaults to 1):
#             Beginning of stream token id.
#         eos_token_id (`int`, *optional*, defaults to 2):
#             End of stream token id.
#         pretraining_tp (`int`, *optional*, defaults to 1):
#             Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
#             document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is
#             necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
#             issue](https://github.com/pytorch/pytorch/issues/76232).
#         tie_word_embeddings (`bool`, *optional*, defaults to `False`):
#             Whether to tie weight embeddings
#         rope_theta (`float`, *optional*, defaults to 10000.0):
#             The base period of the RoPE embeddings.
#         rope_scaling (`Dict`, *optional*):
#             Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
#             strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
#             `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
#             `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
#             these scaling strategies behave:
#             https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
#             experimental feature, subject to breaking API changes in future versions.
#         use_gate (`bool`, *optional*, defaults to `False`): whether use gating in Mamba backbone
#         share_qk (`bool`, *optional*, defaults to `False`): whether share Q/K projection matrix
#         ttt_layer_type (`str`, *optional*, defaults to `"linear"`): ttt block type, "linear" or "mlp", stands for TTT-Linear and TTT-MLP
#         ttt_base_lr (`float`, *optional*, defaults to 1.0): base learning rate for TTT learner
#         mini_batch_size (`int`, *optional*, defaults to 16): mini batch size for TTT
#         warmup_steps (`int`, *optional*, defaults to 1000): number of warmup steps for learning rate
#         warmup_start_lr (`float`, *optional*, defaults to 1e-6): starting learning rate for warmup
#         warmup_end_lr (`float`, *optional*, defaults to 1e-5): ending learning rate for warmup
#         pre_conv (`bool`, *optional*, defaults to `False`): whether use conv before TTT
#         conv_kernel (`int`, *optional*, defaults to 4): kernel size of the conv layer
#         scan_checkpoint_group_size (`int`, *optional*, defaults to 0):
#             gradient checkpoint group size on seq dimension, 0 means no checkpointing.
#             In JAX implementation, we set it 4, which means we group 4 mini-batches together in 1 gradient checkpointg to save memory.


#     ```python
#     >>> from . import TTTModel, TTTConfig

#     >>> # Initializing a TTT ttt-1b style configuration
#     >>> configuration = TTTConfig()

#     >>> # Initializing a model from the ttt-1b style configuration
#     >>> model = TTTModel(configuration)

#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""

#     model_type = "ttt"

#     def __init__(
#         self,
#         vocab_size=32000,
#         hidden_size=2048,
#         intermediate_size=5504,
#         num_hidden_layers=24,
#         num_attention_heads=32,
#         hidden_act="silu",
#         max_position_embeddings=2048,
#         initializer_range=0.02,
#         rms_norm_eps=1e-6,
#         use_cache=False,
#         pad_token_id=None,
#         bos_token_id=1,
#         eos_token_id=2,
#         pretraining_tp=1,
#         tie_word_embeddings=True,
#         rope_theta=10000.0,
#         use_gate=False,
#         share_qk=False,
#         ttt_layer_type="linear",
#         ttt_base_lr=1.0,
#         mini_batch_size=16,
#         warmup_steps=1000,
#         warmup_start_lr=1e-6,
#         warmup_end_lr=5e-4,
#         pre_conv=False,
#         conv_kernel=4,
#         scan_checkpoint_group_size=0,
#         output_proj_dim = 768,
#         **kwargs,
#     ):
#         self.vocab_size = vocab_size
#         self.max_position_embeddings = max_position_embeddings
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads

#         self.hidden_act = hidden_act
#         self.initializer_range = initializer_range
#         self.rms_norm_eps = rms_norm_eps
#         self.pretraining_tp = pretraining_tp
#         self.use_cache = use_cache
#         self.rope_theta = rope_theta

#         self.use_gate = use_gate
#         self.share_qk = share_qk
#         self.ttt_layer_type = ttt_layer_type
#         self.ttt_base_lr = ttt_base_lr
#         self.mini_batch_size = mini_batch_size
#         self.warmup_steps = warmup_steps
#         self.warmup_start_lr = warmup_start_lr
#         self.warmup_end_lr = warmup_end_lr

#         self.pre_conv = pre_conv
#         self.conv_kernel = conv_kernel
#         self.scan_checkpoint_group_size = scan_checkpoint_group_size
        
#         # output_dim
#         self.output_proj_dim = output_proj_dim

#         super().__init__(
#             pad_token_id=pad_token_id,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id,
#             tie_word_embeddings=tie_word_embeddings,
#             **kwargs,
#         )


# ########################
# ### Backbone Modules ###
# ########################


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def permute_qk(q, k):
#     # NOTE: EasyLM and transformers use different method to compute rotary emebdding
#     # we manually reorder the dim here to match our JAX implementation
#     # which may not be optimal for speed
#     # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
#     bsz, num_head, seq_len, head_dim = q.shape
#     q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
#     k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

#     return q, k


# def undo_permute_qk(q, k):
#     # NOTE: EasyLM and transformers use different method to compute rotary emebdding
#     # we manually undo the reorder the dim here to match our JAX implementation
#     # which may not be optimal for speed
#     # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
#     bsz, num_head, seq_len, head_dim = q.shape
#     q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
#     k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

#     return q, k


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)


# class SwiGluMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(self, x):
#         if self.config.pretraining_tp > 1:
#             slice = self.intermediate_size // self.config.pretraining_tp
#             gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
#             up_proj_slices = self.up_proj.weight.split(slice, dim=0)
#             down_proj_slices = self.down_proj.weight.split(slice, dim=1)

#             gate_proj = torch.cat(
#                 [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
#                 dim=-1,
#             )
#             up_proj = torch.cat(
#                 [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
#                 dim=-1,
#             )

#             intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
#             down_proj = [
#                 F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
#             ]
#             down_proj = sum(down_proj)
#         else:
#             down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

#         return down_proj


# class RotaryEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim,
#         max_position_embeddings=16,
#         base=10000,
#         device=None,
#         scaling_factor=1.0,
#     ):
#         super().__init__()
#         self.scaling_factor = scaling_factor
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()
#         # Force float32 since bfloat16 loses precision on long contexts
#         # See https://github.com/huggingface/transformers/pull/29285
#         device_type = x.device.type
#         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cuda:0"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# class Conv(nn.Module):
#     def __init__(self, config, layer_idx):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx

#         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.conv = nn.Conv1d(
#             config.hidden_size,
#             config.hidden_size,
#             bias=True,
#             kernel_size=config.conv_kernel,
#             groups=config.hidden_size,
#             padding=config.conv_kernel - 1,
#         )

#     def __call__(self, hidden_states, cache_params=None):
#         seq_len = hidden_states.shape[1]
#         hidden_states = self.norm(hidden_states)
#         # [B, C, L]
#         hidden_states = hidden_states.transpose(1, 2)

#         if causal_conv1d_fn is None:
#             if cache_params is not None:
#                 if cache_params.seqlen_offset > 0:
#                     conv_state = cache_params.conv_states_dic["pre_conv"][self.layer_idx]
#                     conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
#                     conv_state[:, :, -1] = hidden_states[:, :, 0]
#                     cache_params.conv_states_dic["pre_conv"][self.layer_idx].copy_(conv_state)
#                     hidden_states = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1)
#                     hidden_states += self.conv.bias
#                     hidden_states = hidden_states.unsqueeze(-1)
#                 else:
#                     conv_state = nn.functional.pad(
#                         hidden_states,
#                         (self.config.conv_kernel - hidden_states.shape[-1], 0),
#                     )
#                     cache_params.conv_states_dic["pre_conv"][self.layer_idx].copy_(conv_state)
#                     hidden_states = self.conv(hidden_states)[..., :seq_len]
#             else:
#                 hidden_states = self.conv(hidden_states)[..., :seq_len]
#         else:
#             conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
#             if cache_params is not None and cache_params.seqlen_offset > 0:
#                 hidden_states = causal_conv1d_update(
#                     hidden_states.squeeze(-1),
#                     cache_params.conv_states_dic["pre_conv"][self.layer_idx],
#                     conv_weights,
#                     self.conv.bias,
#                     None,
#                 )
#                 hidden_states = hidden_states.unsqueeze(-1)
#             else:
#                 if cache_params is not None:
#                     conv_states = nn.functional.pad(
#                         hidden_states,
#                         (self.config.conv_kernel - hidden_states.shape[-1], 0),
#                     )
#                     cache_params.conv_states_dic["pre_conv"][self.layer_idx].copy_(conv_states)
#                 hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv.bias, activation=None)

#         # [B, L, C]
#         hidden_states = hidden_states.transpose(1, 2)

#         return hidden_states


# #########################
# ### TTT Layer Modules ###
# #########################


# def scan(f, init, xs, out, checkpoint_group=0):
#     """Minic jax.lax.scan function."""
#     carry = init
#     if isinstance(xs, dict):
#         num_items = len(next(iter(xs.values())))
#     else:
#         num_items = len(xs[0])
        
#     # print("============ scan debug =============")
#     # # print(xs.items())
#     # print(f"num_items: {num_items}")
#     # print(f"checkpoint_group: {checkpoint_group}")

#     def scan_fn(carry, i_start, i_end):
#         for i in range(i_start, i_end):
#             if isinstance(xs, dict):
#                 x = {key: tensor[i] for key, tensor in xs.items()}
#             else:
#                 x = [x[i] for x in xs]
                
#             # for key, tensor in xs.items():
#             #     print(key)
#             #     print(tensor.shape)
#             # print(f"carry: {carry['W1_states'].shape}")
#             # print(f"x: {x['XQ'].shape}")
#             carry, y = f(carry, x)
#             out[i] = y
#         return carry

#     if checkpoint_group > 0:
#         ckpt_every_n = num_items // checkpoint_group
#         for k in range(0, num_items, ckpt_every_n):
#             carry = torch.utils.checkpoint.checkpoint(
#                 scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
#             )
#     else:
#         carry = scan_fn(carry, 0, num_items)

#     return carry, out


# def ln_fwd(x, gamma, beta, eps=1e-6):
#     "Batch forward for LayerNorm."

#     # Mean and variance computation
#     mu = x.mean(dim=-1, keepdim=True)
#     var = x.var(dim=-1, keepdim=True, unbiased=False)

#     # Normalization
#     std = torch.sqrt(var + eps)
#     x_hat = (x - mu) / std

#     # Scale and shift
#     y = gamma * x_hat + beta

#     return y


# def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
#     "Batch backward for LayerNorm fused with L2 loss."
#     D = x.shape[-1]
    
    
#     # print("====== ln_fused_l2_bwd =====")
#     # print(f"x: {x.shape}")
#     # print(f"l2_target: {l2_target.shape}")
#     # print(f"gamma: {gamma.shape}")
#     # print(f"beta: {beta.shape}")

#     # Mean and variance computation
#     mu = x.mean(dim=-1, keepdim=True)
#     var = x.var(dim=-1, keepdim=True, unbiased=False)

#     # Normalization
#     std = torch.sqrt(var + eps)
#     x_hat = (x - mu) / std

#     # Scale and shift
#     y = gamma * x_hat + beta

#     grad_output = y - l2_target
#     grad_x_hat = grad_output * gamma
#     z = (
#         (1.0 / D)
#         * (
#             D * grad_x_hat
#             - grad_x_hat.sum(dim=-1, keepdim=True)
#             - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
#         )
#         / std
#     )

#     return z


# # Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
# def gelu_bwd(x):
#     tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
#     ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
#     return ff

# def compute_feature_correlation(A, B, eps=1e-8):
#     """
#     compute the correlation matrix of two tensors in feature dimension (vectorized implementation)
    
#     Args:
#     - A: tensor with shape [l, f_a]
#     - B: tensor with shape [l, f_b]
    
#     Returns:
#     - C: tensor with shape [f_a, f_b]
#     """
#     # normalize each feature of A
#     A_centered = A - A.mean(dim=0, keepdim=True)
#     A_normalized = A_centered / (torch.sqrt(torch.sum(A_centered**2, dim=0, keepdim=True)) + eps)
    
#     # normalize each feature of B
#     B_centered = B - B.mean(dim=0, keepdim=True)
#     B_normalized = B_centered / (torch.sqrt(torch.sum(B_centered**2, dim=0, keepdim=True)) + eps)
    
#     # compute the correlation matrix
#     C = torch.matmul(A_normalized.t(), B_normalized)
    
#     return C

# def compute_corss_loss(C, k):

#     # feature dimension
#     f = C.shape[0]
    
#     # extract the corresponding parts of the correlation matrix
#     C_k = C[:k, :k] 
#     C_rest = C[k:, k:]
    
#     # compute the L_com loss
#     com_diag_mask = torch.eye(k, device=C.device)
#     com_diag_loss = torch.sum(((1 - C_k) * com_diag_mask) ** 2) / k
#     com_non_diag_mask = 1 - com_diag_mask
#     com_non_diag_loss = torch.sum((C_k * com_non_diag_mask)**2) / (k * (k - 1))
#     L_com = com_diag_loss + com_non_diag_loss
    
#     # compute the L_uni loss
#     uni_diag_mask = torch.eye(f - k, device=C.device)
#     uni_diag_loss = torch.sum((C_rest * uni_diag_mask) ** 2) / (f - k)
#     uni_non_diag_mask = 1 - uni_diag_mask
#     uni_non_diag_loss = torch.sum((C_rest * uni_non_diag_mask) ** 2) / ((f - k) * (f - k - 1))
#     L_uni = uni_diag_loss + uni_non_diag_loss
    
#     # compute the cross loss
#     L_cross = L_com + L_uni
    
#     return L_com, L_uni, L_cross

# def min_kcom(f1, f2):
#     _, f1_len = f1.shape
#     _, f2_len = f2.shape
#     corr_matrix = compute_feature_correlation(f1, f2)
    
#     min_len = min(f1_len, f2_len)
#     start_k = min(1, min_len // 20)
#     end_k = max(start_k, min_len - 100)
    
#     min_k = start_k
#     min_cross_loss = 1e9
    
#     for k in np.unique(np.round(np.linspace(start_k, end_k, 5)).astype(int)).tolist():
#         cur_cross_loss, _, _ = compute_corss_loss(corr_matrix, k)
#         if cur_cross_loss < min_cross_loss:
#             min_cross_loss = cur_cross_loss
#             min_k = k
    
#     return min_k

# class TTTCache:
#     """
#     TTTCache is a data structure that holds the last hidden states and gradients for the TTT layer.

#     Arguments:
#         model: TTTModel
#         batch_size: int

#     Attributes:
#         seqlen_offset: int
#         mini_batch_size: int
#         params_dict: Dict[str, Dict[int, torch.Tensor]]  *_states, *_grad -> # layer_idx -> [batch_size, ...]
#         conv_states_dic: Dict[str, Dict[int, torch.Tensor]]  *_states -> # layer_idx -> [batch_size, ...]

#     """

#     def __init__(self, model, batch_size: int):
#         config = model.config
#         self.seqlen_offset = 0
#         self.mini_batch_size = config.mini_batch_size

#         self.ttt_params_dict = defaultdict(dict)
#         if "linear" in config.ttt_layer_type:
#             self.ttt_param_names = ["W1", "b1"]
#         elif "mlp" in config.ttt_layer_type:
#             self.ttt_param_names = ["W1", "b1", "W2", "b2"]
#         else:
#             raise ValueError(f"TTT Layer Type {config.ttt_layer_type} not supported yet")

#         self.conv_states_dic = defaultdict(dict)
#         logger.info(f"Creating cache of size: {batch_size}")
#         for layer_idx in range(config.num_hidden_layers):
#             for name in self.ttt_param_names:
#                 weight = getattr(model.layers[layer_idx].seq_modeling_block, name)
#                 tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(model.device)
#                 self.ttt_params_dict[f"{name}_states"][layer_idx] = tiled_weight
#                 # for decoding, we need to store the gradients as well
#                 self.ttt_params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

#             if config.pre_conv:
#                 self.conv_states_dic["pre_conv"][layer_idx] = torch.zeros(
#                     batch_size,
#                     config.hidden_size,
#                     config.conv_kernel,
#                     device=model.device,
#                 )
#             if config.share_qk:
#                 self.conv_states_dic["ttt_conv_q"][layer_idx] = torch.zeros(
#                     batch_size,
#                     config.hidden_size,
#                     config.conv_kernel,
#                     device=model.device,
#                 )
#                 self.conv_states_dic["ttt_conv_k"][layer_idx] = torch.zeros(
#                     batch_size,
#                     config.hidden_size,
#                     config.conv_kernel,
#                     device=model.device,
#                 )

#     def update(self, py_tree, layer_idx, seq_len):
#         if seq_len % self.mini_batch_size == 0:
#             # copy last mini-batch states, clear gradients
#             for name in self.ttt_param_names:
#                 self.ttt_params_dict[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
#                 self.ttt_params_dict[f"{name}_grad"][layer_idx].zero_()
#         elif seq_len < self.mini_batch_size:
#             if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.mini_batch_size != 0:
#                 raise ValueError("fractional update not supported yet.")
#             if (seq_len + self.seqlen_offset) % self.mini_batch_size == 0:
#                 # copy last mini-batch states, clear gradients
#                 for name in self.ttt_param_names:
#                     self.ttt_params_dict[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
#                     self.ttt_params_dict[f"{name}_grad"][layer_idx].zero_()
#             else:
#                 # copy gradients for the next update
#                 for name in self.ttt_param_names:
#                     self.ttt_params_dict[f"{name}_grad"][layer_idx].copy_(py_tree[f"{name}_grad"])
#         else:
#             raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")

#     def ttt_params_to_dict(self, layer_idx):
#         return {name: self.ttt_params_dict[name][layer_idx] for name in self.ttt_params_dict}


# class TTTBase(nn.Module):
#     def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         if layer_idx is None:
#             logger.warning_once(
#                 f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
#                 "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
#                 "when creating this class."
#             )

#         self.width = config.hidden_size
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.width // self.num_heads
#         self.mini_batch_size = config.mini_batch_size
#         self.output_proj_dim = config.output_proj_dim

#         # token_idx is a scale factor that scale the summation in Eqn. 4
#         token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
#         self.register_buffer("token_idx", token_idx, persistent=False)
#         # make the scale factor learnable
#         self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

#         self.share_qk = config.share_qk
#         self.conv_kernel = config.conv_kernel
#         self._init_qkvo_proj()
#         self._init_rope()
#         # Learnable eta in Sec. 2.7
#         self._init_ttt_lr_gate()
#         self._init_ttt_ln()

#         # use gating as in Mamba backbone
#         self.use_gate = config.use_gate
#         if self.use_gate:
#             self.g_proj = nn.Linear(self.width, self.width, bias=False)

#         self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

#     def _init_qkvo_proj(self):
#         self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
#         # we share Q/K projection when using Mamba backbone
#         if not self.share_qk:
#             self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
#         self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
#         self.o_proj = nn.Linear(self.width, self.output_proj_dim, bias=False)

#         # after share Q/K projection, we use different conv layers for Q and K
#         if self.share_qk:
#             self.conv_q = nn.Conv1d(
#                 self.hidden_size,
#                 self.hidden_size,
#                 bias=True,
#                 kernel_size=self.conv_kernel,
#                 groups=self.hidden_size,
#                 padding=self.conv_kernel - 1,
#             )
#             self.conv_k = nn.Conv1d(
#                 self.hidden_size,
#                 self.hidden_size,
#                 bias=True,
#                 kernel_size=self.conv_kernel,
#                 groups=self.hidden_size,
#                 padding=self.conv_kernel - 1,
#             )

#     def _init_rope(self):
#         self.rope_theta = self.config.rope_theta
#         self.rotary_emb = RotaryEmbedding(
#             self.head_dim,
#             max_position_embeddings=self.mini_batch_size,
#             base=self.rope_theta,
#         )

#     def _init_ttt_lr_gate(self):
#         # [width, 1]
#         linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
#         # prepending head dim -> [num_heads, width, 1]
#         self.learnable_ttt_lr_weight = nn.Parameter(
#             torch.stack(
#                 [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
#                 dim=0,
#             )
#         )
#         linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
#         # init bias to 0 following original JAX impl.
#         # [num_heads, 1]
#         self.learnable_ttt_lr_bias = nn.Parameter(
#             torch.stack(
#                 [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
#                 dim=0,
#             )
#         )

#     def _init_ttt_ln(self):
#         ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
#         # prepending head dim -> [num_heads, width]
#         self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
#         ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
#         self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))
#         self.f_ttt_norm_weight = nn.Parameter(nn.LayerNorm(self.hidden_size).weight.data)
#         self.f_ttt_norm_bias = nn.Parameter(nn.LayerNorm(self.hidden_size).bias.data)

#     def get_qkv_projections(self, hidden_states, cache_params: Optional[TTTCache] = None):
#         if self.share_qk:
#             xq, XV = self.q_proj(hidden_states), self.v_proj(hidden_states)
#             seq_len = xq.shape[1]
#             xq = xq.transpose(1, 2)
#             if causal_conv1d_fn is None:
#                 if cache_params is not None:
#                     if cache_params.seqlen_offset > 0:
#                         conv_q_state = cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx]
#                         conv_q_state = torch.roll(conv_q_state, shifts=-1, dims=-1)
#                         conv_q_state[:, :, -1] = xq[:, :, 0]
#                         cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx].copy_(conv_q_state)
#                         XQ = torch.sum(conv_q_state * self.conv_q.weight[:, 0, :], dim=-1)
#                         XQ += self.conv_q.bias
#                         XQ = XQ.unsqueeze(-1)

#                         conv_k_state = cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx]
#                         conv_k_state = torch.roll(conv_k_state, shifts=-1, dims=-1)
#                         conv_k_state[:, :, -1] = xq[:, :, 0]
#                         cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx].copy_(conv_k_state)
#                         XK = torch.sum(conv_k_state * self.conv_k.weight[:, 0, :], dim=-1)
#                         XK += self.conv_k.bias
#                         XK = XK.unsqueeze(-1)
#                     else:
#                         conv_q_state = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
#                         cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx].copy_(conv_q_state)
#                         XQ = self.conv_q(xq)[..., :seq_len]
#                         conv_k_state = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
#                         cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx].copy_(conv_k_state)
#                         XK = self.conv_k(xq)[..., :seq_len]
#                 else:
#                     XQ = self.conv_q(xq)[..., :seq_len]
#                     XK = self.conv_k(xq)[..., :seq_len]
#             else:
#                 conv_q_weights = self.conv_q.weight.view(self.conv_q.weight.size(0), self.conv_q.weight.size(2))
#                 conv_k_weights = self.conv_k.weight.view(self.conv_k.weight.size(0), self.conv_k.weight.size(2))
#                 if cache_params is not None and cache_params.seqlen_offset > 0:
#                     XQ = causal_conv1d_update(
#                         xq.squeeze(-1),
#                         cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx],
#                         conv_q_weights,
#                         self.conv_q.bias,
#                         None,
#                     )
#                     XQ = XQ.unsqueeze(-1)
#                     XK = causal_conv1d_update(
#                         xq.squeeze(-1),
#                         cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx],
#                         conv_k_weights,
#                         self.conv_k.bias,
#                         None,
#                     )
#                     XK = XK.unsqueeze(-1)
#                 else:
#                     if cache_params is not None:
#                         conv_q_states = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
#                         cache_params.conv_states_dic["ttt_conv_q"][self.layer_idx].copy_(conv_q_states)
#                         conv_k_states = nn.functional.pad(xq, (self.config.conv_kernel - xq.shape[-1], 0))
#                         cache_params.conv_states_dic["ttt_conv_k"][self.layer_idx].copy_(conv_k_states)
#                     XQ = causal_conv1d_fn(xq, conv_q_weights, self.conv_q.bias, activation=None)
#                     XK = causal_conv1d_fn(xq, conv_k_weights, self.conv_k.bias, activation=None)

#             XQ = XQ.transpose(1, 2)
#             XK = XK.transpose(1, 2)
#         else:
#             XQ, XK, XV = (
#                 self.q_proj(hidden_states),
#                 self.k_proj(hidden_states),
#                 self.v_proj(hidden_states),
#             )
#         return XQ, XK, XV

#     def _split_heads(self, hidden_states):
#         return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

#     def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
#         # [B, num_heads, num_mini_batch, mini_batch_size, 1]
#         ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
#             1, -1, 1, 1, 1
#         )
#         ttt_lr = F.sigmoid(ttt_lr)

#         # [B, num_heads, num_mini_batch, 1, mini_batch_size]
#         ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
#         ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim

#         # [B, L]
#         token_idx = self.token_idx + self.learnable_token_idx
#         token_idx = token_idx[mini_batch_step_offset : mini_batch_step_offset + mini_batch_size]

#         # token idx should be greast than 0
#         token_idx = torch.clamp_min(token_idx, 0.0)

#         # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
#         # [B, num_heads, num_mini_batch, mini_batch_size, 1]
#         token_eta = torch.broadcast_to(
#             token_idx.reshape(1, 1, 1, mini_batch_size, 1),
#             (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
#         )

#         return token_eta, ttt_lr_eta

#     def apply_gate(self, hidden_states, ttt_output):
#         y = self.g_proj(hidden_states)
#         # use 'tanh' approximation for matching JAX impl.
#         y = F.gelu(y, approximate="tanh")
#         output = y * ttt_output
#         return output

#     def get_ttt_inputs(self, inputs, mini_batch_size, cache_params):
#         XQ = inputs["XQ"]
#         XK = inputs["XK"]
#         XV = inputs["XV"]
#         X = inputs["X"]
#         B, L, C = X.shape
#         num_mini_batch = L // mini_batch_size
#         # [B ,num_mini_batch, mini_batch_size, C]
#         X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)

#         XQ = XQ.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
#         XK = XK.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
#         XV = XV.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)

#         if cache_params is not None:
#             mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
#         else:
#             mini_batch_step_offset = 0
#         token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
#         eta = token_eta * ttt_lr_eta
#         # decouple token_coeff and ilr_coeff for decoding
#         inputs = {
#             "XQ": XQ,
#             "XK": XK,
#             "XV": XV,
#             "eta": eta,
#             "token_eta": token_eta,
#             "ttt_lr_eta": ttt_lr_eta,
#         }
#         return inputs

#     def ttt(
#         self,
#         inputs,
#         mini_batch_size,
#         last_mini_batch_params_dict,
#         cache_params: Optional[TTTCache] = None,
#     ):
#         raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         cache_params: Optional[TTTCache] = None,
#     ):
#         B, L = hidden_states.shape[:2]
#         reminder_len = L % self.mini_batch_size
#         num_mini_batch = L // self.mini_batch_size
#         last_mini_batch_params_dict = None
        
#         # print(f"hidden_states: {hidden_states.shape}")
#         # print(f"num_heads: {self.num_heads}  head_dim: {self.head_dim}")

#         XQ, XK, XV = self.get_qkv_projections(hidden_states, cache_params=cache_params)

#         # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
#         XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
#         XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
#         XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

#         cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)

#         # permute_qk and undo_permute_qk is just for aligning pytorch with jax pre-training
#         XQ, XK = permute_qk(XQ, XK)
#         XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
#         XQ, XK = undo_permute_qk(XQ, XK)

#         output_hidden_states = []
#         # when input sequence length is not a multiple of mini_batch_size
#         # we need to compute them seperately, when computing the reminder,
#         # we will need the last_mini_batch_params_dict to continue TTT learning
#         if num_mini_batch > 0:
#             inputs = {
#                 "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
#                 "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
#                 "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
#                 "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
#             }
#             output_mod, last_mini_batch_params_dict = self.ttt(
#                 self.get_ttt_inputs(inputs, self.mini_batch_size, cache_params),
#                 mini_batch_size=self.mini_batch_size,
#                 last_mini_batch_params_dict=last_mini_batch_params_dict,
#                 cache_params=cache_params,
#             )
#             output_hidden_states.append(output_mod)
#         if reminder_len > 0:
#             inputs = {
#                 "XQ": XQ[:, :, -reminder_len:],
#                 "XK": XK[:, :, -reminder_len:],
#                 "XV": XV[:, :, -reminder_len:],
#                 "X": hidden_states[:, -reminder_len:],
#             }
#             output_reminder, _ = self.ttt(
#                 self.get_ttt_inputs(inputs, reminder_len, cache_params),
#                 mini_batch_size=reminder_len,
#                 last_mini_batch_params_dict=last_mini_batch_params_dict,
#                 cache_params=cache_params,
#             )
#             output_hidden_states.append(output_reminder)
            
#         # print(f"len(output_hidden_states): {len(output_hidden_states)}")
#         # for i in range(len(output_hidden_states)):
#         #     print(f"output_hidden_state_{i} : {output_hidden_states[i].shape}")

#         output_hidden_states = torch.cat(output_hidden_states, dim=1)
#         output_hidden_states = self.post_norm(output_hidden_states)
#         if self.use_gate:
#             output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
#         output_hidden_states = self.o_proj(output_hidden_states)

#         # print(f"output_hidden_states: {output_hidden_states.shape}")
#         return output_hidden_states


# class TTTLinear(TTTBase):
#     def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
#         super().__init__(config, layer_idx)
#         # TTT model initialization for TTT-Linear
#         self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
#         self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
#         self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.hidden_size, self.hidden_size)))
#         self.b2 = nn.Parameter(torch.zeros(1, self.hidden_size))

#     def ttt(
#         self,
#         inputs,
#         mini_batch_size,
#         last_mini_batch_params_dict,
#         cache_params: Optional[TTTCache] = None,
#     ):
#         if mini_batch_size is None:
#             mini_batch_size = self.mini_batch_size
            
#         # print("=================  ttt debug =================")
#         # print(f"XV: {inputs["XV"].shape}")
#         # print(f"XQ: {inputs["XQ"].shape}")
#         # print(f"XK: {inputs["XK"].shape}")

#         # in this case, we are decoding
#         if last_mini_batch_params_dict is None and cache_params is not None:
#             last_mini_batch_params_dict = cache_params.ttt_params_to_dict(self.layer_idx)

#         # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
#         B = inputs["XV"].shape[0]
#         num_mini_batch = inputs["XV"].shape[2]
#         L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
#         device = inputs["XV"].device
#         dtype = inputs["XV"].dtype
        
#         # print(f"last_mini_batch_params_dict: {last_mini_batch_params_dict}")

#         # NOTE:
#         # for prefilling, we will always use dual form for faster computation
#         # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
#         # since we need store the gradient for the next mini-batch computation
#         use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

#         def compute_mini_batch(params_dict, inputs):
#             # [B, nh, f, f], nh=num_heads, f=head_dim
#             W1_init = params_dict["W1_states"]
#             # [B, nh, 1, f]
#             b1_init = params_dict["b1_states"]
            
#             # W2_init = params_dict["W2_states"]
#             # b2_init = params_dict["b2_states"]

#             # [B,nh,K,f], K=mini_batch_size
#             XQ_mini_batch = inputs["XQ"]
#             XV_mini_batch = inputs["XV"]
#             XK_mini_batch = inputs["XK"]
#             # [B, nh, K, 1]
#             eta_mini_batch = inputs["eta"]
#             token_eta_mini_batch = inputs["token_eta"]
#             ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]
        
#             X1 = XK_mini_batch
#             # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
#             Z1 = X1 @ W1_init + b1_init
#             reconstruction_target = XV_mini_batch - XK_mini_batch

#             ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim).to(XK_mini_batch.device)
#             ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim).to(XK_mini_batch.device)
#             # f_ln_weight = self.f_ttt_norm_weight.reshape(1, self.hidden_size).to(XK_mini_batch.device)
#             # f_ln_bias = self.f_ttt_norm_bias.reshape(1, self.hidden_size).to(XK_mini_batch.device)
            
#             # frames, _, sl, _ = XK_mini_batch.shape
#             # # [B, K, hideen_size]
#             # X2 = XK_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
            
#             # cur_grad_W2 = torch.zeros_like(W2_init)
#             # cur_grad_b2 = torch.zeros_like(b2_init)
            
#             # print(f"frames: {frames}")
#             # print(f"X2_shape: {X2.shape}")
       
#             # for f_id in range(frames-1):
#             #     f1_copy = X2[f_id].detach().clone()
#             #     f2_copy = X2[f_id + 1].detach().clone()
#             #     k_com = min_kcom(f1_copy, f2_copy)
#             #     # print(f"k_com: {k_com}")
#             #     f1com_f2uni = torch.cat([f1_copy[:, 0 : k_com], f2_copy[:, k_com : ]], dim=-1)
#             #     f2com_f1uni = torch.cat([f2_copy[:, 0 : k_com], f1_copy[:, k_com : ]], dim=-1)
#             #     # cross frame reconstruction
#             #     rec_f2 = f1com_f2uni @ W2_init + b2_init
#             #     rec_f1 = f2com_f1uni @ W2_init + b2_init
#             #     # ln backfoward
#             #     grad_Z2 = ln_fused_l2_bwd(rec_f1, f1_copy, f_ln_weight, f_ln_bias)
#             #     grad_Z3 = ln_fused_l2_bwd(rec_f2, f2_copy, f_ln_weight, f_ln_bias)
#             #     cur_grad_W2 += f2com_f1uni.t() @ grad_Z2
#             #     cur_grad_W2 += f1com_f2uni.t() @ grad_Z3
#             #     cur_grad_b2 += grad_Z2.sum(dim=0, keepdim=True)
#             #     cur_grad_b2 += grad_Z3.sum(dim=0, keepdim=True)
            
#             # update W2 b2
#             # 计算当前步数
#             # current_step = getattr(self, '_step_counter', 0)
#             # self._step_counter = current_step + 1
            
#             # # 计算 warmup 学习率
#             # if current_step < self.config.warmup_steps:
#             #     progress = current_step / self.config.warmup_steps
#             #     eta_W2 = self.config.warmup_start_lr + progress * (self.config.warmup_end_lr - self.config.warmup_start_lr)
#             # else:
#             #     eta_W2 = self.config.warmup_end_lr
                
#             # grad_W2_last = cur_grad_W2 / frames
#             # grad_b2_last = cur_grad_b2 / frames
#             # W2_last = W2_init - eta_W2 * grad_W2_last
#             # b2_last = b2_init - eta_W2 * grad_b2_last
#             # # [B, K, hideen_size]
#             # f_XQ_mini_batch = XQ_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
#             # # # [B, K, K]
#             # # Attn2 = torch.tril(torch.matmul(f_XQ_mini_batch, X2.transpose(-2, -1))) / self.hidden_size
#             # Z2_bar = torch.matmul(f_XQ_mini_batch, W2_last) + b2_last
#             # Z2_bar = ln_fwd(Z2_bar, f_ln_weight, f_ln_bias)
#             # # [B,nh,K,f]
#             # Z2_bar = Z2_bar.reshape(frames, sl, self.num_heads, self.head_dim).transpose(2, 1)
           
#             # print(f"use_dual_form: {use_dual_form}")
           
#             # [B,nh,K,f]
#             grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

#             if use_dual_form:
#                 # [B,nh,K,K]
#                 Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                
#                 # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
#                 b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
#                 # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
#                 Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

#                 last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
#                 # print(f"eta_mini_batch.shape: {eta_mini_batch.shape}")
#                 # print(f"last_eta_mini_batch: {last_eta_mini_batch.shape}")
#                 # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
#                 W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
#                 # [B,nh,1,f]
#                 b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
#                 grad_W1_last = torch.zeros_like(W1_last)
#                 grad_b1_last = torch.zeros_like(b1_last)
#             else:
#                 ttt_lr_eta_mini_batch = torch.broadcast_to(
#                     ttt_lr_eta_mini_batch,
#                     (
#                         *ttt_lr_eta_mini_batch.shape[:2],
#                         mini_batch_size,
#                         mini_batch_size,
#                     ),
#                 )

#                 # [B, nh, K, f, f]
#                 grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
#                 grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
#                 grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
#                 # [B, nh, K, f]
#                 grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
#                 grad_b1 = grad_b1 + params_dict["b1_grad"]

#                 W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
#                 b1_bar = b1_init - grad_b1 * token_eta_mini_batch

#                 # [B, nh, K, 1, f] @ [B, nh, K, f, f]
#                 Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

#                 W1_last = W1_bar[:, :, -1]
#                 b1_last = b1_bar[:, :, -1:]
#                 grad_W1_last = grad_W1[:, :, -1]
#                 grad_b1_last = grad_b1[:, :, -1:]

#             Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

#             # XQW_mini_batch = XQ_mini_batch + Z1_bar + Z2_bar
#             XQW_mini_batch = XQ_mini_batch + Z1_bar

#             last_param_dict = {
#                 "W1_states": W1_last,
#                 "b1_states": b1_last,
#                 "W1_grad": grad_W1_last,
#                 "b1_grad": grad_b1_last
#                 # "W2_states": W2_last,
#                 # "b2_states": b2_last,
#                 # "W2_grad": grad_W2_last,
#                 # "b2_grad": grad_b2_last,
#             }
#             return last_param_dict, XQW_mini_batch

#         if last_mini_batch_params_dict is not None:
#             init_params_dict = last_mini_batch_params_dict
#         else:
#             init_params_dict = {
#                 "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
#                 "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
#                 "W2_states": self.W2,
#                 "b2_states": self.b2,
#             }
#             init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
#             init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))
#             init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict["W2_states"]))
#             init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict["b2_states"]))

#         # [B,num_heads, num_mini_batch, mini_batch_size, f] -> [num_mini_batch, B, num_heads, mini_batch_size, f]
#         inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

#         # allocate output tensor
#         XQW_batch = torch.empty(
#             (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
#             device=device,
#             dtype=dtype,
#         )
#         # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
#         batch_params_dict, XQW_batch = scan(
#             compute_mini_batch,
#             init_params_dict,
#             inputs,
#             XQW_batch,
#             self.config.scan_checkpoint_group_size if self.training else 0,
#         )

#         # [B, num_heads, L, C]
#         if cache_params is not None:
#             cache_params.update(batch_params_dict, self.layer_idx, L)

#         # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
#         XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
#         # [B, L, C]
#         XQW_batch = XQW_batch.reshape(B, L, self.width)
#         return XQW_batch, batch_params_dict






# class TTTMLP(TTTBase):
#     def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
#         super().__init__(config, layer_idx)
#         # TTT model initialization for TTT-MLP
#         self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
#         self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
#         self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
#         self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
#         self.W_cross = nn.Parameter(torch.normal(0, 0.02, size=(self.hidden_size, self.hidden_size)))
#         self.b_cross = nn.Parameter(torch.zeros(1, self.hidden_size))

#     def ttt(
#         self,
#         inputs,
#         mini_batch_size,
#         last_mini_batch_params_dict,
#         cache_params: Optional[TTTCache] = None,
#     ):
#         if mini_batch_size is None:
#             mini_batch_size = self.mini_batch_size

#         # in this case, we are decoding
#         if last_mini_batch_params_dict is None and cache_params is not None:
#             last_mini_batch_params_dict = cache_params.ttt_params_to_dict(self.layer_idx)

#         # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
#         B = inputs["XV"].shape[0]
#         num_mini_batch = inputs["XV"].shape[2]
#         L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
#         device = inputs["XV"].device
#         dtype = inputs["XV"].dtype
#         # NOTE:
#         # for prefilling, we will always use dual form for faster computation
#         # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
#         # since we need store the gradient for the next mini-batch computation
#         use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

#         def compute_mini_batch(params_dict, inputs):
#             time1 = time.time()
#             # [B, nh, f, 4f]
#             W1_init = params_dict["W1_states"]
#             # [B, nh, 1, 4f]
#             b1_init = params_dict["b1_states"]
#             # [B, nh, 4f, f]
#             W2_init = params_dict["W2_states"]
#             # [B, nh, 1, f]
#             b2_init = params_dict["b2_states"]
            
#             # [hidden_dim, hidden_dim]
#             W_cross_init = params_dict["W_cross_states"]
#             # [hidden_dim, hidden_dim]
#             b_cross_init = params_dict["b_cross_states"]

#             # [B,nh,K,f]
#             XQ_mini_batch = inputs["XQ"]
#             XV_mini_batch = inputs["XV"]
#             XK_mini_batch = inputs["XK"]
#             # [B,nh,K,1]
#             eta_mini_batch = inputs["eta"]
#             token_eta_mini_batch = inputs["token_eta"]
#             ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

#             X1 = XK_mini_batch
#             # [B,nh,K,f] @ [B,nh,f,4f] -> [B,nh,K,4f]
#             Z1 = X1 @ W1_init + b1_init
#             X2 = F.gelu(Z1, approximate="tanh")
#             # [B,nh,K,4f] @ [B,nh,4f,f] -> [B,nh,K,f]
#             Z2 = X2 @ W2_init + b2_init
#             reconstruction_target = XV_mini_batch - XK_mini_batch

#             ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
#             ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)

            
#             ####################
#             ### cross update ###
#             ####################
#             f_ln_weight = self.f_ttt_norm_weight.reshape(1, self.hidden_size).to(XK_mini_batch.device)
#             f_ln_bias = self.f_ttt_norm_bias.reshape(1, self.hidden_size).to(XK_mini_batch.device)
            
#             frames, _, sl, _ = XK_mini_batch.shape
#             # [B, K, hideen_size]
#             X_cross = XK_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
            
#             cur_grad_W_cross = torch.zeros_like(W_cross_init)
#             cur_grad_b_cross = torch.zeros_like(b_cross_init)
            
#             # print(f"frames: {frames}")
#             # print(f"X_cross_shape: {X_cross.shape}")
       
#             # for f_id in range(frames-1):
#             #     f1_copy = X_cross[f_id].detach().clone()
#             #     f2_copy = X_cross[f_id + 1].detach().clone()
#             #     k_com = min_kcom(f1_copy, f2_copy)
#             #     # print(f"k_com: {k_com}")
#             #     f1com_f2uni = torch.cat([f1_copy[:, 0 : k_com], f2_copy[:, k_com : ]], dim=-1)
#             #     f2com_f1uni = torch.cat([f2_copy[:, 0 : k_com], f1_copy[:, k_com : ]], dim=-1)
#             #     # cross frame reconstruction
#             #     rec_f2 = f1com_f2uni @ W_cross_init + b_cross_init
#             #     rec_f1 = f2com_f1uni @ W_cross_init + b_cross_init
#             #     # ln backfoward
#             #     grad_Z2 = ln_fused_l2_bwd(rec_f1, f1_copy, f_ln_weight, f_ln_bias)
#             #     grad_Z3 = ln_fused_l2_bwd(rec_f2, f2_copy, f_ln_weight, f_ln_bias)
#             #     cur_grad_W_cross += f2com_f1uni.t() @ grad_Z2
#             #     cur_grad_W_cross += f1com_f2uni.t() @ grad_Z3
#             #     cur_grad_b_cross += grad_Z2.sum(dim=0, keepdim=True)
#             #     cur_grad_b_cross += grad_Z3.sum(dim=0, keepdim=True)
            
#             # update W2 b2
#             current_step = getattr(self, '_step_counter', 0)
#             self._step_counter = current_step + 1
            
#             # 计算 warmup 学习率
#             if current_step < self.config.warmup_steps:
#                 progress = current_step / self.config.warmup_steps
#                 eta_W_cross = self.config.warmup_start_lr + progress * (self.config.warmup_end_lr - self.config.warmup_start_lr)
#             else:
#                 eta_W_cross = self.config.warmup_end_lr
                
#             grad_W_cross_last = cur_grad_W_cross / frames
#             grad_b_cross_last = cur_grad_b_cross / frames
#             W_cross_last = W_cross_init - eta_W_cross * grad_W_cross_last
#             b_cross_last = b_cross_init - eta_W_cross * grad_b_cross_last
#             # # [B, K, hideen_size]
#             # f_XQ_mini_batch = XQ_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
#             # # # [B, K, K]
#             # # Attn2 = torch.tril(torch.matmul(f_XQ_mini_batch, X2.transpose(-2, -1))) / self.hidden_size
#             # Z_cross_bar = torch.matmul(f_XQ_mini_batch, W_cross_last) + b_cross_last
#             # Z_cross_bar = ln_fwd(Z_cross_bar, f_ln_weight, f_ln_bias)
#             # # [B,nh,K,f]
#             # Z_cross_bar = Z_cross_bar.reshape(frames, sl, self.num_heads, self.head_dim).transpose(2, 1)
            
#             ####################
#             ### intra update ###
#             ####################
#             # [B, nh, K, f]
#             grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
#             # [B, nh, K, 4f]
#             grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

#             if use_dual_form:
#                 Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))  # [B,nh,K,K]
#                 # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,4f] -> [B,nh,K,4f]
#                 b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
#                 # [B,nh,K,f] @ [B,nh,f,4f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,4f] + [B,nh,K,4f]
#                 Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
#                 X2_bar = F.gelu(Z1_bar, approximate="tanh")

#                 # [B,nh,K,K]
#                 Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
#                 # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
#                 b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
#                 # [B,nh,K,f] @ [1,nh,4f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
#                 Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

#                 last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
#                 # [B,nh,f,4f] - [B,nh,f,K] @ [B,nh,K,4f]
#                 W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
#                 # [B,nh,1,4f]
#                 b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
#                 # [B,nh,4f,f] - [B,nh,4f,K] @ [B,nh,K,f]
#                 W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
#                 # [B,nh,1,f]
#                 b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)
#                 grad_W1_last = torch.zeros_like(W1_last)
#                 grad_b1_last = torch.zeros_like(b1_last)
#                 grad_W2_last = torch.zeros_like(W2_last)
#                 grad_b2_last = torch.zeros_like(b2_last)

#             else:
#                 ttt_lr_eta_mini_batch = torch.broadcast_to(
#                     ttt_lr_eta_mini_batch,
#                     (
#                         *ttt_lr_eta_mini_batch.shape[:2],
#                         mini_batch_size,
#                         mini_batch_size,
#                     ),
#                 )

#                 # [B, nh, K, 4f, f]
#                 grad_W2 = torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2)
#                 grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W2)
#                 grad_W2 = grad_W2 + params_dict["W2_grad"].unsqueeze(2)
#                 # [B, nh, K, f]
#                 grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z2)
#                 grad_b2 = grad_b2 + params_dict["b2_grad"]

#                 # [B, nh, K, f, 4f]
#                 grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
#                 grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
#                 grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
#                 # [B, nh, K, 4f]
#                 grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
#                 grad_b1 = grad_b1 + params_dict["b1_grad"]

#                 W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
#                 b1_bar = b1_init - grad_b1 * token_eta_mini_batch
#                 W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_eta_mini_batch.unsqueeze(-1)
#                 b2_bar = b2_init - grad_b2 * token_eta_mini_batch

#                 # [B, nh, K, 1, f] @ [B, nh, K, f, 4f] -> [B, nh, K, 4f]
#                 Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
#                 X2_bar = F.gelu(Z1_bar, approximate="tanh")
#                 Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar

#                 W1_last = W1_bar[:, :, -1]
#                 b1_last = b1_bar[:, :, -1:]
#                 W2_last = W2_bar[:, :, -1]
#                 b2_last = b2_bar[:, :, -1:]
#                 grad_W1_last = grad_W1[:, :, -1]
#                 grad_b1_last = grad_b1[:, :, -1:]
#                 grad_W2_last = grad_W2[:, :, -1]
#                 grad_b2_last = grad_b2[:, :, -1:]

#             Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

#             # XQW_mini_batch = XQ_mini_batch + Z2_bar + Z_cross_bar
#             XQW_mini_batch = XQ_mini_batch + Z2_bar

#             last_param_dict = {
#                 "W1_states": W1_last,
#                 "b1_states": b1_last,
#                 "W2_states": W2_last,
#                 "b2_states": b2_last,
#                 "W1_grad": grad_W1_last,
#                 "b1_grad": grad_b1_last,
#                 "W2_grad": grad_W2_last,
#                 "b2_grad": grad_b2_last,
#                 "W_cross_states": W_cross_last,
#                 "b_cross_states": b_cross_last,
#                 "W_cross_grad": grad_W_cross_last,
#                 "b_cross_grad": grad_b_cross_last
#             }
#             time2 = time.time()



from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

from transformers import PretrainedConfig

import time
import logging


class TTTConfig(PretrainedConfig):
   
    model_type = "ttt"

    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pretraining_tp=1,
        rope_theta=10000.0,
        use_gate=False,
        bidirectional=False,
        ttt_layer_type="linear",
        ttt_base_lr=1.0,
        mini_batch_size=16,
        warmup_steps=1000,
        warmup_start_lr=1e-6,
        warmup_end_lr=5e-4,
        scan_checkpoint_group_size=0,
        output_proj_dim = 768,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.rope_theta = rope_theta
        self.use_gate = use_gate
        self.ttt_layer_type = ttt_layer_type
        self.ttt_base_lr = ttt_base_lr
        self.mini_batch_size = mini_batch_size
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr

        self.scan_checkpoint_group_size = scan_checkpoint_group_size
        
        # output_dim
        self.output_proj_dim = output_proj_dim

        self.bidirectional = bidirectional

        super().__init__(
            **kwargs,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    return q, k


def undo_permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    return q, k


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=32,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cuda:0"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def scan(f, init, xs, out, checkpoint_group=0):
   
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])
        
    # # print(xs.items())
    # print(f"checkpoint_group: {checkpoint_group}")

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
                
            # for key, tensor in xs.items():
            #     print(key)
            #     print(tensor.shape)
            # print(f"carry: {carry['W1_states'].shape}")
            # print(f"x: {x['XQ'].shape}")
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out


def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]
    
    
    # print("====== ln_fused_l2_bwd =====")
    # print(f"x: {x.shape}")
    # print(f"l2_target: {l2_target.shape}")
    # print(f"gamma: {gamma.shape}")
    # print(f"beta: {beta.shape}")

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

def compute_feature_correlation(A, B, eps=1e-8):
    """
    compute the correlation matrix of two tensors in feature dimension (vectorized implementation)
    
    Args:
    - A: tensor with shape [l, f_a]
    - B: tensor with shape [l, f_b]
    
    Returns:
    - C: tensor with shape [f_a, f_b]
    """
    # normalize each feature of A
    A_centered = A - A.mean(dim=0, keepdim=True)
    A_normalized = A_centered / (torch.sqrt(torch.sum(A_centered**2, dim=0, keepdim=True)) + eps)
    
    # normalize each feature of B
    B_centered = B - B.mean(dim=0, keepdim=True)
    B_normalized = B_centered / (torch.sqrt(torch.sum(B_centered**2, dim=0, keepdim=True)) + eps)
    
    # compute the correlation matrix
    C = torch.matmul(A_normalized.t(), B_normalized)
    
    return C

def compute_corss_loss(C, k):

    # feature dimension
    f = C.shape[0]
    
    # extract the corresponding parts of the correlation matrix
    C_k = C[:k, :k] 
    C_rest = C[k:, k:]
    
    # compute the L_com loss
    com_diag_mask = torch.eye(k, device=C.device)
    com_diag_loss = torch.sum(((1 - C_k) * com_diag_mask) ** 2) / k
    com_non_diag_mask = 1 - com_diag_mask
    com_non_diag_loss = torch.sum((C_k * com_non_diag_mask)**2) / (k * (k - 1))
    L_com = com_diag_loss + com_non_diag_loss
    
    # compute the L_uni loss
    uni_diag_mask = torch.eye(f - k, device=C.device)
    uni_diag_loss = torch.sum((C_rest * uni_diag_mask) ** 2) / (f - k)
    uni_non_diag_mask = 1 - uni_diag_mask
    uni_non_diag_loss = torch.sum((C_rest * uni_non_diag_mask) ** 2) / ((f - k) * (f - k - 1))
    L_uni = uni_diag_loss + uni_non_diag_loss
    
    # compute the cross loss
    L_cross = L_com + L_uni
    
    return L_com, L_uni, L_cross

def min_kcom(f1, f2):
    _, f1_len = f1.shape
    _, f2_len = f2.shape
    corr_matrix = compute_feature_correlation(f1, f2)
    
    min_len = min(f1_len, f2_len)
    start_k = min(1, min_len // 20)
    end_k = max(start_k, min_len - 100)
    
    min_k = start_k
    min_cross_loss = 1e9
    
    for k in np.unique(np.round(np.linspace(start_k, end_k, 3)).astype(int)).tolist():
        cur_cross_loss, _, _ = compute_corss_loss(corr_matrix, k)
        if cur_cross_loss < min_cross_loss:
            min_cross_loss = cur_cross_loss
            min_k = k
    
    return min_k



class TTTBase(nn.Module):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logging.warning(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.mini_batch_size
        self.output_proj_dim = config.output_proj_dim

        # token_idx is a scale factor that scale the summation in Eqn. 4
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        # make the scale factor learnable
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self._init_qkvo_proj()
        self._init_rope()

        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        # use gating in the cross-linear
        self.use_gate = config.use_gate
        if self.use_gate:
            self.gate_alpha = nn.Parameter(torch.ones(self.width) * 0.1)

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

        # bidirctional process
        self.bidirectional = config.bidirectional

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.output_proj_dim, bias=False)

    def _init_rope(self):
        self.rope_theta = self.config.rope_theta
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=self.rope_theta,
        )

    def _init_ttt_lr_gate(self):
        # [width, 1]
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        # prepending head dim -> [num_heads, width, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        # [num_heads, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))
        
        cross_feature_dim = self.head_dim * self.num_heads
        self.f_ttt_norm_weight = nn.Parameter(nn.LayerNorm(cross_feature_dim).weight.data)
        self.f_ttt_norm_bias = nn.Parameter(nn.LayerNorm(cross_feature_dim).bias.data)

    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        return XQ, XK, XV

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        # [B, num_heads, num_mini_batch, 1, mini_batch_size]
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim

        # [B, L]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset : mini_batch_step_offset + mini_batch_size]

        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, ttt_lr_eta

    def apply_gate(self, hidden_states, ttt_output):
        """Apply gating as per equation 6: gate(TTT, X; α) = tanh(α) ⊗ TTT(X) + X"""
        gate_values = torch.tanh(self.gate_alpha)
        gated_output = gate_values.view(1, 1, -1) * ttt_output
        return gated_output + hidden_states

    def get_ttt_inputs(self, inputs, mini_batch_size):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        # [B ,num_mini_batch, mini_batch_size, C]
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)

        
        mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
        eta = token_eta * ttt_lr_eta
        # decouple token_coeff and ilr_coeff for decoding
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        return inputs

    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        # bidirectional processing
        if self.bidirectional:
            
            forward_output = self._forward_direction(hidden_states, attention_mask, position_ids)
            
            if position_ids is not None:
                reversed_states, reversed_positions = self.reverse_sequence(hidden_states, position_ids)
                reversed_output = self._forward_direction(reversed_states, attention_mask, reversed_positions)
            else:
                reversed_states = self.reverse_sequence(hidden_states)
                reversed_output = self._forward_direction(reversed_states, attention_mask, None)
            
            # reverse the sequence for chronological processing
            reversed_output = self.reverse_sequence(reversed_output)
            
            output_hidden_states = (forward_output + reversed_output) / 2
        else:
            # original directional processing
            output_hidden_states = self._forward_direction(hidden_states, attention_mask, position_ids)
        
        return output_hidden_states

    def _forward_direction(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """Standard forward pass in a single direction"""
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None
        
        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)

        # permute_qk and undo_permute_qk is just for aligning pytorch with jax pre-training
        XQ, XK = permute_qk(XQ, XK)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ, XK = undo_permute_qk(XQ, XK)

        output_hidden_states = []
        # when input sequence length is not a multiple of mini_batch_size
        # we need to compute them seperately, when computing the reminder,
        # we will need the last_mini_batch_params_dict to continue TTT learning
        if num_mini_batch > 0:
            inputs = {
                "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            output_mod, last_mini_batch_params_dict = self.ttt(
                self.get_ttt_inputs(inputs, self.mini_batch_size),
                mini_batch_size=self.mini_batch_size,
                last_mini_batch_params_dict=last_mini_batch_params_dict 
            )
            output_hidden_states.append(output_mod)
        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_reminder, _ = self.ttt(
                self.get_ttt_inputs(inputs, reminder_len),
                mini_batch_size=reminder_len,
                last_mini_batch_params_dict=last_mini_batch_params_dict
            
            )
            output_hidden_states.append(output_reminder)
        
        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        
        # gate control
        if self.use_gate:
            output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
        
        output_hidden_states = self.o_proj(output_hidden_states)
        
        return output_hidden_states

    # reverse the sequence for bidirectional processing
    def reverse_sequence(self, hidden_states, position_ids=None):
        """Reverse the sequence for bidirectional processing."""
        reversed_states = torch.flip(hidden_states, dims=[1])
        
        if position_ids is not None:
            reversed_positions = torch.flip(position_ids, dims=[1])
            return reversed_states, reversed_positions
        return reversed_states


class TTTLinear(TTTBase):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # TTT model initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.hidden_size, self.hidden_size)))
        self.b2 = nn.Parameter(torch.zeros(1, self.hidden_size))

    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size
            
        # print("=================  ttt debug =================")
        # print(f"XV: {inputs["XV"].shape}")
        # print(f"XQ: {inputs["XQ"].shape}")
        # print(f"XK: {inputs["XK"].shape}")



        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype
        
        # print(f"last_mini_batch_params_dict: {last_mini_batch_params_dict}")

        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, f]
            b1_init = params_dict["b1_states"]
            
            # W2_init = params_dict["W2_states"]
            # b2_init = params_dict["b2_states"]

            # [B,nh,K,f], K=mini_batch_size
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]
        
            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim).to(XK_mini_batch.device)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim).to(XK_mini_batch.device)
            # f_ln_weight = self.f_ttt_norm_weight.reshape(1, self.hidden_size).to(XK_mini_batch.device)
            # f_ln_bias = self.f_ttt_norm_bias.reshape(1, self.hidden_size).to(XK_mini_batch.device)
            
            # frames, _, sl, _ = XK_mini_batch.shape
            # # [B, K, hideen_size]
            # X2 = XK_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
            
            # cur_grad_W2 = torch.zeros_like(W2_init)
            # cur_grad_b2 = torch.zeros_like(b2_init)
            
            # print(f"frames: {frames}")
            # print(f"X2_shape: {X2.shape}")
       
            # for f_id in range(frames-1):
            #     f1_copy = X2[f_id].detach().clone()
            #     f2_copy = X2[f_id + 1].detach().clone()
            #     k_com = min_kcom(f1_copy, f2_copy)
            #     # print(f"k_com: {k_com}")
            #     f1com_f2uni = torch.cat([f1_copy[:, 0 : k_com], f2_copy[:, k_com : ]], dim=-1)
            #     f2com_f1uni = torch.cat([f2_copy[:, 0 : k_com], f1_copy[:, k_com : ]], dim=-1)
            #     # cross frame reconstruction
            #     rec_f2 = f1com_f2uni @ W2_init + b2_init
            #     rec_f1 = f2com_f1uni @ W2_init + b2_init
            #     # ln backfoward
            #     grad_Z2 = ln_fused_l2_bwd(rec_f1, f1_copy, f_ln_weight, f_ln_bias)
            #     grad_Z3 = ln_fused_l2_bwd(rec_f2, f2_copy, f_ln_weight, f_ln_bias)
            #     cur_grad_W2 += f2com_f1uni.t() @ grad_Z2
            #     cur_grad_W2 += f1com_f2uni.t() @ grad_Z3
            #     cur_grad_b2 += grad_Z2.sum(dim=0, keepdim=True)
            #     cur_grad_b2 += grad_Z3.sum(dim=0, keepdim=True)
            
            # update W2 b2
            # 计算当前步数
            # current_step = getattr(self, '_step_counter', 0)
            # self._step_counter = current_step + 1
            
            # # 计算 warmup 学习率
            # if current_step < self.config.warmup_steps:
            #     progress = current_step / self.config.warmup_steps
            #     eta_W2 = self.config.warmup_start_lr + progress * (self.config.warmup_end_lr - self.config.warmup_start_lr)
            # else:
            #     eta_W2 = self.config.warmup_end_lr
                
            # grad_W2_last = cur_grad_W2 / frames
            # grad_b2_last = cur_grad_b2 / frames
            # W2_last = W2_init - eta_W2 * grad_W2_last
            # b2_last = b2_init - eta_W2 * grad_b2_last
            # # [B, K, hideen_size]
            # f_XQ_mini_batch = XQ_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
            # # # [B, K, K]
            # # Attn2 = torch.tril(torch.matmul(f_XQ_mini_batch, X2.transpose(-2, -1))) / self.hidden_size
            # Z2_bar = torch.matmul(f_XQ_mini_batch, W2_last) + b2_last
            # Z2_bar = ln_fwd(Z2_bar, f_ln_weight, f_ln_bias)
            # # [B,nh,K,f]
            # Z2_bar = Z2_bar.reshape(frames, sl, self.num_heads, self.head_dim).transpose(2, 1)
           
            # print(f"use_dual_form: {use_dual_form}")
           
            # [B,nh,K,f]
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            if use_dual_form:
                # [B,nh,K,K]
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # print(f"eta_mini_batch.shape: {eta_mini_batch.shape}")
                # print(f"last_eta_mini_batch: {last_eta_mini_batch.shape}")
                # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            # XQW_mini_batch = XQ_mini_batch + Z1_bar + Z2_bar
            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last
                # "W2_states": W2_last,
                # "b2_states": b2_last,
                # "W2_grad": grad_W2_last,
                # "b2_grad": grad_b2_last,
            }
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "W2_states": self.W2,
                "b2_states": self.b2,
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))
            init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict["W2_states"]))
            init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict["b2_states"]))

        # [B,num_heads, num_mini_batch, mini_batch_size, f] -> [num_mini_batch, B, num_heads, mini_batch_size, f]
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


class TTTMLP(TTTBase):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # TTT model initialization for TTT-MLP
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        self.W_cross = nn.Parameter(torch.normal(0, 0.02, size=(self.hidden_size, self.hidden_size)))
        self.b_cross = nn.Parameter(torch.zeros(1, self.hidden_size))
        
        # gate parameters
        if self.use_gate:
            self.gate_alpha_z2 = nn.Parameter(torch.ones(self.head_dim) * 0.1)
            self.gate_alpha_cross = nn.Parameter(torch.ones(self.head_dim) * 0.1)

    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype
     
        use_dual_form = L % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, 4f]
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, 4f]
            b1_init = params_dict["b1_states"]
            # [B, nh, 4f, f]
            W2_init = params_dict["W2_states"]
            # [B, nh, 1, f]
            b2_init = params_dict["b2_states"]
            
            # [hidden_dim, hidden_dim]
            W_cross_init = params_dict["W_cross_states"]
            # [hidden_dim, hidden_dim]
            b_cross_init = params_dict["b_cross_states"]

            # [B,nh,K,f]
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B,nh,K,1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,4f] -> [B,nh,K,4f]
            Z1 = X1 @ W1_init + b1_init
            X2 = F.gelu(Z1, approximate="tanh")
            # [B,nh,K,4f] @ [B,nh,4f,f] -> [B,nh,K,f]
            Z2 = X2 @ W2_init + b2_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            
            ####################
            ### intra update ###
            ####################
            # [B, nh, K, f]
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
            # [B, nh, K, 4f]
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))  # [B,nh,K,K]
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,4f] -> [B,nh,K,4f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,4f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,4f] + [B,nh,K,4f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")

                # [B,nh,K,K]
                Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
                # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
                # [B,nh,K,f] @ [1,nh,4f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,4f] - [B,nh,f,K] @ [B,nh,K,4f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,4f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                # [B,nh,4f,f] - [B,nh,4f,K] @ [B,nh,K,f]
                W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
                # [B,nh,1,f]
                b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
                grad_W2_last = torch.zeros_like(W2_last)
                grad_b2_last = torch.zeros_like(b2_last)

            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, 4f, f]
                grad_W2 = torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2)
                grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W2)
                grad_W2 = grad_W2 + params_dict["W2_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z2)
                grad_b2 = grad_b2 + params_dict["b2_grad"]

                # [B, nh, K, f, 4f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                # [B, nh, K, 4f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_eta_mini_batch.unsqueeze(-1)
                b2_bar = b2_init - grad_b2 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, 4f] -> [B, nh, K, 4f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")
                Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
                grad_W2_last = grad_W2[:, :, -1]
                grad_b2_last = grad_b2[:, :, -1:]

            ####################
            ### cross update ###
            ####################
            frames, _, sl, _ = XK_mini_batch.shape
            # [B, K, hideen_size]
            X_cross = XK_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
            
            # print(f"frames: {frames}")
            # print(f"X_cross_shape: {X_cross.shape}")
            
            f_ln_weight = self.f_ttt_norm_weight.reshape(1, X_cross.shape[-1]).to(XK_mini_batch.device)
            f_ln_bias = self.f_ttt_norm_bias.reshape(1, X_cross.shape[-1]).to(XK_mini_batch.device)
            
            cur_grad_W_cross = torch.zeros_like(W_cross_init)
            cur_grad_b_cross = torch.zeros_like(b_cross_init)
            
            for f_id in range(frames-1):
                f1_copy = X_cross[f_id].detach().clone()
                f2_copy = X_cross[f_id + 1].detach().clone()
                k_com = min_kcom(f1_copy, f2_copy)
                # print(f"k_com: {k_com}")
                f1com_f2uni = torch.cat([f1_copy[:, 0 : k_com], f2_copy[:, k_com : ]], dim=-1)
                f2com_f1uni = torch.cat([f2_copy[:, 0 : k_com], f1_copy[:, k_com : ]], dim=-1)
                # cross frame reconstruction
                rec_f2 = f1com_f2uni @ W_cross_init + b_cross_init
                rec_f1 = f2com_f1uni @ W_cross_init + b_cross_init
                # ln backfoward
                grad_Z2 = ln_fused_l2_bwd(rec_f1, f1_copy, f_ln_weight, f_ln_bias)
                grad_Z3 = ln_fused_l2_bwd(rec_f2, f2_copy, f_ln_weight, f_ln_bias)
                cur_grad_W_cross += f2com_f1uni.t() @ grad_Z2
                cur_grad_W_cross += f1com_f2uni.t() @ grad_Z3
                cur_grad_b_cross += grad_Z2.sum(dim=0, keepdim=True)
                cur_grad_b_cross += grad_Z3.sum(dim=0, keepdim=True)
            
            # update W2 b2
            current_step = getattr(self, '_step_counter', 0)
            self._step_counter = current_step + 1
            
            #  warmup lr
            if current_step < self.config.warmup_steps:
                progress = current_step / self.config.warmup_steps
                eta_W_cross = self.config.warmup_start_lr + progress * (self.config.warmup_end_lr - self.config.warmup_start_lr)
            else:
                eta_W_cross = self.config.warmup_end_lr
                
            grad_W_cross_last = cur_grad_W_cross / frames
            grad_b_cross_last = cur_grad_b_cross / frames
            W_cross_last = W_cross_init - eta_W_cross * grad_W_cross_last
            b_cross_last = b_cross_init - eta_W_cross * grad_b_cross_last
            # [B, K, hideen_size]
            f_XQ_mini_batch = XQ_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
            # # [B, K, K]
            # Attn2 = torch.tril(torch.matmul(f_XQ_mini_batch, X2.transpose(-2, -1))) / self.hidden_size
            Z_cross_bar = torch.matmul(f_XQ_mini_batch, W_cross_last) + b_cross_last
            Z_cross_bar = ln_fwd(Z_cross_bar, f_ln_weight, f_ln_bias)
            # [B,nh,K,f]
            Z_cross_bar = Z_cross_bar.reshape(frames, sl, self.num_heads, self.head_dim).transpose(2, 1)
            
            
            # gate update
            if self.use_gate:
                gate_values_z2 = torch.tanh(self.gate_alpha_z2)
                Z2_bar = gate_values_z2.view(1, 1, 1, -1) * Z2_bar
                
                gate_values_cross = torch.tanh(self.gate_alpha_cross)
                Z_cross_bar = gate_values_cross.view(1, 1, 1, -1) * Z_cross_bar
            
            # similar residual connection
            XQW_mini_batch = XQ_mini_batch + Z2_bar + Z_cross_bar
            
            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
                "W2_grad": grad_W2_last,
                "b2_grad": grad_b2_last,
                "W_cross_states": W_cross_last,
                "b_cross_states": b_cross_last,
                "W_cross_grad": grad_W_cross_last,
                "b_cross_grad": grad_b_cross_last
            }
            
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
                "W_cross_states": self.W_cross,
                "b_cross_states": self.b_cross,
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))
            init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict["W2_states"]))
            init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict["b2_states"]))
            init_params_dict.update(W_cross_grad=torch.zeros_like(init_params_dict["W_cross_states"]))
            init_params_dict.update(b_cross_grad=torch.zeros_like(init_params_dict["b_cross_states"]))
        
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]
        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )


        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


if __name__ == "__main__":
    # 创建 TTT 配置
    config = TTTConfig(
        hidden_size=12,            # 隐藏层大小
        intermediate_size=8,       # 中间层大小
        num_hidden_layers=2,       # 隐藏层数量
        num_attention_heads=3,     # 注意力头数量
        max_position_embeddings=64,# 最大位置嵌入
        ttt_layer_type="mlp",      # 使用 TTT-MLP 类型
        ttt_base_lr=1.0,           # TTT 学习率
        mini_batch_size=27,        # 小批量大小
        use_gate=True,             # 启用门控机制
        bidirectional=True,        # 启用双向处理
        output_proj_dim = 768
    )

    # 创建层实例
    ttt_layer = TTTMLP(config, layer_idx=0)

    # 准备输入数据
    batch_size = 16
    seq_len = 729
    hidden_size = config.hidden_size

    # 创建隐藏状态输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 创建位置 ID
    position_ids = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1)

    # 前向传播
    output = ttt_layer(
        hidden_states=hidden_states,
        position_ids=position_ids
    )

    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")

#             print(f"compute minibath using {time2 - time1} s")
#             return last_param_dict, XQW_mini_batch

#         if last_mini_batch_params_dict is not None:
#             init_params_dict = last_mini_batch_params_dict
#         else:
#             init_params_dict = {
#                 "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
#                 "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
#                 "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
#                 "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
#                 "W_cross_states": self.W_cross,
#                 "b_cross_states": self.b_cross,
#             }
#             init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
#             init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))
#             init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict["W2_states"]))
#             init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict["b2_states"]))
#             init_params_dict.update(W_cross_grad=torch.zeros_like(init_params_dict["W_cross_states"]))
#             init_params_dict.update(b_cross_grad=torch.zeros_like(init_params_dict["b_cross_states"]))
#         inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]
#         # allocate output tensor
#         XQW_batch = torch.empty(
#             (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
#             device=device,
#             dtype=dtype,
#         )
#         # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
#         batch_params_dict, XQW_batch = scan(
#             compute_mini_batch,
#             init_params_dict,
#             inputs,
#             XQW_batch,
#             self.config.scan_checkpoint_group_size if self.training else 0,
#         )

#         # [B, num_heads, L, C]
#         if cache_params is not None:
#             cache_params.update(batch_params_dict, self.layer_idx, L)

#         # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
#         XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
#         # [B, L, C]
#         XQW_batch = XQW_batch.reshape(B, L, self.width)
#         return XQW_batch, batch_params_dict


# ################################
# ### E2E Architecture Modules ###
# ################################






# if __name__ == "__main__":
#     # 创建 TTT 配置
#     config = TTTConfig(
#         hidden_size=12,            # 隐藏层大小
#         intermediate_size=8,     # 中间层大小
#         num_hidden_layers=2,       # 隐藏层数量
#         num_attention_heads=3,     # 注意力头数量
#         max_position_embeddings=64,# 最大位置嵌入
#         ttt_layer_type="mlp",    # 使用 TTT-Linear 类型
#         ttt_base_lr=1.0,            # TTT 学习率
#         mini_batch_size=27,         # 小批量大小
#         use_gate=False,             # 是否使用门控
#         share_qk=True,             # 是否共享 Q/K 投影
#         output_proj_dim = 768
#     )

#     # 创建 TTT-Linear 层
#     ttt_layer = TTTMLP(config, layer_idx=0)

#     # 准备输入数据
#     batch_size = 3
#     seq_len = 729
#     hidden_size = config.hidden_size

#     # 创建隐藏状态输入
#     hidden_states = torch.randn(batch_size, seq_len, hidden_size)

#     # 创建位置 ID
#     position_ids = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1)

#     # 前向传播
#     output = ttt_layer(
#         hidden_states=hidden_states,
#         position_ids=position_ids
#     )

#     print(f"输入形状: {hidden_states.shape}")
#     print(f"输出形状: {output.shape}")

#     # # 使用 TTTCache 进行解码示例
#     # cache = TTTCache(model=torch.nn.Module(), batch_size=batch_size)
#     # cache.ttt_params_dict["W1_states"][0] = torch.tile(ttt_layer.W1.unsqueeze(0), dims=(batch_size, 1, 1, 1))
#     # cache.ttt_params_dict["b1_states"][0] = torch.tile(ttt_layer.b1.unsqueeze(0), dims=(batch_size, 1, 1, 1))
#     # cache.ttt_params_dict["W1_grad"][0] = torch.zeros_like(cache.ttt_params_dict["W1_states"][0])
#     # cache.ttt_params_dict["b1_grad"][0] = torch.zeros_like(cache.ttt_params_dict["b1_states"][0])

#     # # 生成一个新的 token
#     # new_token = torch.randn(batch_size, 1, hidden_size)
#     # new_position_id = torch.tensor([[seq_len]])

#     # # 使用缓存进行单步预测
#     # output_with_cache = ttt_layer(
#     #     hidden_states=new_token,
#     #     position_ids=new_position_id,
#     #     cache_params=cache
#     # )

#     # print(f"单步输入形状: {new_token.shape}")
#     # print(f"单步输出形状: {output_with_cache.shape}")