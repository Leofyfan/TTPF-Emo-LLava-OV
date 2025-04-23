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
        use_cache=False,
        pretraining_tp=1,
        rope_theta=10000.0,
        use_gate=False,
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
        self.use_cache = use_cache
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
        max_position_embeddings=16,
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

#########################
### TTT Layer Modules ###
#########################


def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])
        
    # print("============ scan debug =============")
    # # print(xs.items())
    # print(f"num_items: {num_items}")
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
    
    for k in np.unique(np.round(np.linspace(start_k, end_k, 5)).astype(int)).tolist():
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
            self.g_proj = nn.Linear(self.width, self.width, bias=False)

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

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
        self.f_ttt_norm_weight = nn.Parameter(nn.LayerNorm(self.hidden_size).weight.data)
        self.f_ttt_norm_bias = nn.Parameter(nn.LayerNorm(self.hidden_size).bias.data)

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
        y = self.g_proj(hidden_states)
        # use 'tanh' approximation for matching JAX impl.
        y = F.gelu(y, approximate="tanh")
        output = y * ttt_output
        return output

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
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None
        
        
        # print(f"hidden_states: {hidden_states.shape}")
        # print(f"num_heads: {self.num_heads}  head_dim: {self.head_dim}")

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
        
          
        # print(f"len(output_hidden_states): {len(output_hidden_states)}")
        # for i in range(len(output_hidden_states)):
        #     print(f"output_hidden_state_{i} : {output_hidden_states[i].shape}")

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        if self.use_gate:
            output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        # print(f"output_hidden_states: {output_hidden_states.shape}")
        return output_hidden_states


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
            time1 = time.time()
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
            ### cross update ###
            ####################
            f_ln_weight = self.f_ttt_norm_weight.reshape(1, self.hidden_size).to(XK_mini_batch.device)
            f_ln_bias = self.f_ttt_norm_bias.reshape(1, self.hidden_size).to(XK_mini_batch.device)
            
            frames, _, sl, _ = XK_mini_batch.shape
            # [B, K, hideen_size]
            X_cross = XK_mini_batch.transpose(1, 2).reshape(frames, sl, -1)
            
            cur_grad_W_cross = torch.zeros_like(W_cross_init)
            cur_grad_b_cross = torch.zeros_like(b_cross_init)
            
            print(f"frames: {frames}")
            print(f"X_cross_shape: {X_cross.shape}")
       
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
            
            # 计算 warmup 学习率
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

            Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z2_bar + Z_cross_bar
            # XQW_mini_batch = XQ_mini_batch + Z2_bar

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
            time2 = time.time()
            print(f"compute minibath using {time2 - time1} s")
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
        intermediate_size=8,     # 中间层大小
        num_hidden_layers=2,       # 隐藏层数量
        num_attention_heads=3,     # 注意力头数量
        max_position_embeddings=64,# 最大位置嵌入
        ttt_layer_type="mlp",    # 使用 TTT-Linear 类型
        ttt_base_lr=1.0,            # TTT 学习率
        mini_batch_size=27,         # 小批量大小
        use_gate=False,             # 是否使用门控
        output_proj_dim = 768
    )

    # 创建 TTT-Linear 层
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
