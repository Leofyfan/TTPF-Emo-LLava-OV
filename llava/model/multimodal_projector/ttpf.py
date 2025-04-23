import torch
from .ttt import TTTConfig, TTTLinear, TTTMLP
import torch.nn as nn


class VisionTTTProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        print(f"input_dim, output_dim: {input_dim}  {output_dim}")
        self.hidden_size = input_dim
        self.output_proj_dim = output_dim
        self.layer_idx = 0
        self.ttt_config = self._init_ttt_config()
        self.ttt_layer = self._get_ttt_layer()
        
    def _init_ttt_config(self):
        ttt_config = TTTConfig(
            hidden_size=self.hidden_size,          
            intermediate_size=2048,     
            num_hidden_layers=2,       
            num_attention_heads=6,     
            max_position_embeddings=2048,
            ttt_layer_type="mlp",    
            ttt_base_lr=1e-1,         
            mini_batch_size=27,       
            use_gate=True,       
            bidirectional=True,  
            output_proj_dim = self.output_proj_dim   
        )
        return ttt_config
        
    def _get_ttt_layer(self):
        return TTTMLP(self.ttt_config, self.layer_idx)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1).to(x.device)
        output = self.ttt_layer(hidden_states=x, position_ids=position_ids)
        return output
    
    @property
    def config(self):
        return {"mm_projector_type": "ttpf"}
        
    