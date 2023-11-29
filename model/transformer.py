import torch 
from torch import nn
import math
from transformers.activations import ACT2FN
import torch.autograd as autograd
import numpy as np


"""
MultiHeadAttention with Attention Masking
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, downsample_rate = None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        downsample_rate = downsample_rate or 1
        
        self.internal_dim = hidden_dim // downsample_rate
        if self.internal_dim % self.num_heads != 0:
            print(self.internal_dim, self.num_heads)
            raise ValueError("num_attention_heads must divide hidden_size.")
        
        self.q_proj = nn.Linear(self.hidden_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_dim)

    def _separate_heads(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        c_per_head = hidden_dim // self.num_heads
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, c_per_head)
        return hidden_states.transpose(1, 2)
    
    def _recombine_heads(self, hidden_states):
        batch_size, head, seq_len, c_per_head = hidden_states.shape
        hidden_dim = self.num_heads * c_per_head
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        return hidden_states
    
    def forward(self, query, key, value, attention_mask = None):
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self._separate_heads(query)
        key = self._separate_heads(key)
        value = self._separate_heads(value)

        _, _, _, c_per_head = query.shape
        attn = query @ key.transpose(2, 3)
        attn = attn / math.sqrt(c_per_head)
        #print(attn.shape)      # [N H L L]

        if attention_mask is not None:
            # add 1 for cls token
            cls_attention = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([cls_attention, attention_mask], dim=1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            #print(attention_mask.shape)        # [N 1 1 L]
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim = -1)

        out = attn @ value
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn
        
class FFNBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, hidden_dim)
        self.act = ACT2FN['gelu']

    def forward(self, hidden_states):
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states



class Encoder(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, num_heads, drop_out=0.0, downsample_rate = None):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, downsample_rate = downsample_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop_out1 = nn.Dropout(drop_out)
        self.mlp = FFNBlock(hidden_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop_out2 = nn.Dropout(drop_out)

    def forward(self, hidden_states, attention_mask = None, output_attentions = False):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            query = hidden_states,
            key = hidden_states,
            value = hidden_states,
            attention_mask = attention_mask,
        )
        hidden_states = self.drop_out1(hidden_states)
        hidden_states = residual + hidden_states
        layernorm_output = self.norm2(hidden_states)
        hidden_states = hidden_states + self.drop_out2(self.mlp(layernorm_output))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs[0] if len(outputs) == 1 else outputs



class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, num_heads, seq_len, num_classes, drop_out=0.0, downsample_rate = None, mlp_dim = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim or 4 * hidden_dim
        self.layers = layers
        self.num_classes = num_classes

        #self.proj = nn.Linear(input_dim, hidden_dim, bias=False)

        self.layers = nn.ModuleList([
            Encoder(hidden_dim, mlp_dim, num_heads, drop_out, downsample_rate) for _ in range(layers)
        ])
        scale = hidden_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(hidden_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn(seq_len + 1, hidden_dim))
        self.ln_pre = nn.LayerNorm(hidden_dim)
        self.ln_post = nn.LayerNorm(hidden_dim)

        # linear classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # nonlinear classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     ACT2FN['gelu'],
        #     nn.Linear(hidden_dim // 2, num_classes),
        # )
        
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.class_embedding, std=self.hidden_dim ** -0.5)
        #nn.init.normal_(self.proj.weight, std=self.hidden_dim ** -0.5)

        for layer in self.layers:
            mha = layer.self_attn
            std = mha.internal_dim ** -0.5
            nn.init.normal_(mha.q_proj.weight, std=std)
            nn.init.normal_(mha.k_proj.weight, std=std)
            nn.init.normal_(mha.v_proj.weight, std=std)
            nn.init.normal_(mha.out_proj.weight, std=std)
            
            nn.init.zeros_(mha.q_proj.bias)
            nn.init.zeros_(mha.k_proj.bias)
            nn.init.zeros_(mha.v_proj.bias)
            nn.init.zeros_(mha.out_proj.bias)
            
            fc_std = mha.internal_dim ** -0.5
            nn.init.normal_(layer.mlp.lin1.weight, std=fc_std)
            nn.init.normal_(layer.mlp.lin2.weight, std=fc_std)
            nn.init.zeros_(layer.mlp.lin1.bias)
            nn.init.zeros_(layer.mlp.lin2.bias)

        classifier_std = self.hidden_dim ** -0.5
        nn.init.normal_(self.classifier.weight, std=classifier_std)
        nn.init.zeros_(self.classifier.bias)
        #nn.init.normal_(self.classifier[2].weight, std=classifier_std)
        #nn.init.zeros_(self.classifier[2].bias)
        
    def forward_features(self, x, attention_mask = None, output_attentions = False):
        # project to hidden dimension
        #x = self.proj(x)
        # x = [N L D]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [N (L + 1) D]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for layer in self.layers:
            x = layer(x, attention_mask = attention_mask, output_attentions = output_attentions)
        # cls token
        x = self.ln_post(x[:, 0, :])
        return x
    
    def forward(self, x, attention_mask = None, output_attentions = False):
        x = self.forward_features(x, attention_mask = attention_mask, output_attentions = output_attentions)
        x = self.classifier(x)
        return x
    
    # this never WORKS
    def rsc(self, x, attention_mask = None, output_attentions = False, labels = None):
        device = x.device
        f = self.forward_features(x, attention_mask = attention_mask, output_attentions = output_attentions)
        p = self.classifier(f)
        o = torch.nn.functional.one_hot(labels, num_classes = 9)
        g = autograd.grad((p * o).sum(), f)[0]
        percentiles = np.percentile(g.cpu(), 0.75, axis = 1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, g.size(1))
        mask_f = g.lt(percentiles.to(device)).float()

        f_muted = f * mask_f
        p_muted = self.classifier(f_muted)
        
        s = torch.nn.functional.softmax(p, dim = 1)
        s_muted = torch.nn.functional.softmax(p_muted, dim = 1)
        changes = (s * o).sum(1) - (s_muted * o).sum(1)
        percentiles = np.percentile(changes.detach().cpu(), 0.75)
        mask_b = changes.lt(percentiles).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        p_muted_again = self.classifier(f * mask)
        return p_muted_again