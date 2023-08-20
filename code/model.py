import torch
from torch import nn
from torch.nn import functional as F
import mobilevit


def compute_freqs_cis(embed_dim: int, max_seq_length: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, embed_dim, 2)[: (embed_dim // 2)].float() / embed_dim))
    t = torch.arange(max_seq_length, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_real = torch.cos(freqs)
    freqs_imag = torch.sin(freqs)
    return freqs_real, freqs_imag

def apply_rotary_emb(query, key, C, T):
    xq_ = query.float().reshape(*query.shape[:-1], -1, 2)
    xk_ = key.float().reshape(*key.shape[:-1], -1, 2)

    xq_real = xq_[:, :, :, 0]
    xq_imag = xq_[:, :, :, 1]

    xk_real = xk_[:, :, :, 0]
    xk_imag = xk_[:, :, :, 1]

    freqs_real, freqs_imag = compute_freqs_cis(C, T)
    freqs_real = freqs_real.to(xq_.device)
    freqs_imag = freqs_imag.to(xq_.device)

    freqs_real = freqs_real.unsqueeze(0)
    freqs_imag = freqs_imag.unsqueeze(0)

    xq_out_real = xq_real * freqs_real - xq_imag * freqs_imag
    xq_out_imag = xq_real * freqs_imag + xq_imag * freqs_real
    xq_out = torch.stack([xq_out_real, xq_out_imag], dim = -1).flatten(3)

    xk_out_real = xk_real * freqs_real - xk_imag * freqs_imag
    xk_out_imag = xk_real * freqs_imag + xk_imag * freqs_real
    xk_out = torch.stack([xk_out_real, xk_out_imag], dim = -1).flatten(3)

    return xq_out.type_as(query), xk_out.type_as(key)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value  = self.c_attn(x).split(self.embed_dim, dim=2)
        query, key = apply_rotary_emb(query, key, C, T)


        key = key.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        query = query.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        value = value.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.query_attn = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_attn = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_attn = nn.Linear(embed_dim, embed_dim, bias=bias)

        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout


    def forward(self, query, key, value):
        query_batch, query_block, query_channels = query.size() # batch size, sequence length, embedding dimensionality (n_embd)
        key_batch, key_block, key_channels = key.size() # batch size, sequence length, embedding dimensionality (n_embd)
        value_batch, value_block, value_channels = value.size() # batch size, sequence length, embedding dimensionality (n_embd)

        assert query_batch == key_batch == value_batch
        assert query_channels == key_channels == value_channels

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query = self.query_attn(query)
        key = self.key_attn(key)
        value = self.value_attn(value)
        

        key = key.view(key_batch, key_block, self.num_heads, key_channels // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        query = query.view(query_batch, query_block, self.num_heads, query_channels // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        value = value.view(value_batch, value_block, self.num_heads, value_channels // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels

        y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)


        y = y.transpose(1, 2).contiguous().view(query_batch, query_block, query_channels) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape, num_heads, dropout = 0):
        super(Decoder, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(input_shape, input_shape*2, bias = False),
            nn.SiLU(),
            nn.Linear(input_shape*2, output_shape, bias = False),
            nn.SiLU(),
        )
        self.layernorm = RMSNorm(input_shape)
        self.layernorm2 = RMSNorm(input_shape)
        self.layernorm3 = RMSNorm(input_shape)
        self.MHA = SelfAttention(embed_dim=input_shape, num_heads=num_heads, bias = False, dropout=dropout) 
        self.MHA2 = Attention(embed_dim=input_shape, num_heads=num_heads, bias = False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, features):
        x = self.dropout(x)
        x1 = self.layernorm(x)
        x = x + self.MHA(x1)
        x2 = self.layernorm2(x)
        # print(x2.shape, features.shape)
        x = x + self.MHA2(x2, features, features)[0]
        x3 = self.layernorm3(x)
        x = x + self.feed_forward(x3)
        return x

class Foundation(nn.Module):
    def __init__(self, num_blocks,
                       num_heads, 
                       unk_char,
                       vocab_size = 19200, 
                       embd_size = 768, 
                       dropout = 0.1,
                       ):
        super(Foundation, self).__init__()
        self.text_embedding = nn.Embedding(vocab_size, embd_size)
        self.block = nn.ModuleList([Decoder(embd_size, embd_size, num_heads, dropout) for _ in range(num_blocks)])
        self.feature_encoder = mobilevit.mobilevit_xs()
        self.feature_encoder.eval()
        self.feature_vector = nn.Linear(8*8, embd_size)
        self.feature_vector2 = nn.Linear(384, embd_size)
        self.dense = nn.Linear(embd_size, vocab_size)
        self.embed_size = embd_size
        self.unk_char = unk_char

    # @torch.compile(dynamic = True, mode = 'reduce-overhead')
    def forward(self, x, y = None, return_loss = False):
        tokens, image = x        
        image = self.feature_encoder(image)
        image = image.view(-1, 384, 64)
        image = self.feature_vector(image).view(-1, self.embed_size, 384)
        image = self.feature_vector2(image).view(-1, self.embed_size, self.embed_size)
        
        x = self.text_embedding(tokens)
        for decoder in self.block:
            x = decoder(x, image)
        x = self.dense(x)
        if return_loss:
            mask = (y != self.unk_char).to(torch.int64).view(-1)
            x = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-1, reduction='none')
            x = (x * mask).sum() / mask.sum()
        else: 
            x = F.softmax(x, dim = -1)
        return x
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    

    