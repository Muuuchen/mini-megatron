"""小型Llama风格dense oracle的结构已接好；不是HF/Qwen权重兼容实现。"""
import torch
from torch import nn
from torch.nn import functional as F
from megatron_mini.exercise import todo


def rms_norm(x,weight,eps=1e-6):
    """T51: 在hidden维计算mean(x^2)，FP32中间值（double测试不降精度），
    rsqrt 后还原dtype并乘weight。不是减mean的LayerNorm。"""
    todo('T51')


def apply_rope(x,positions,base=10000.0):
    """T52: x[B,heads,S,D]，positions[S]使用global CP offset。
    本课程固定相邻偶奇配对：rotate(x_even,x_odd)，频率base^(-2i/D)。
    旋转保持范数；position=0是identity；D必须偶数。禁止不同rank重置positions。
    """
    todo('T52')


def gqa_attention(q,k,v):
    """T53: q[B,Hq,S,D], kv[B,Hkv,S,D]，Hq%Hkv==0。
    通过repeat_interleave映射Q组对应KV，然后显式causal attention。
    这是dense reference；TP版本还要求本课程Hkv能被TP整除，不做KV复制特例。
    验收与逐Q-head选择所属KV的独立公式对齐，含dK/dV累加。
    """
    todo('T53')

class RMSNorm(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.weight=nn.Parameter(torch.ones(hidden))
    def forward(self,x): return rms_norm(x,self.weight)

class LlamaAttention(nn.Module):
    def __init__(self,hidden,heads,kv_heads):
        super().__init__()
        if hidden%heads or heads%kv_heads or (hidden//heads)%2: raise ValueError('invalid heads')
        self.heads,self.kv_heads,self.dim=heads,kv_heads,hidden//heads
        self.q=nn.Linear(hidden,hidden,bias=False)
        self.k=nn.Linear(hidden,kv_heads*self.dim,bias=False)
        self.v=nn.Linear(hidden,kv_heads*self.dim,bias=False)
        self.proj=nn.Linear(hidden,hidden,bias=False)
    def forward(self,x,positions):
        b,s,h=x.shape
        q=self.q(x).view(b,s,self.heads,self.dim).transpose(1,2)
        k=self.k(x).view(b,s,self.kv_heads,self.dim).transpose(1,2)
        v=self.v(x).view(b,s,self.kv_heads,self.dim).transpose(1,2)
        y=gqa_attention(apply_rope(q,positions),apply_rope(k,positions),v)
        return self.proj(y.transpose(1,2).reshape(b,s,h))

class LlamaBlock(nn.Module):
    def __init__(self,hidden=32,heads=4,kv_heads=2,ffn=64):
        super().__init__();self.norm1=RMSNorm(hidden);self.norm2=RMSNorm(hidden)
        self.attn=LlamaAttention(hidden,heads,kv_heads)
        self.gate=nn.Linear(hidden,ffn,bias=False);self.up=nn.Linear(hidden,ffn,bias=False)
        self.down=nn.Linear(ffn,hidden,bias=False)
    def forward(self,x,positions=None):
        if positions is None: positions=torch.arange(x.shape[1],device=x.device)
        x=x+self.attn(self.norm1(x),positions);y=self.norm2(x)
        return x+self.down(F.silu(self.gate(y))*self.up(y))
