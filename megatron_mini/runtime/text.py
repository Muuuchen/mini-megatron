"""本地UTF-8字节样本，无下载或第三方tokenizer依赖。"""
from dataclasses import dataclass
import torch
from megatron_mini.exercise import todo

@dataclass
class LossStat:
    backward_loss: torch.Tensor
    reported_loss: torch.Tensor
    valid_tokens: int


def byte_examples(text,seq_len,pad=256):
    if seq_len<1: raise ValueError('seq_len must be positive')
    data=list(text.encode('utf-8'));out=[]
    for start in range(0,max(0,len(data)-1),seq_len):
        piece=data[start:start+seq_len+1];valid=len(piece)-1
        inp=piece[:-1]+[pad]*(seq_len-valid);tgt=piece[1:]+[pad]*(seq_len-valid)
        out.append((torch.tensor(inp),torch.tensor(tgt),torch.arange(seq_len)<valid))
    return out


def masked_token_loss(logits,targets,mask,data_axis):
    """T54: 返回LossStat。输入各rank的有效token数可不同，不能平均local means。
    backward_loss=local有效CE sum / global_valid_count；reported_loss=
    global有效CE sum / global_valid_count。只有日志项detach，计数/日志做allreduce。
    所有rank同时零token时抛错；单个rank零token时返回有图的0并参与通信。
    这个API固定采用SUM/global_count约定，后续grad只SUM不平均。
    普通DP/DDP平均梯度约定需要乘data_axis.size，不能不改归约就直接替换。
    """
    todo('T54')
