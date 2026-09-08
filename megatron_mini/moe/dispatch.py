from dataclasses import dataclass
import torch
import torch.distributed as dist
from megatron_mini.exercise import todo

@dataclass
class RoutePlan:
    token_ids: torch.Tensor
    slots: torch.Tensor
    expert_ids: torch.Tensor
    send_splits: tuple
    tokens: int

def pack(x,expert_ids,num_experts,axis):
    """T43: 重复topk tokens，按owner=floor(expert_id/(E/EP))稳定排序。
    返回(packed_x,RoutePlan)。相同owner内部用expert_id排序方便本地执行。
    token_ids/slots记录原来token与topk槽位，反向combine不可靠猜。
    所有send_splits长度=EP、和=N*k，允许0个token去某个rank。
    """
    todo('T43')

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,send_splits,recv_splits,axis):
        ctx.send,ctx.recv,ctx.axis=send_splits,recv_splits,axis
        if axis.size==1: return x.clone()
        out=x.new_empty((sum(recv_splits),*x.shape[1:]))
        dist.all_to_all_single(out,x.contiguous(),output_split_sizes=list(recv_splits),
                              input_split_sizes=list(send_splits),group=axis.group)
        return out
    @staticmethod
    def backward(ctx,grad):
        """T44: inverse AllToAll；交换 send/recv split sizes，其他参数无梯度。
        返回(grad_input,None,None,None)；不能allreduce，也不除EP。
        """
        todo('T44')

def all_to_all(x,send,recv,axis): return _AllToAll.apply(x,tuple(send),tuple(recv),axis)

def receive_splits(send,axis):
    if axis.size==1: return tuple(send)
    device='cuda' if dist.get_backend(axis.group)=='nccl' else 'cpu'
    src=torch.tensor(send,device=device,dtype=torch.int64);dst=torch.empty_like(src)
    dist.all_to_all_single(dst,src,group=axis.group)
    return tuple(dst.tolist())

def combine(returned,plan,weights):
    """T45: returned顺序与packed_x相同，用weights[token_ids,slots]加权，
    然后 index_add/scatter_add 回[N,H]。topk>1不得覆盖写；不detach weights。
    参数梯度与router梯度都要对齐，而不是只看token顺序。
    """
    todo('T45')
