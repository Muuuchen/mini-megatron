import torch
import torch.distributed as dist
from megatron_mini.runtime.overlap import Pending
from megatron_mini.exercise import todo

class TypedP2P:
    """配置约定固定shape/dtype，避免沿用v1硬编码FP32；同peer消息按FIFO匹配。"""
    def __init__(self,shape,dtype,device,group=None):
        self.shape,self.dtype,self.device,self.group=tuple(shape),dtype,device,group
        self.pending=[]
    def send(self,x,peer):
        if tuple(x.shape)!=self.shape or x.dtype!=self.dtype:
            raise ValueError('P2P payload disagrees with activation spec')
        payload=x.detach().contiguous()
        h=Pending(dist.isend(payload,dst=peer,group=self.group),(payload,))
        self.pending.append(h);return h
    def recv(self,peer,requires_grad=False):
        x=torch.empty(self.shape,dtype=self.dtype,device=self.device)
        dist.recv(x,src=peer,group=self.group)
        return x.requires_grad_(requires_grad)
    def finish(self):
        for h in self.pending: h.wait()
        self.pending.clear()


def build_pipeline_io(stage_chunks,mesh,activation_spec,microbatches):
    """T59: 用TypedP2P和PipelineIO适配实际stage chunks。
    首logical stage提供tokens；末stage提供loss；其余receive叶子activation。
    forward peer按logical stage跨chunk wrap-around，backward反向；不把group rank
    当global rank传给isend。BF16与FP32 payload遵从activation_spec。
    为每step建立独立buffer队列；发送完成前不释放；最后统一finish。
    验收pp2 bf16、pp2 vpp2与FP32 dense step对照；CPU仅验证FP32。
    """
    todo('T59')
