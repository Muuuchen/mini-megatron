"""固定形状、阻塞 P2P，FP32；传输与 autograd 图连接由 scheduler 显式管理。

不发送 shape header，不使用 tags（NCCL 不支持 tag），依靠相邻 ranks 一致的
microbatch 顺序。send 使用 WORLD 的 global rank，不能把 pp_rank 当 global rank。
"""
import torch
import torch.distributed as dist


def recv_forward(mesh, batch_size, seq_len, hidden_size):
    x = torch.empty(batch_size, seq_len, hidden_size, device=mesh.device, dtype=torch.float32)
    dist.recv(x, src=mesh.peer(mesh.pp_rank - 1))
    return x.requires_grad_(True)  # 本 stage 的叶子输入，backward 后读取 x.grad。


def send_forward(x, mesh):
    dist.send(x.detach().contiguous(), dst=mesh.peer(mesh.pp_rank + 1))


def recv_backward(output, mesh):
    grad = torch.empty_like(output)
    dist.recv(grad, src=mesh.peer(mesh.pp_rank + 1))
    return grad


def send_backward(grad, mesh):
    if grad is None:
        raise RuntimeError("missing received activation's gradient; check leaf/graph handling")
    dist.send(grad.detach().contiguous(), dst=mesh.peer(mesh.pp_rank - 1))
