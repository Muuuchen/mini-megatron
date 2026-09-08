"""词表并行 CE：通信 [B,S]，不是 gather 完整 [B,S,V] logits。"""
import torch
import torch.distributed as dist
from megatron_mini.exercise import todo
from megatron_mini.distributed.mesh import sum_
from megatron_mini.training.loss import causal_lm_loss


class _VocabCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, mesh):
        """T08: local logits=[B,S,V/TP]，返回逐 token loss=[B,S]。

        契约：稳定 logsumexp，不改原始 logits，暂不支持 ignore_index。
        1. 全局 max：局部 max 后 TP MAX；2. exp(z-max) 的分母做 TP SUM；
        3. 仅 target 所在 shard 贡献 target logit，TP SUM；
        4. loss=log(global_sum_exp)+global_max-target_logit。
        保存本地 softmax、安全的本地 target index 和 target ownership mask，
        用于 T09。输入 fp32/fp64 保持 dtype；主线不启用混合精度。
        """
        todo("T08")

    @staticmethod
    def backward(ctx, grad_output):
        """T09: 返回 (local_dlogits, None, None)。

        dlogits=(local_softmax-local_one_hot)*grad_output[...,None]。
        只在 target 的归属 shard 减 1；不额外乘/除 TP，不在这里做 mean。
        grad_output 来自外层 mean 及 microbatch loss 缩放。
        """
        todo("T09")


def vocab_parallel_cross_entropy(local_logits, targets, mesh):
    return _VocabCE.apply(local_logits, targets, mesh)


def language_model_loss(local_logits, targets, mesh):
    if mesh.dims.tp == 1:
        return causal_lm_loss(local_logits, targets)
    return vocab_parallel_cross_entropy(local_logits, targets, mesh).mean()
