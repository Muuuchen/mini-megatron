"""v2训练生命周期已固定；T55负责将具体模型/通信策略接入这些窄接口。"""
from dataclasses import dataclass
from typing import Callable
from megatron_mini.exercise import todo

@dataclass
class TrainRuntime:
    zero_grad: Callable
    forward_backward: Callable   # (microbatches)->detached logged loss; scaled backward inside
    finish_communication: Callable  # complete ALL grad reductions + waits; before finite check
    unscale_and_check: Callable  # ()->global found_inf bool
    finalize_and_clip: Callable  # ()->preclip norm and clip ONLY; all reductions already finished
    optimizer_step: Callable
    update_scale: Callable      # (found_inf)->None
    model: object=None          # expose owned shards for integration checks/checkpoint
    optimizer: object=None
    mesh: object=None


def train_step_v2(runtime,microbatches):
    if not microbatches: raise ValueError('empty microbatch window')
    runtime.zero_grad()
    loss=runtime.forward_backward(microbatches)
    runtime.finish_communication()
    found_inf=runtime.unscale_and_check()
    norm=None
    if not found_inf:
        norm=runtime.finalize_and_clip()
        runtime.optimizer_step()
    runtime.update_scale(found_inf)
    return {'loss':loss,'grad_norm':norm,'skipped':found_inf}


def build_training_runtime(config,model_config,device):
    """T55: 将已验收模块装配为TrainRuntime，config.validate()后按recipe建图。
    顺序: T21创建groups -> dense/llama/moe stage -> TP/SP/CP布局 ->
    block/attention重算 -> sharding(ddp/zero1/zero2/zero3/fsdp/torch_fsdp2) -> optimizer。
    v3: offload配置走T71共享装配；SAC用T68或T69/T70，不能只嵌套T33。
    再绑定GPipe/1F1B/VPP、mixed precision、GradRole及overlap hooks。
    finish_communication完成全部角色归约后再检测overflow；finalize_and_clip只做
    norm/clip，不再归约梯度。可复用T49，将第二次调用roles的reductions置为空。
    旧T14只能用于纯dense旧布局，不得对FSDP/EP重复套average。
    第一步先支持dense_tp_zero2；然后pipeline_1f1b；再moe_ep_fsdp。
    不需要实现全部配置笛卡尔积，不支持的组合显式NotImplementedError并记录。
    不能退回full dense单进程假装完成分布式。见docs/INTEGRATION.md逐行接线步骤。
    """
    todo('T55')
