from dataclasses import dataclass,field
from typing import Callable
from megatron_mini.exercise import todo

@dataclass
class PipelineIO:
    forward: Callable  # forward(microbatch,chunk,input_leaf)->(output,optional_loss)
    receive: Callable  # receive(direction,microbatch,chunk)->tensor or None
    send: Callable     # send(direction,microbatch,chunk,tensor)->pending handle
    finish: Callable   # finish(): wait all pending; retained payload buffers still alive
    is_first: Callable # is_first(chunk)->bool
    is_last: Callable  # is_last(chunk)->bool


def execute_events(events,io:PipelineIO,microbatches:int):
    """T31: 通用 GPipe/1F1B/VPP executor，records[(mb,chunk)]保存图。
    F: receive input leaf -> forward -> 非末 async send detached output。
    B: 末 stage(loss/M).backward；非末 recv dout -> autograd.backward；
       非首 send input.grad；删除该 record。只有最后 logical stage 缩放 loss。
    在相邻 rank 能都先发的场合不能使用两个阻塞 send；已提交的 send buffer
    在 handle 完成前不能释放/重用。最后 io.finish()。返回 detached mean loss。
    验收: pipeline lab + recipe pipeline_1f1b，M<P/非零 bias/多次 update。
    """
    todo('T31')
