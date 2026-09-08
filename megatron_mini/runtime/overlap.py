"""异步发起不是 overlap 的证据。pending保存 buffer，消费前建立依赖。"""
from dataclasses import dataclass,field
from megatron_mini.exercise import todo

@dataclass
class Pending:
    work: object
    buffers: tuple=()
    finished: bool=False
    def wait(self):
        if not self.finished:
            if self.work is not None: self.work.wait()
            self.finished=True
        return self.buffers

@dataclass
class Bucket:
    names: tuple
    buffer: object
    ready: set=field(default_factory=set)
    pending: object=None

def mark_bucket_ready(bucket,name,is_last_microbatch,launch):
    """T34: autograd post-accumulate hook 最后一个 microbatch 标记 grad ready。
    所有参数都ready时恰好一次launch(buffer)；launch返回Pending。旧梯度不能丢。
    每个rank统一bucket定义与launch顺序，未用专家参数要显式zero，不可skip collective。
    optimizer.step 前wait所有bucket；不可每microbatch把同一累积grad重复归约。
    lab bucket模拟ready时序；CUDA trace才验真正compute/comm overlap。
    """
    todo('T34')

def prefetch_next(current,next_unit,launch_gather,compute):
    """T41: 先 launch next_unit 的参数 gather，执行 current compute，然后
    在下一个 unit 消费参数前 wait；最多当前+下一unit full buffers。
    API 返回(result, Pending或None)。next=None不能发collective。
    本函数教学单元返回的pending由executor交给下一个unit，而非立刻wait。
    """
    todo('T41')

def tp_chunked_overlap(chunks,launch_gather,compute):
    """T56: AG(chunk i+1) 与 GEMM(chunk i) 双buffer流水线，返回按序outputs。
    launch_gather(i)->Pending，其buffers[0]为通信输出；compute(i,tensor)。
    首次必须等第0块；消费前wait；不可在不同rank上动态改变chunk数量/顺序。
    此为独立原语实验；用于SP时还须补相应RS backward，不能用无autograd拷贝。
    """
    todo('T56')

def ep_chunked_overlap(chunks,dispatch,expert_compute,combine):
    """T57: dispatch(i+1)与expert_compute(i)重叠，再combine(i)。
    dispatch返回Pending buffers[0]；compute(i,tensor)；combine(i,y)->output。
    专家不同 token counts/空chunk也必须参与一致通信；返回按source chunk顺序结果。
    可先双microbatch；不要依赖未知的下一个router结果提前发错all-to-all。
    """
    todo('T57')

def cp_ring_overlap(rounds,exchange,consume):
    """T58: exchange(i)返回下一KV buffer的Pending；consume(i)处理当前KV。
    每轮先发下一块交换再算当前块，之后wait，调用者在exchange闭包里swap。
    rounds对应逻辑KV块数；最后一轮无需再传已消费的KV。输出由consume闭包累积。
    这里只验发送/消费顺序；T29负责online softmax与backward。
    """
    todo('T58')
