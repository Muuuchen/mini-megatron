from dataclasses import dataclass
from megatron_mini.exercise import todo

@dataclass(frozen=True)
class Event:
    op: str
    microbatch: int
    chunk: int=0

def plan_1f1b(stage,stages,microbatches):
    """T30: 非交错 per-stage event 列表，支持 M<P。
    warmup=min(P-stage-1,M)；先 warmup 次 F；随后 F/B；最后 cooldown B。
    FIFO backward ID，不要沿用 GPipe 的 reverse microbatch 顺序。
    只能返回 F/B event，通信由独立 executor 管理。
    验收: event 覆盖、依赖、live records；P=2/4, M=1/2/8。
    """
    todo('T30')

def plan_interleaved(stage,stages,microbatches,chunks):
    """T32: 最小 VPP=2 的 per-physical-stage event 列表。
    chunk c 在物理 p 上表示 logical stage c*P+p；跨 chunk 的边界是
    p=P-1 -> p=0，而不是两个本地 chunk 连着执行。backward 反向。
    可先限制 M 为 P 的正整数倍，其他情况显式 ValueError。
    每个(microbatch,chunk)一次 F/B；FIFO 与跨 rank matching 不可混淆。
    用 validate_schedule_simulation 做全局依赖/死锁验收再接 executor。
    """
    todo('T32')

def validate_schedule_simulation(plans,stages,microbatches,chunks=1):
    """已提供的离散依赖模拟，不等价于 NCCL 时序/无死锁证明。"""
    if len(plans)!=stages: raise AssertionError('missing physical stage')
    expected={(op,m,c*stages+p) for op in ('F','B') for m in range(microbatches)
              for c in range(chunks) for p in range(stages)}
    seen=set(); cursor=[0]*stages; live=[0]*stages; peak=[0]*stages
    while any(cursor[p]<len(plans[p]) for p in range(stages)):
        progressed=False
        for p in range(stages):
            if cursor[p]==len(plans[p]): continue
            e=plans[p][cursor[p]]; logical=e.chunk*stages+p; key=(e.op,e.microbatch,logical)
            if key not in expected or key in seen: raise AssertionError('invalid/duplicate event')
            deps=[]
            if e.op=='F' and logical>0: deps.append(('F',e.microbatch,logical-1))
            if e.op=='B':
                deps.append(('F',e.microbatch,logical))
                if logical<stages*chunks-1: deps.append(('B',e.microbatch,logical+1))
            if all(d in seen for d in deps):
                seen.add(key);cursor[p]+=1;progressed=True
                live[p]+=1 if e.op=='F' else -1;peak[p]=max(peak[p],live[p])
        if not progressed: raise AssertionError('schedule dependency deadlock')
    if seen!=expected or any(live): raise AssertionError('incomplete schedule')
    return peak
