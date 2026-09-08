from dataclasses import dataclass
from pathlib import Path
from megatron_mini.exercise import todo

@dataclass(frozen=True)
class ShardMetadata:
    global_name: str
    global_shape: tuple
    offsets: tuple
    sizes: tuple
    replica_id: int=0


def save_sharded_checkpoint(directory,rank,payload,axis):
    """T50: checkpoint v2 payload包含model shards,optimizer moments,step,
    mesh/layout/parameter roles,precision/scaler,RNG,data cursor。
    每rank atomic replace写shard；确认所有shard写完后，rank0最后提交manifest。
    没有manifest/缺任一shard必须拒绝恢复，不能拿半套文件继续训练。
    提交后无待通信buffer。这里只保证同topology完整快照，不承诺容错服务。
    返回最终manifest路径；load/validate可写成同文件私有helper。
    """
    todo('T50')


def reshard_flat_state(shards,true_numel,new_world_size):
    """T60: 输入按old shard rank排序的一维tensor列表；剔除old padding，
    按新world重新padding/切分；返回新的tensor列表，不修改输入。
    分别用于FP32 master param、Adam m/v；step标量保持相同。
    本任务仅flat shard的2->1/4离线重分片，不等价于任意TP/EP转换或在线弹性。
    对总长度不足、负numel/非正world、dtype/device不一致明确报错。
    """
    todo('T60')
