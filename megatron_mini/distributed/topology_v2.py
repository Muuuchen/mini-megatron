"""v2 的 EP 是 dense-DP 维度内的子组，不再额外乘 WORLD_SIZE。"""
from dataclasses import dataclass
from megatron_mini.distributed.topology import GroupSpec
from megatron_mini.exercise import todo

@dataclass(frozen=True)
class MeshDims:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    ep: int = 1

    def __post_init__(self):
        if min(self.tp,self.pp,self.dp,self.cp,self.ep)<1: raise ValueError('positive dims required')
        if self.dp % self.ep: raise ValueError('EP must divide dense DP in this teaching layout')

    @property
    def world_size(self): return self.tp*self.pp*self.cp*self.dp
    @property
    def edp(self): return self.dp//self.ep

    def rank(self,d,p,c,t):
        if not (0<=d<self.dp and 0<=p<self.pp and 0<=c<self.cp and 0<=t<self.tp):
            raise ValueError('coordinates out of bounds')
        return ((d*self.pp+p)*self.cp+c)*self.tp+t

    def coords(self,r):
        if not 0<=r<self.world_size: raise ValueError('rank out of bounds')
        return (r//(self.pp*self.cp*self.tp), (r//(self.cp*self.tp))%self.pp,
                (r//self.tp)%self.cp, r%self.tp)

    def expert_coords(self,r):
        d,_,_,_=self.coords(r)
        return d//self.ep,d%self.ep  # expert replica index, expert owner index


def plan_groups_v2(dims):
    """T21: 返回全世界相同顺序的 GroupSpec，包含 singleton。
    顺序 tp,pp,cp,dp,ep,edp,dp_cp,edp_cp。
    tp/pp/cp/dp: 只变化对应坐标。ep: 固定 d//ep，变化 d%ep。
    edp: 固定 d%ep，变化 d//ep。dp_cp: 固定(p,t)，变化(d,c)。
    edp_cp: 固定(p,t,d%ep)，变化(d//ep,c)。
    每类按 ranks tuple 排序。FSDP 在 dp/edp 上分片，CP 先归约贡献。
    验收: test_v2_exercises::test_mesh_groups；不能每个 rank 自己 new_group。
    """
    todo('T21')
