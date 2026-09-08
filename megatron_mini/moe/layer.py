"""已连接路由/变长AllToAll/本地专家/反向回传；四个算法TODO需要学生填。"""
import torch
from torch import nn
from .router import topk_route
from .dispatch import pack,receive_splits,all_to_all,combine
from megatron_mini.exercise import todo

class ExpertParallelMoE(nn.Module):
    def __init__(self,router,local_experts,num_experts,topk,axis):
        super().__init__()
        self.router=router;self.experts=nn.ModuleDict({str(k):v for k,v in local_experts.items()})
        self.num_experts,self.topk,self.axis=num_experts,topk,axis
        if num_experts%axis.size: raise ValueError('experts must divide EP')

    def forward(self,x):
        shape=x.shape;x=x.reshape(-1,shape[-1])
        ids,weights,aux=topk_route(self.router(x),self.topk)
        payload,plan=pack(x,ids,self.num_experts,self.axis)
        recv=receive_splits(plan.send_splits,self.axis)
        rx=all_to_all(payload,plan.send_splits,recv,self.axis)
        eid=all_to_all(plan.expert_ids,plan.send_splits,recv,self.axis)
        # 不用临时全专家模型；只持有当前owner的专家。
        y=rx*0
        for name,expert in self.experts.items():
            positions=(eid==int(name)).nonzero(as_tuple=True)[0]
            y=y.index_copy(0,positions,expert(rx.index_select(0,positions)))
            # 空专家也产生明确的0梯度，所有EDP rank按相同顺序参加同步。
            y=y+sum((p.sum()*0 for p in expert.parameters()))
        returned=all_to_all(y,recv,plan.send_splits,self.axis)
        return combine(returned,plan,weights).view(shape),aux


def synchronize_moe_gradients(router_params,expert_params,dense_axis,edp_axis,ep_size):
    """T46: 等长local mean损失约定：router在dense-DP SUM/denseDP。
    expert先在EP dispatcher backward已合并不同源token贡献，再在EDP SUM/denseDP
    （不是只除EDP，也不在EP组同步不同专家）。denseDP=EP*EDP。
    此lab CP=1；CP>1由T49扩为dp_cp与edp_cp。零token专家仍参与。
    若外部loss已除global tokens，则两路均只SUM，禁止再平均。
    """
    todo('T46')
