from megatron_mini.exercise import todo

def topk_route(logits,k):
    """T42: logits[N,E] -> (expert_ids[N,k], weights[N,k], aux_loss)。
    在FP32 softmax上topk，再对选中的k个prob归一化；weights保留router梯度。
    aux=E*sum(mean_full_probs * (token_assignment_counts/(N*k)))。
    离散counts不求导。k=1归一化权重恒1，主loss没有gate梯度是该约定的结果；
    因此主课用top2并单测aux梯度。没有capacity/drop/padding：dropless。
    """
    todo('T42')
