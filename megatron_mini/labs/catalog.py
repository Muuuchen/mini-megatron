from dataclasses import dataclass
from megatron_mini.runtime.config import RecipeConfig
from megatron_mini.offload.config import OffloadConfig

# case -> required exercise IDs. Empty list means provided plumbing, NOT student feature pass.
CASES={
 'transport':[], 'topology':['T21'],
 'sp':['T22','T23','T24','T25'], 'cp':['T26','T27'], 'ring':['T28','T29'],
 'pipeline':['T30','T31'], 'interleaved':['T31','T32'],
 'recompute':['T33'], 'bucket':['T34'], 'flat':['T35'],
 'zero1':['T35','T36'], 'zero2':['T35','T37'], 'fsdp':['T35','T38','T39'],
 'moe':['T42','T43','T44','T45','T46'], 'precision':['T47','T48'],
 'residency':['T61'], 'zero3':['T35','T38','T39','T62'],
 'offload_transfer':['T65'], 'offload_optimizer':['T63','T65'],
 'offload_param':['T64','T65'], 'offload_activation':['T65','T66'],
 'sac':['T67','T68'], 'zero3_sac':['T35','T38','T39','T62','T67','T69','T70'],
 'offload_checkpoint':['T63','T65','T72'],
 'roles':['T49'], 'model':['T51','T52','T53'], 'text':['T54'],
 'overlap':['T41','T56','T57','T58'], 'reshard':['T60'], 'checkpoint':['T50']
}
# These are target integration recipes. build_training_runtime T55 deliberately remains a task.
RECIPES={
 'dense_sp': RecipeConfig(tp=2,sp=True),
 'dense_tp_zero2': RecipeConfig(tp=2,dp=2,sp=True,sharding='zero2'),
 'long_context': RecipeConfig(cp=2,cp_mode='ring',recompute='attention'),
 'pipeline_1f1b': RecipeConfig(pp=2,schedule='1f1b'),
 'pipeline_interleaved': RecipeConfig(pp=2,schedule='interleaved',virtual_stages=2),
 'fsdp_recompute': RecipeConfig(dp=2,sharding='fsdp',recompute='block'),
 'moe_ep': RecipeConfig(dp=2,ep=2,model='moe'),
 'moe_ep_fsdp': RecipeConfig(dp=4,ep=2,model='moe',sharding='fsdp'),
 'moe_tp_ep': RecipeConfig(tp=2,dp=2,ep=2,sp=True,model='moe'),
 'torch_fsdp2': RecipeConfig(dp=2,sharding='torch_fsdp2'),
 'zero3': RecipeConfig(dp=2,sharding='zero3'),
 'zero3_cpu': RecipeConfig(dp=2,sharding='zero3',offload=OffloadConfig(optimizer='cpu',parameters='cpu')),
 'zero3_cpu_sac': RecipeConfig(dp=2,sharding='zero3',recompute='sac',offload=OffloadConfig(optimizer='cpu',parameters='cpu',activations='cpu')),
 'fsdp_cpu_sac': RecipeConfig(dp=2,sharding='fsdp',recompute='sac',offload=OffloadConfig(optimizer='cpu',parameters='cpu',activations='cpu')),
 'sac_dense': RecipeConfig(recompute='sac'),
}

def recipe_tasks(c):
    # Uniform list: early rejection before creating any communicator.
    ids={'T55','T21','T49','T01','T03','T04','T05','T06','T07','T10','T11','T12','T13'}
    if c.tp>1: ids.update(('T08','T09'))
    if c.sp: ids.update(CASES['sp'])
    if c.cp>1: ids.update(CASES['ring' if c.cp_mode=='ring' else 'cp'])
    if c.pp>1:
        ids.add('T59')
        ids.update(CASES['interleaved' if c.schedule=='interleaved' else 'pipeline'])
    else: ids.add('T15')
    if c.sharding=='zero1': ids.update(CASES['zero1'])
    if c.sharding=='zero2': ids.update(CASES['zero2'])
    if c.sharding=='zero3': ids.update(CASES['zero3']);ids.add('T61')
    if c.sharding=='fsdp': ids.update(CASES['fsdp']);ids.add('T40')
    if c.model=='moe': ids.update(CASES['moe'])
    if c.model=='llama': ids.update(CASES['model'])
    if c.precision!='fp32': ids.update(CASES['precision'])
    if c.recompute=='sac':
        ids.update(CASES['sac'])
        if c.sharding in ('zero3','fsdp'): ids.update(('T69','T70'))
    elif c.recompute!='none': ids.add('T33')
    if c.offload.enabled:
        ids.update(('T65','T71','T72'))
        if c.offload.optimizer=='cpu': ids.add('T63')
        if c.offload.parameters=='cpu': ids.add('T64')
        if c.offload.activations=='cpu': ids.add('T66')
    for axis in c.overlap:
        ids.add({'dp':'T34','fsdp':'T41','tp':'T56','ep':'T57','cp':'T58','pp':'T59'}[axis])
    return sorted(ids)
