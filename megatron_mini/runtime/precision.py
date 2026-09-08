from dataclasses import dataclass,asdict
from megatron_mini.exercise import todo

@dataclass
class DynamicScale:
    value: float=65536.0
    growth_interval: int=2000
    good_steps: int=0
    def update(self,found_inf):
        if found_inf:
            self.value=max(self.value/2,1.0);self.good_steps=0
        else:
            self.good_steps+=1
            if self.good_steps>=self.growth_interval:
                self.value*=2;self.good_steps=0
    def state_dict(self): return asdict(self)
    def load_state_dict(self,state):
        for key in ('value','growth_interval','good_steps'): setattr(self,key,state[key])


def autocast_context(device_type,precision):
    """T47: fp32 nullcontext，bf16/fp16 torch.autocast。FP32 master params/Adam
    moments不直接转半精度；loss/grad reduction FP32。FP8显式ValueError。
    本课程fp16运行实验要求CUDA；CPU可运行bf16可用算子或测scaler逻辑。
    PP激活传输dtype必须与真实output一致，由T59传输协议负责。
    """
    todo('T47')


def unscale_and_check(parameters,scale,axes):
    """T48: grad原地除scale一次，检查isfinite。axes上归约 found_inf OR，
    返回全局相同的bool；所有rank决定是否跳过这次optimizer.step。
    在clip前unscale；整个microbatch窗口scale固定；不得每microbatch update。
    axes覆盖共同step的TP/PP/DP/CP/EP成员，哪怕某个rank没有grad也参与。
    参数为空时调用者须提供至少一个设备占位tensor，或在runtime提前拒绝。
    """
    todo('T48')
