from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 64
    seq_len: int = 16
    hidden_size: int = 32
    num_heads: int = 4
    num_layers: int = 4
    ffn_size: int = 64
    seed: int = 2026

    def validate(self, tp: int = 1, pp: int = 1):
        for name, value in asdict(self).items():
            if name != "seed" and value <= 0:
                raise ValueError(f"{name} must be positive")
        if tp < 1 or pp < 1:
            raise ValueError("tp and pp must be positive")
        for name in ("vocab_size", "num_heads", "ffn_size"):
            if getattr(self, name) % tp:
                raise ValueError(f"{name} must be divisible by tp={tp}")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers % pp:
            raise ValueError("num_layers must be divisible by pp (equal contiguous stages)")
        return self

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads
