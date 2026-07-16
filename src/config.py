import exprim.anchor as anchor
from dataclasses import field, dataclass

#@dataclass(frozen=True)
#class DeviceConfig(anchor.BaseConfig):
#    cuda: str = 'cuda'

@dataclass(frozen=True)
class KarateClubDSCFG(anchor.BaseConfig):
    name: str = "KarateClub"
    num_features: int  = 34
    num_classes: int = 4

@dataclass(frozen=True)
class gnnCFG(anchor.BaseConfig):
    hidden_dim: int = 16
    epochs: int = 200

@dataclass(frozen=True)
class optimCFG(anchor.BaseConfig):
    lr: float = 0.01
    weight_decay: float = 5e-4

#_: Main config
@dataclass(frozen=True)
class mainCFG:
    seed: int = 42
    device: str = "cuda"
    dataset: KarateClubDSCFG = field(default_factory=KarateClubDSCFG)
    
    @dataclass(frozen=True)
    class archCFG(anchor.BaseConfig):
        gnn: gnnCFG = field(default_factory=gnnCFG)
        optim: optimCFG = field(default_factory=optimCFG)

    arch: archCFG = field(default_factory=archCFG)
