import exprim.anchor as anchor
from dataclasses import dataclass

@dataclass(frozen=True)
class DeviceConfig(anchor.BaseConfig):
    cuda: str = 'cuda'
