from dataclasses import dataclass


@dataclass(frozen=True)
class FireAction:
    """A complete shot. Both human and (future) AI controllers produce this."""
    angle: float   # degrees, 0..180
    power: float   # 0..100
    weapon: int
