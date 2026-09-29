from dataclasses import dataclass

from crosswind import config


@dataclass
class Player:
    index: int
    x: float
    y: float
    angle: float           # degrees, 0 = right, 90 = straight up, 180 = left
    power: float = 60.0    # 0..100
    hp: float = config.PLAYER_HP
    fuel: float = config.PLAYER_FUEL
    weapon: int = 0
    reload: float = 0.0    # seconds until this player can fire again
    vy: float = 0.0        # vertical speed while airborne (px/s, down is positive)
    airborne: bool = False

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def ready(self) -> bool:
        return self.alive and self.reload <= 0
