"""Game-wide constants. Units: pixels and seconds."""

WIDTH = 1280
HEIGHT = 720
FPS = 60

# Fixed physics step, so simulation is deterministic and can run headless.
PHYSICS_DT = 1.0 / 120.0

GRAVITY = 300.0          # px/s^2 (low-ish: alien planet)
MAX_LAUNCH_SPEED = 720.0  # px/s at power 100
WIND_MAX = 90.0          # px/s^2 at ground level
WIND_ALTITUDE_BOOST = 1.0  # wind is (1 + boost) times stronger at the top of the screen

BEDROCK = 12             # bottom rows that can never be destroyed

PLAYER_HP = 100
PLAYER_FUEL = 120.0      # px of movement per turn
PLAYER_RADIUS = 14
PLAYER_SPEED = 60.0      # px/s
MAX_CLIMB = 5            # px of height a tank can climb per px moved
ANGLE_SPEED = 45.0       # deg/s
POWER_SPEED = 35.0       # power units/s

RESOLVE_DELAY = 1.0      # seconds after an impact before the turn switches
