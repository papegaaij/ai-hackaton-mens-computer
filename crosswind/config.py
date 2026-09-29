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
WIND_SHIFT_MIN = 1.5     # s: shortest time before the wind picks a new random target
WIND_SHIFT_MAX = 4.0     # s: longest time before the wind picks a new random target
WIND_RESPONSE = 1.0      # s: time constant with which the wind eases toward its target

BEDROCK = 12             # bottom rows that can never be destroyed

PLAYER_HP = 100
PLAYER_FUEL = 240.0      # px of movement in a full tank (2 s of driving)
FUEL_REGEN = 40.0        # px/s the tank refills
PLAYER_RADIUS = 28
PLAYER_SPEED = 120.0     # px/s
MAX_CLIMB = 5            # px of height a tank can climb per px moved
JUMP_SPEED = 175.0       # px/s upward launch speed of a jump (about 50 px high)
JUMP_FUEL = 60.0         # fuel a jump costs; steering in the air costs fuel like driving
ANGLE_SPEED = 45.0       # deg/s at full speed
POWER_SPEED = 35.0       # power units/s at full speed
AIM_FINE = 0.2           # fraction of full aim speed when a key is first pressed
AIM_RAMP = 0.6           # s of holding a key to reach full aim speed

ENERGY_MAX = 100.0       # a full energy bar; each weapon costs part of it
ENERGY_REGEN = 30.0      # energy/s refilled (full bar in ~3.3 s)
FIRE_COOLDOWN = 0.25     # s between any two shots of the same player
START_COUNTDOWN = 3.0    # s before the first shot is allowed
END_DELAY = 1.5          # s after the last kill before the game is over; shots in the air still land
