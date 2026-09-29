"""Computer players. The modules here run while playing and only use numpy: no torch, and no physics.

The AI sees what a human sees (the HUD, the terrain, the tanks, shots in flight and where its own last
shot came down) and plays through the same inputs as a human. The networks it uses were trained with the
scripts in `crosswind.ai.train` and are loaded from `models/*.npz`.
"""
