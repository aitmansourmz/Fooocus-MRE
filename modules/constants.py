STEPS_SPEED = 30
STEPS_QUALITY = 60
SWITCH_SPEED = 20
SWITCH_QUALITY = 40

MIN_SEED = 0
MAX_SEED = 2**63 - 1

# exclusive, needed by modules\expansion.py -> transformers\trainer_utils.py -> np.random.seed()
SEED_LIMIT_NUMPY = 2**32

# min - native SDXL resolution (1024x1024), max - determined by SDXL context size (2048)
MIN_MEGAPIXELS = 1.0
MAX_MEGAPIXELS = 4.0

# min - native SD 1.5 resolution (512x512), max - determined by SD 2.x context size (1024)
MIN_MEGAPIXELS_SD = 0.25
MAX_MEGAPIXELS_SD = 1.0
