import torch

# -------------------------------------------------------------------
# Data & Text Settings
# -------------------------------------------------------------------
TRAIN_IMAGE_DIR       = "/path/to/training/images"
TRAIN_TEXT_DESC_FILE  = "/path/to/training/text_descriptions.json"
TEST_IMAGE_DIR       = "/path/to/test/images"
TEST_TEXT_DESC_FILE  = "/path/to/test/text_descriptions.json"
VAL_IMAGE_DIR         = "/path/to/validation/images"
VAL_TEXT_DESC_FILE    = "/path/to/validation/text_descriptions.json"

# -------------------------------------------------------------------
# Model Hyperparameters
# -------------------------------------------------------------------
IMAGE_SIZE       = (256, 256)  # H, W
FEATURE_DIM      = 64
TEXT_EMBED_DIM   = 768
HIDDEN_DIM       = 256

# -------------------------------------------------------------------
# Training Hyperparameters
# -------------------------------------------------------------------
EPOCHS_STAGE1    = 100
EPOCHS_STAGE2    = 50
BATCH_SIZE       = 16
LEARNING_RATE    = 1e-4
ALPHA            = 1.0
BETA             = 0.5
GAMMA            = 0.6
ETA              = 0.5

# -------------------------------------------------------------------
# Checkpoints & Device
# -------------------------------------------------------------------
CHECKPOINT_DIR   = "checkpoints"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
