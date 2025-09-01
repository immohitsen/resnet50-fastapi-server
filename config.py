# config.py

# Data aur Model Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2  # 2 classes: Cancer aur Normal
EPOCHS = 10  # Training ke rounds

# Model ka naam
MODEL_NAME = "microsoft/resnet-50"

# Saved model ka path
MODEL_WEIGHTS_PATH = "saved_model/fine_tuned_resnet_weights.h5"