from ultralytics import YOLO
import torch
import yaml
import os
import json

# Get absolute paths
base_dir = os.path.abspath('datasets/Ball Dataset VAI.v1i.coco')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')

# Create dataset configuration
dataset_config = {
    'path': base_dir,  # dataset root dir
    'train': train_dir,  # train images (absolute path)
    'val': val_dir,  # val images (absolute path)
    'names': {
        0: 'ball'  # class names
    },
    'nc': 1,  # number of classes
    'task': 'detect',  # task type
}

# Verify directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found at {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found at {val_dir}")

# Verify images exist
train_images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
val_images = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not train_images:
    raise FileNotFoundError(f"No images found in training directory {train_dir}")
if not val_images:
    raise FileNotFoundError(f"No images found in validation directory {val_dir}")

print(f"Found {len(train_images)} training images and {len(val_images)} validation images")

# Save dataset configuration
with open(os.path.join(base_dir, 'data.yaml'), 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with additional parameters to handle potential issues
results = model.train(
    data=os.path.join(base_dir, 'data.yaml'),  # path to data config file
    epochs=100,  # number of epochs
    imgsz=640,  # image size
    batch=4,  # reduced batch size for CPU training
    name='ball_detection',  # experiment name
    patience=50,  # early stopping patience
    save=True,  # save checkpoints
    device='cpu',  # use CPU for training
    workers=0,  # reduce number of workers to avoid potential issues
    verbose=True,  # enable verbose output
    exist_ok=True,  # overwrite existing experiment
    pretrained=True,  # use pretrained weights
    optimizer='Adam',  # use Adam optimizer
    lr0=0.001,  # initial learning rate
    lrf=0.01,  # final learning rate
    momentum=0.937,  # SGD momentum
    weight_decay=0.0005,  # optimizer weight decay
    warmup_epochs=3.0,  # warmup epochs
    warmup_momentum=0.8,  # warmup momentum
    warmup_bias_lr=0.1,  # warmup bias learning rate
    box=7.5,  # box loss gain
    cls=0.5,  # cls loss gain
    dfl=1.5,  # dfl loss gain
    close_mosaic=10,  # disable mosaic augmentation for final epochs
    plots=True  # generate plots
) 