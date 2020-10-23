from glob import glob

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import MaskDataset
from model import Unet
from train_module import train_model

# Load data
img_fns = glob("./data/*jpg")

# Create train and val set
val_fns = []
train_fns = []
random_idx = np.random.randint(0, len(img_fns), size=1500)
for idx in range(len(img_fns)):
    if idx in random_idx:
        val_fns.append(img_fns[idx])
    else:
        train_fns.append(img_fns[idx])

mask_dir = "./data"

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_set = MaskDataset(train_fns, mask_dir, train_transform)
val_set = MaskDataset(val_fns, mask_dir, train_transform)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 32

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}


# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class = 2
model = Unet(num_class)

# freeze resnet layers
for i, child in enumerate(model.children()):
    if i <= 7:
        for param in child.parameters():
            param.requires_grad = False

model.to(device)

# summary(model, (3, 224, 224))

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=100)