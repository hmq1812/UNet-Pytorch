from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import MaskDataset
from utils import *

model.eval()   # Set model to the evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_fns = glob("./test/*jpg")
mask_dir = "./test"

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_set = MaskDataset(test_fns, mask_dir, test_transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)


# Get the first batch
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

# Predict
pred = model(inputs)
# The loss functions include the sigmoid function.
pred = F.sigmoid(pred)
pred = pred.data.cpu().numpy()
print(pred.shape)

# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [masks_to_colorimg(x) for x in pred]

plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])