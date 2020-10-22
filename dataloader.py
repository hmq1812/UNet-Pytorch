import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


def toTensor(np_array, axis=(2,0,1)):
    return torch.tensor(np_array).permute(axis)
    

class MaskDataset(Dataset):
    def __init__(self, img_fns, mask_dir, transforms=None):
        self.img_fns = img_fns
        self.transforms = transforms
        self.mask_dir = mask_dir
        
    def __getitem__(self, idx):
        img_path = self.img_fns[idx]
        img_name = img_path.split("/")[-1].split(".")[0]
        mask_fn = f"{self.mask_dir}/{img_name}.png"

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_fn)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask!=2)*1.0
        mask = cv2.resize(mask, (224, 224))
        mask = np.reshape(mask, (224, 224, 1))
        mask = 1.0*(mask[:,:,0]>0.2)


        if img_name.startswith('cat'):
            mask_dog = np.zeros((224, 224, 1), dtype=np.float32)
            mask = np.dstack((mask, mask_dog))
        elif img_name.startswith('dog'):
            mask_cat = np.zeros((224, 224, 1), dtype=np.float32)
            mask = np.dstack((mask_cat, mask))

        if self.transforms:
            img = self.transforms(img)

        # img = toTensor(img)
        mask = toTensor(mask)
        
        return [img, mask]
            
    def __len__(self):
        return len(self.img_fns)