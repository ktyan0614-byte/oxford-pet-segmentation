import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, txt_file, is_train=True, target_size=(256, 256)):
        self.target_size = target_size
        import glob
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.is_train = is_train
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')

        self.file_names = []

        print(f"Loading dataset... (Train Mode: {self.is_train})")
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    img_name = line.split(' ')[0]

                    search_pattern = os.path.join(self.images_dir, f"{img_name}.*")
                    matching_imgs = [img for img in glob.glob(search_pattern) if not img.endswith(('.xml', '.mat'))]
                    mask_path = os.path.join(self.masks_dir, f"{img_name}.png")

                    if matching_imgs and os.path.exists(mask_path):
                        self.file_names.append((img_name, matching_imgs[0], mask_path))
                    else:
                        pass

        print(f"Loaded {len(self.file_names)} valid samples.")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name, img_path, mask_path = self.file_names[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256), interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_np = np.array(mask)
        binary_mask = np.where(mask_np == 1, 1.0, 0.0).astype(np.float32)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

        return image_tensor, mask_tensor
