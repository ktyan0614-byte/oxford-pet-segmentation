import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.resnet34_unet import ResNet34_UNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.7, weight_dice=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice_score.mean()
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss

def main():
    set_seed(42)

    DATASET_ROOT = '/content/dataset/oxford-iiit-pet'
    TRAIN_VAL_TXT = os.path.join(DATASET_ROOT, 'annotations/trainval.txt')
    SAVE_DIR = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/saved_models'

    BEST_MODEL_NAME = 'best_resnet_model.pth'
    LAST_MODEL_NAME = 'last_resnet_model.pth'

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    BATCH_SIZE = 16
    EPOCHS = 80
    LEARNING_RATE = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ResNet34_UNet(n_classes=1, pretrained=False).to(device)

    full_train_dataset = OxfordPetDataset(DATASET_ROOT, TRAIN_VAL_TXT, is_train=True)
    full_val_dataset   = OxfordPetDataset(DATASET_ROOT, TRAIN_VAL_TXT, is_train=False)

    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_dataset = Subset(full_train_dataset, indices[:train_size])
    val_dataset = Subset(full_val_dataset, indices[train_size:])
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    scaler = torch.amp.GradScaler('cuda')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    criterion = BCEDiceLoss(weight_bce=0.7, weight_dice=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_dice = 0.0
    best_model_path = os.path.join(SAVE_DIR, BEST_MODEL_NAME)

    print("Training from scratch.")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        for images, masks in tqdm(train_loader, desc=f"  [Train]"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        total_val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"  [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)

                pred_probs = torch.sigmoid(outputs).squeeze(1)
                pred_mask = (pred_probs > 0.5).float()
                true_mask = masks.squeeze(1)

                intersection = (pred_mask * true_mask).sum(dim=(1, 2))
                union = pred_mask.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2))
                dice_batch = (2. * intersection + 1e-6) / (union + 1e-6)
                total_val_dice += dice_batch.sum().item()

        avg_val_dice = total_val_dice / len(val_dataset)
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {current_lr:.6f}")

        scheduler.step()

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, LAST_MODEL_NAME))
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (Dice: {best_val_dice:.4f})")

    print(f"\nTraining complete. Best Dice: {best_val_dice:.4f}")

if __name__ == '__main__':
    main()
