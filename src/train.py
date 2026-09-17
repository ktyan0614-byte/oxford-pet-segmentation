import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.unet import UNet

def dice_score(pred_logits, target_masks, smooth=1e-5):
    with torch.no_grad():
        pred_probs = torch.sigmoid(pred_logits)
        preds = (pred_probs > 0.5).float()

        preds = preds.view(preds.size(0), -1)
        targets = target_masks.view(target_masks.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean().item()

def main():
    DATASET_ROOT = '/content/dataset/oxford-iiit-pet'
    TRAIN_VAL_TXT = os.path.join(DATASET_ROOT, 'annotations/trainval.txt')
    SAVE_DIR = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/saved_models'

    BATCH_SIZE = 16
    EPOCHS = 40
    LEARNING_RATE = 1e-4
    EXPERIMENT_NAME = "unet_aug"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full_train_dataset = OxfordPetDataset(DATASET_ROOT, TRAIN_VAL_TXT, is_train=True)
    full_val_dataset   = OxfordPetDataset(DATASET_ROOT, TRAIN_VAL_TXT, is_train=False)

    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    EPOCHS = 40
    best_val_dice = 0.0

    print(f"Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        total_val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                total_val_dice += dice_score(outputs, masks)

        avg_val_dice = total_val_dice / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        scheduler.step(avg_val_dice)

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last_model.pth'))

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_unet_model.pth'))
            print(f"New best model saved (Dice: {best_val_dice:.4f})")

    print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f}")

if __name__ == '__main__':
    main()
