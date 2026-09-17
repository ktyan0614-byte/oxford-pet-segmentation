import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

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

def evaluate(model, val_loader, device):
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_dice += dice_score(outputs, masks)
    return total_dice / len(val_loader)

if __name__ == '__main__':
    DATASET_ROOT = '/content/dataset/oxford-iiit-pet'
    TRAIN_VAL_TXT = os.path.join(DATASET_ROOT, 'annotations/trainval.txt')
    BATCH_SIZE = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_val_dataset = OxfordPetDataset(DATASET_ROOT, TRAIN_VAL_TXT, is_train=False)
    dataset_size = len(full_val_dataset)
    train_size = int(0.8 * dataset_size)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    val_dataset = Subset(full_val_dataset, indices[train_size:])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    unet = UNet(n_channels=3, n_classes=1).to(device)
    unet.load_state_dict(torch.load('/content/drive/MyDrive/DL_Lab2/saved_models/best_unet_model.pth', map_location=device))
    print(f"UNet Val Dice: {evaluate(unet, val_loader, device):.4f}")

    resnet = ResNet34_UNet(n_classes=1).to(device)
    resnet.load_state_dict(torch.load('/content/drive/MyDrive/DL_Lab2/saved_models/best_resnet_model.pth', map_location=device))
    print(f"ResNet34_UNet Val Dice: {evaluate(resnet, val_loader, device):.4f}")
