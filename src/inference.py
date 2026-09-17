import os
import glob
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from models.unet import UNet

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    images_dir = '/content/dataset/oxford-iiit-pet/images'
    test_txt_path = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/test_unet.txt'
    model_path = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/saved_models/best_unet_model.pth'
    output_csv_path = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/submission_unet.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded.")

    with open(test_txt_path, 'r') as f:
        test_images = [line.strip().split('.')[0] for line in f.readlines() if line.strip()]

    results = []
    print(f"Running inference on {len(test_images)} images...")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for img_name in tqdm(test_images):
        search_pattern = os.path.join(images_dir, f"{img_name}.*")
        matching_files = [f for f in glob.glob(search_pattern) if not f.endswith(('.xml', '.mat'))]

        if not matching_files:
            results.append({'image_id': img_name, 'encoded_mask': ''})
            continue

        img_path = matching_files[0]

        try:
            img = Image.open(img_path).convert("RGB")
            original_w, original_h = img.size
        except Exception as e:
            results.append({'image_id': img_name, 'encoded_mask': ''})
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output_raw = model(input_tensor)
            prob_raw = torch.sigmoid(output_raw).squeeze()

        img_flipped = TF.hflip(img)
        input_tensor_flipped = transform(img_flipped).unsqueeze(0).to(device)
        with torch.no_grad():
            output_flipped = model(input_tensor_flipped)
            prob_flipped_wrong_way = torch.sigmoid(output_flipped).squeeze()
            prob_flipped = torch.flip(prob_flipped_wrong_way, dims=[1])

        pred_probs = (prob_raw + prob_flipped) / 2.0
        pred_mask = (pred_probs > 0.5).cpu().numpy().astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
        pred_mask_resized = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        rle = rle_encode(pred_mask_resized)
        results.append({'image_id': img_name, 'encoded_mask': rle})

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Done. Saved to: {output_csv_path}")

if __name__ == '__main__':
    main()
