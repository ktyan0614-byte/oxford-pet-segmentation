import os
import glob
import torch
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

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

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    set_seed(42)

    images_dir = '/content/dataset/oxford-iiit-pet/images'
    TEST_TXT_PATH = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/test_res_unet.txt'
    MODEL_PATH = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/saved_models/best_resnet_model.pth'
    OUTPUT_CSV_PATH = '/content/drive/MyDrive/DL_Lab2_H24111269_顏愷廷/submission_resnet_final.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ResNet34_UNet(n_classes=1, pretrained=False).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded.")
    else:
        print(f"Model weights not found: {MODEL_PATH}")
        return

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(TEST_TXT_PATH):
        print(f"Test list not found: {TEST_TXT_PATH}")
        return

    with open(TEST_TXT_PATH, 'r') as f:
        test_images = [line.strip() for line in f.readlines() if line.strip()]

    results = []
    print(f"Running inference on {len(test_images)} images...")

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
        except Exception:
            results.append({'image_id': img_name, 'encoded_mask': ''})
            continue

        with torch.no_grad():
            input_tensor = transform(img).unsqueeze(0).to(device)
            output_raw = model(input_tensor)
            prob_raw = torch.sigmoid(output_raw).squeeze()

            # TTA: horizontal flip
            img_flipped = TF.hflip(img)
            input_tensor_flipped = transform(img_flipped).unsqueeze(0).to(device)
            output_flipped = model(input_tensor_flipped)
            prob_flipped_wrong_way = torch.sigmoid(output_flipped).squeeze()
            prob_flipped = torch.flip(prob_flipped_wrong_way, dims=[1])

            pred_probs = (prob_raw + prob_flipped) / 2.0
            pred_mask = (pred_probs > 0.5).cpu().numpy().astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

        pred_mask_resized = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        rle = rle_encode(pred_mask_resized)
        results.append({'image_id': img_name, 'encoded_mask': rle})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Done. Saved to: {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    main()
