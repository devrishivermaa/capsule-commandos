from Eval_metrics_gen_excel import *
import os
import platform
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
print(f"Using device: {device}")


def load_and_preprocess_image(full_path, target_size=(224, 224)):
    print(f"Loading and preprocessing image: {full_path}")
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(full_path).convert('RGB')  
    preprocessed_img = transform(img)
    return preprocessed_img


def load_test_data(test_dir, image_size=(224, 224)):
    image_filenames = [fname for fname in os.listdir(test_dir) if fname.lower().endswith('jpg')]
    print(f"Found images: {image_filenames}") 
    X_test = [load_and_preprocess_image(os.path.join(test_dir, fname), image_size) for fname in image_filenames]
    return X_test, image_filenames


def batch_process_test_data(test_dir, model, batch_size=32):
    print("Loading test data...")
    X_test, image_paths = load_test_data(test_dir)

    # Check if any images were loaded
    if len(X_test) == 0:
        print("No images loaded. Exiting test processing.")
        return [], []

    y_pred_probs = []

    print("Starting batch processing for test images...")
    for start in range(0, len(X_test), batch_size):
        end = start + batch_size
        print(f"Processing batch: {start} to {end}")

        X_batch = X_test[start:end]
        
        # Check if the batch has valid images
        if len(X_batch) == 0:
            print("No valid images in this batch, skipping...")
            continue
        
        X_batch = torch.stack(X_batch).to(device)

        with torch.no_grad():
            y_batch_pred = torch.softmax(model(X_batch), dim=1).cpu().numpy()
        y_pred_probs.append(y_batch_pred)

    if len(y_pred_probs) == 0:
        print("No predictions were made. Exiting test processing.")
        return [], []

    y_pred_probs = np.vstack(y_pred_probs)
    print("Finished batch processing for test data.")
    return y_pred_probs, image_paths


num_classes = 10
print("Initializing model...")
model = timm.create_model('davit_small', pretrained=True, num_classes=num_classes)
model = model.to(device)


model_path = 'davit_small1.pth'
print(f"Loading model weights from {model_path}")
model.load_state_dict(torch.load(model_path))  
model = model.eval()
print("Model is ready for evaluation.")


test_dir = 'C:\\Users\\devri\\OneDrive\\Desktop\\new_cvip\\Dataset\\test\\Images'
print(f"Test directory: {test_dir}")

print("Starting test data evaluation...")
y_test_pred, test_image_paths = batch_process_test_data(test_dir, model=model)

output_test_predictions = "test_excel_new_final.xlsx"
print(f"Saving test predictions to Excel: {output_test_predictions}")
save_predictions_to_excel(test_image_paths, y_test_pred, output_test_predictions)
print("Test predictions saved.")
