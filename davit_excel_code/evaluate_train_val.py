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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(full_path).convert('RGB')  
    preprocessed_img = transform(img)
    return preprocessed_img

def batch_process_data(excel_path, base_dir, model, batch_size=32):
    print(f"Reading data from Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    
    if platform.system() == 'Windows':
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('/', os.sep))
    else:
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('\\', os.sep))

    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 
                     'Foreign Body', 'Lymphangiectasia', 'Normal', 
                     'Polyp', 'Ulcer', 'Worms']
    y_true = df[class_columns].values
    y_pred_probs = []

    print("Starting batch processing for images...")
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        print(f"Processing batch: {start} to {end}")

        batch_paths = df['image_path'].values[start:end]
        X_batch = [load_and_preprocess_image(os.path.join(base_dir, path)) for path in batch_paths]
        
        X_batch = [img for img in X_batch if img is not None]
        if len(X_batch) == 0:
            print("No valid images in this batch, skipping...")
            continue
        
        X_batch = torch.stack(X_batch).to(device)
        
        with torch.no_grad():
            y_batch_pred = torch.softmax(model(X_batch), dim=1).cpu().numpy()
        y_pred_probs.append(y_batch_pred)

    y_pred_probs = np.vstack(y_pred_probs)
    print("Finished batch processing.")
    return y_true, y_pred_probs, df


num_classes = 10  
print("Initializing model...")
model = timm.create_model('davit_small', pretrained=True, num_classes=num_classes)
model = model.to(device)


model_path = 'davit_small1.pth'
print(f"Loading model weights from {model_path}")
model.load_state_dict(torch.load(model_path))  
model = model.eval()
print("Model is ready for evaluation.")


base_dir = 'C:\\Users\\devri\\OneDrive\\Desktop\\new_cvip\\Dataset'
val_excel_path = 'Dataset\\training\\training_data.xlsx' #replace with validation to get validation results


print("Loading validation data...")
y_val, y_val_pred, val_df = batch_process_data(val_excel_path, base_dir=base_dir, model=model)

print("Evaluation complete. Generating metrics report...")
df = generate_metrics_report(y_val, y_val_pred)
print(df)

output_val_predictions = "davit_train_excel_final.xlsx"
print(f"Saving predictions to Excel: {output_val_predictions}")
save_predictions_to_excel(val_df['image_path'].values, y_val_pred, output_val_predictions)
print("Predictions saved.")

# Uncomment for test data processing
'''
test_path = os.path.join(os.getcwd(), 'Dataset', 'test')
X_test, image_paths = load_test_data(test_path)
print("Starting test data evaluation...")
y_test_pred = model.predict(X_test)
output_test_predictions = "test_excel.xlsx"
save_predictions_to_excel(image_paths, y_test_pred, output_test_predictions)
print("Test predictions saved.")
'''
