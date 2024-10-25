#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:38:04 2024

@author: arpan
"""

from Eval_metrics_gen_excel import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, recall_score, f1_score, balanced_accuracy_score
from torchvision import transforms
from torchvision.io import read_image
import os
import timm
from PIL import Image
import platform
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
def load_and_preprocess_image(full_path, target_size=(224, 224)):
    
    transform = transforms.Compose([
        transforms.Resize(target_size),              
        transforms.ToTensor(),                        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    img = Image.open(full_path).convert('RGB')  
    preprocessed_img = transform(img)
    return preprocessed_img



def get_data(excel_path, base_dir, image_size=(224, 224)):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    
    if platform.system() == 'Windows':
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('/', os.sep))
    else:
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('\\', os.sep))
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    X = np.array([load_and_preprocess_image(os.path.join(base_dir, path), image_size) for path in df['image_path'].values])
    y = df[class_columns].values
    return X, y, df


def load_test_data(test_dir, image_size=(224, 224)):
    image_filenames = [fname for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))] #creates my file_names array for test set 
    X_test = np.array([load_and_preprocess_image(os.path.join(test_dir, fname), image_size) for fname in image_filenames])
    return X_test, image_filenames


base_dir = '/home/arpan/VisionWorkspace/CapsuleVision/Dataset'
num_classes = 10  
image_size=(224,224)
model = timm.create_model('davit_small', pretrained=True, num_classes=num_classes)
model = model.to(device) 
model_path = os.path.join(os.getcwd(), '../../model_davit_small_epoch_20.pth')
model.load_state_dict(torch.load(model_path))  
#val_excel_path = os.path.join(os.getcwd(), 'Dataset', 'validation', 'validation_data.xlsx')
val_excel_path = '/home/arpan/VisionWorkspace/CapsuleVision/Dataset/validation/validation_data.xlsx'

X_val, y_val, val_df = get_data(val_excel_path, base_dir=base_dir, image_size=image_size)
model = model.eval()

# y = []
# batchsize = 128
# for batch in range(len(X_val)//batchsize +1):
#     start = batch*batchsize
#     end = (batch+1)*batchsize
#     if batch%10==0:
#         print("last batch ends at : {}".format(end))
#     x_temp = X_val[start:end].copy()
#     x_temp = torch.tensor(x_temp).to(device)
#     with torch.no_grad():
#         y_temp = model(x_temp).cpu()
#     y.append(y_temp)
    
# y_val_pred = torch.cat(y, axis=0).numpy()
# df = generate_metrics_report(y_val, y_val_pred)
# print(df)
# output_val_predictions="davit_small_validation_excel.xlsx"
# save_predictions_to_excel(val_df['image_path'].values, y_val_pred, output_val_predictions)

# For Test data - uncomment when you have test data
test_path = "/home/arpan/VisionWorkspace/CapsuleVision/Dataset/test/Images"
X_test, image_paths = load_test_data(test_path, image_size=image_size)
y = []
batchsize = 128
for batch in range(len(X_test)//batchsize +1):
    start = batch*batchsize
    end = (batch+1)*batchsize
    if batch%10==0:
        print("last batch ends at : {}".format(end))
    x_temp = X_test[start:end].copy()
    x_temp = torch.tensor(x_temp).to(device)
    with torch.no_grad():
        y_temp = model(x_temp).cpu()
    y.append(y_temp)
    
y_test_pred = torch.cat(y, axis=0).numpy()
#y_test_pred = model.predict(X_test)
output_test_predictions="test_excel.xlsx"
save_predictions_to_excel(image_paths, y_test_pred, output_test_predictions)

