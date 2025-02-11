import torch
import pandas as pd
import os
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score
from data_loader import get_dataloaders, load_labels

def calculate_native_distribution():

    
    df = load_labels()
    
    label_mapping = {
        0: "Normal", 
        1: "Mild Diabetic Retinopathy", 
        2: "Mild Diabetic Retinopathy",
        3: "Severe Diabetic Retinopathy", 
        4: "Severe Diabetic Retinopathy"
    }
    df['category'] = df['level'].map(label_mapping)

    # reduce to 3 classes
    category_mapping = {
        "Normal": 0, 
        "Mild Diabetic Retinopathy": 1, 
        "Severe Diabetic Retinopathy": 2
    }
    df['category_id'] = df['category'].map(category_mapping)
    
    class_counts = df['category_id'].value_counts()
    total_samples = len(df)
    class_probs = np.zeros(3)
    
    # Calculate class probabilities
    for i in range(3):
        class_probs[i] = class_counts.get(i, 0) / total_samples
    
    print("Class distribution:")
    for i, prob in enumerate(class_probs):
        print(f"Class {i}: {prob:.3f}")
    
    return class_probs

#  evaluate native model
def evaluate_native_model(class_probs, test_loader):
   
    all_preds = []
    all_labels = []
    
    # get the predicted class
    pred_class = np.argmax(class_probs)
    
    # test data
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())
        all_preds.extend([pred_class] * len(labels))
    
    # f1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Native Model F1 Score: {f1:.4f}")
    return f1

def save_native_model():
    # calculate class probabilities
    class_probs = calculate_native_distribution()
    
    #  load test data
    _, test_loader = get_dataloaders(batch_size=32)  
    f1_score = evaluate_native_model(class_probs, test_loader)
    
    # set up model saving
    os.makedirs('./models', exist_ok=True)
    
    save_dict = {
        'class_probs': torch.tensor(class_probs, dtype=torch.float32),
        'f1_score': f1_score
    }

    torch.save(save_dict, './models/native_model.pth')
    print(f"Native model saved to ./models/native_model.pth")
    print(f"Model performance - F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    save_native_model()