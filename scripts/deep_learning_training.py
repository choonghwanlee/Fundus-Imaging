import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import f1_score
import time
import os
from scripts.data_loader_local import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_class_weights(train_loader):
    # Get the class distribution
    class_counts = np.zeros(3)  # Assuming 3 classes
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Calculate the weight for each class
    total_samples = len(train_loader.dataset)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

def get_oversampled_train_loader(train_loader, class_weights):
    # Calculate sample weights
    sample_weights = []
    for _, labels in train_loader:
        for label in labels:
            sample_weights.append(class_weights[label])
    
    # Create a sampler with the calculated sample weights
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create a new DataLoader with the sampler
    oversampled_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )
    
    return oversampled_train_loader

def get_model(model_name="resnet18"):
    """Selects the pre-trained model based on the input."""
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to reduce overfitting
            nn.Linear(512, 3)
        )
        
    elif model_name == "vgg16" :
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Sequential(
            nn.Linear(model.classifier[6].in_features, 512),
            nn.BatchNorm1d(512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    return model.to(device)

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=6,patience=2):
    model.train()
    previous_end_time = time.time()  # Track the end time of the previous batch

    
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}")
        running_loss = 0
        
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            start_time = time.time()  # Start time for batch processing
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            end_time = time.time()  # End time for batch processing
            batch_time = end_time - start_time  # Calculate batch processing time
            delay_time = start_time - previous_end_time  # Delay time from the previous batch

            print(f"Batch {batch_idx + 1}/{len(train_loader)} - Time: {batch_time:.4f} sec, Delay: {delay_time:.4f} sec")

            previous_end_time = end_time  # Update previous end time for the next batch

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"F1 Score: {f1:.4f}")
    return f1

def fine_tune_model(model, model_name):
    # Freeze all layers except the fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the fully connected layer
    if "resnet" in model_name:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif "vgg" in model_name or "densenet" in model_name:
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    return model

def train_and_evaluate_all_models(train_loader, val_loader,test_loader, model_names=["vgg16","densenet121","resnet18"]):
    best_f1 = 0
    best_model_name = ""

    # Dictionary to store F1 scores
    model_f1_scores = {}

    
    
    # Set up the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Specify the directory to save models
    model_save_path = "./models"  
    os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Store the models to save all of them
    saved_models = {}

    loss_dict = {}

    for model_name in model_names:
        model_path = os.path.join(model_save_path, f"{model_name}_model.pth")

        # Check if the model file exists
        if os.path.exists(model_path):
            print(f"Skipping training for {model_name} (model already exists).")
            model = get_model(model_name)
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
        else:
            print(f"\nTraining model: {model_name}...")
            model = get_model(model_name)

            
            
            # Fine-tune the model
            model = fine_tune_model(model, model_name)

            # Set up optimizer
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,weight_decay=1e-5)  # Add weight decay

            # Train the model
            train_losses, val_losses = train_model(model, train_loader, val_loader,test_loader, optimizer, criterion, num_epochs=5)

            # Save the trained model
            torch.save(model.state_dict(), model_path)
            print(f"Saved {model_name}_model.pth")

            # Store losses for visualization
            loss_dict[model_name] = {"train_loss": train_losses, "test_loss": val_losses}

        # Evaluate the model
        f1 = evaluate_model(model, test_loader)

        # Store F1 score for the model
        model_f1_scores[model_name] = f1

        # Update best model if necessary
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name

        # Store loaded/trained model
        saved_models[model_name] = model

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print best model
    print(f"\nBest model: {best_model_name} with F1 Score: {best_f1:.4f}")

    return saved_models, loss_dict, model_f1_scores



if __name__ == "__main__":
    print("start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("detect finish")
    # Load your data
    train_loader,val_loader,test_loader = get_dataloaders(batch_size=32, num_workers=4, pin_memory=True)

    # Inside your main function or where you're calling the data loaders
    class_weights = get_class_weights(train_loader)

    # Create the oversampled train loader
    oversampled_train_loader = get_oversampled_train_loader(train_loader, class_weights)
    print("traing start")
    
    # Train and evaluate all models
    saved_models, loss_dict, model_f1_scores = train_and_evaluate_all_models(oversampled_train_loader, val_loader, test_loader)

    # Print all model F1 scores
    print("\nF1 Scores for all models:")
    for model_name, f1 in model_f1_scores.items():
        print(f"{model_name}: {f1:.4f}")
