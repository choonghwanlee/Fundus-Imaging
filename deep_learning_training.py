import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import f1_score
import time
import os
from data_loader import get_dataloaders


from torchvision import transforms


# Define augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained VGG mean/std
])

def get_model(model_name="resnet18"):
    """Selects the pre-trained model based on the input."""
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "vgg16" or model_name == "vgg16_augmentation":
        model = models.vgg16(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    # Adjust the final fully connected layer
    if "resnet" in model_name:
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif "vgg" in model_name:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
    elif "densenet" in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, 3)
    
    return model.to(device)

def train_model(model, train_loader, optimizer, criterion, num_epochs=5):
    model.train()
    previous_end_time = time.time()  # Track the end time of the previous batch
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

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
    # Freeze all layers except the fully connected (fc) layer
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

def train_and_evaluate_all_models(train_loader, test_loader, model_names=["vgg16", "vgg16_augmentation","densenet121","resnet18" ]):
    best_f1 = 0
    best_model = None
    best_model_name = ""
    
    # Set up the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Specify the directory on your local machine to save models
    model_save_path = "./models"  # Modify this path if needed
    os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Store the models to save all of them
    saved_models = {}

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

            # Apply augmentation only for the vgg_augmentation model
            if model_name == "vgg16_augmentation":
                # Replace the transformation for this model with augmentation
                train_loader.dataset.transform = augmentation_transforms
            
            # Fine-tune the model
            model = fine_tune_model(model, model_name)

            # Set up optimizer
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

            # Train the model
            train_model(model, train_loader, optimizer, criterion, num_epochs=5)

            # Save the trained model
            torch.save(model.state_dict(), model_path)
            print(f"Saved {model_name}_model.pth")

        # Evaluate the model
        f1 = evaluate_model(model, test_loader)

        # Update best model if necessary
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

        # Store loaded/trained model
        saved_models[model_name] = model

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Print best model
    print(f"\nBest model: {best_model_name} with F1 Score: {best_f1:.4f}")

    return saved_models

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your data
    train_loader, test_loader = get_dataloaders(batch_size=32, num_workers=4, pin_memory=True)

    # Train and evaluate all models
    saved_models = train_and_evaluate_all_models(train_loader, test_loader)
