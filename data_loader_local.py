import gcsfs
import pandas as pd
import torch
import io
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Define dataset file paths
csv_path = "trainLabels.csv"
image_folder = "resized_train"

def load_labels():
    '''Load the labels from the csv file and map them to categories'''
    df = pd.read_csv(csv_path)
    label_mapping = {0: "Normal", 1: "Mild Diabetic Retinopathy", 2: "Mild Diabetic Retinopathy", 
                     3: "Severe Diabetic Retinopathy", 4: "Severe Diabetic Retinopathy"}
    df['category'] = df['level'].map(label_mapping)
    category_mapping = {"Normal": 0, "Mild Diabetic Retinopathy": 1, "Severe Diabetic Retinopathy": 2}
    df['category_id'] = df['category'].map(category_mapping).fillna(-1).astype(int)
    return df

class LocalImageDataset(Dataset):
    '''Custom PyTorch dataset for loading local images and their corresponding labels.'''

    def __init__(self, df, folder_path, transform=None):
        self.df = df
        self.folder_path = folder_path
        self.transform = transform
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        '''Load and return an image and its corresponding label.'''
        row = self.df.iloc[idx]
        image_name = row['image']
        label = int(row['category_id'])
        image = Image.open(f"{self.folder_path}/{image_name}.jpeg").convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
def compute_mean_std(dataset):
    '''Compute the mean and standard deviation of the dataset for normalization.'''
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1) # Flatten image pixels

        mean += images.mean(dim=[0, 2]) * batch_samples
        std += images.std(dim=[0, 2]) * batch_samples
        num_samples += batch_samples
    
    mean /= num_samples
    std /= num_samples
    return mean, std

def get_dataloaders(batch_size=32, train_ratio=0.8, val_ratio=0.1, num_workers=4, pin_memory=False):
    '''Load the dataset and split it into training, validation, and test sets.'''
    df = load_labels()
    dataset = LocalImageDataset(df, image_folder, transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    mean, std = compute_mean_std(dataset)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    dataset = LocalImageDataset(df, image_folder, transform)
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size- val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader,val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    # Get a batch of training images and labels
    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}, Labels: {labels}")
