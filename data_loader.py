import gcsfs
import pandas as pd
import torch
import io
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# Define dataset
bucket_name = "aipi540-cv"
csv_path = f"{bucket_name}/classification/trainLabels.csv"
image_folder = "classification/resized_train"

# Initialize Google Cloud Storage filesystem
fs = gcsfs.GCSFileSystem()

def load_labels():
    '''Load CSV from GCS and group labels'''
    with fs.open(csv_path) as f:
        df = pd.read_csv(f)
    
    # Change 0=>Normal(0) 1,2=>Mild(1), 3,4=>Servere(2)
    # Map labels
    label_mapping = {0:"Normal", 1:"Mild Diabetic Retinopathy", 2:"Mild Diabetic Retinopathy", 3:"Servere Diabetic Retinopathy", 4:"Servere Diabetic Retinopathy"}
    df['category'] = df['level'].map(label_mapping)

    # Convert to numerical labels 
    category_mapping = {"Normal":0,"Mild Diabetic Retinopathy":1,"Servere Diabetic Retinopathy":2}
    df['category_id'] = df['category'].map(category_mapping).fillna(-1).astype(int)

    return df

class GCSImageData(Dataset):
    def __init__(self, df, bucket_name, folder_path, transform=None):
        self.df = df
        self.bucket_name = bucket_name
        self.folder_path = folder_path
        self.transform = transform
        self.fs = gcsfs.GCSFileSystem()
        self.counter = 0  # Counter to track the number of prints

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image']
        label = row['category_id']

        # Ensure label is an integer (in case of float)
        label = int(label)

        # Read images from GCS
        image_path = f"{self.bucket_name}/{self.folder_path}/{image_name}.jpeg"
        with self.fs.open(image_path, "rb") as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

         # Print the first 10 image-label pairs
        if self.counter < 10:
            print(f"Image: {image_name}, Label: {label}")
            self.counter += 1

        return image, torch.tensor(label, dtype=torch.long)
    
def get_dataloaders(batch_size=32, split_ratio=0.8):
        '''Split data into training and testing set'''
        df = load_labels()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset = GCSImageData(df, bucket_name, image_folder, transform)

        # Split dataset
        train_size = int(split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)

        return train_loader, test_loader
    
if __name__ == "__main__":
        train_loader, test__loader = get_dataloaders()
        images, labels = next(iter(train_loader))
        print(f"Train batch shape: {images.shape}, Labels: {labels}")