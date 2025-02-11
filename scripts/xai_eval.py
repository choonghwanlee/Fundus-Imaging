### Conducts evaluation of IoU
from torchvision import models, transforms
import torch.nn as nn
import torch
from google.cloud import storage
from pytorch_grad_cam import GradCAM
from PIL import Image
import cv2
import tifffile
import numpy as np
import io
import os


def load_model(model_path: str):
    """
    Load a fine-tuned VGG model from model path

    Input:
        model_path (str): absolute/relative path to trained VGG model (.pth file)
    """
    ## set model architecture with custom classification head
    vgg_model = models.vgg16(pretrained=False)
    vgg_model.classifier[6] = nn.Sequential(
        nn.Linear(vgg_model.classifier[6].in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,3)
    ) 
    ## load weights and set to eval mode
    vgg_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    vgg_model.eval()

    return vgg_model

def convert_to_gradcam(model):
    """
    Initalize a Grad-CAM explainer for the provided model

    Input:
        model: a trained / loaded PyTorch model
    """

    ## target gradients/activations from the last layer. as recommended by 'pytorch_grad_cam' package's README 
    target_layers = [model.features[-1]]

    ## create GradCAM instance for model explanation
    return GradCAM(model=model, target_layers=target_layers)

def preprocess_image(image):
    """
    Apply image pre-processing to prepare for VGG-16 model


    """
    ## same transform pipeline from model training
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3205, 0.2244,0.1613], std=[0.2996, 0.2158, 0.1711])])
    
    return transform(image)
    
def list_files_in_gcs_directory(client, prefix: str):
    """
    List all files in a given GCS directory prefix.

    Input:
        client (storage.Client): a GCS client instance for a Google Cloud project
        prefix (str): path of parent directory for files we want to retrieve 
    """
    blobs = client.list_blobs('aipi540-cv', prefix=prefix)
    return [blob.name for blob in blobs if not blob.name.endswith("/")] ## filter out subdirectories

def load_image_from_gcs(bucket, file_path: str):
    """
    Load an image from GCS and return as a PIL image.

    Input:
        bucket: Google Cloud bucket instance corresponding to DB to load from
        file_path: full path to image to load on Cloud Storage bucket
    """
    blob = bucket.blob(file_path)
    img_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(img_data))
    ## resize to 224,224 for standardized
    return image.resize((224,224), resample= Image.Resampling.NEAREST)

def compute_iou(mask1: np.ndarray, mask2: np.ndarray):
    """
    Compute the IoU between two binary numpy masks of the same size

    Input:
        mask1 (np.array): a binary mask (ground truth segmentation or GradCAM activation)
        mask2 (np.array): a binary mask (ground truth segmentation or GradCAM activation)
    """

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum() ## area of intersection
    union = np.logical_or(mask1, mask2).sum() ## area of union

    # edge case
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union

def overlay_segmentations(image_id: str, bucket, base_path: str, set: str):
    # init a blank numpy array to store all segmentations
    combined_segmentation = np.zeros((224,224), dtype=np.uint8)

    segmentation_types = ["haemorrhages", 'hard_exudates', 'microaneurysms', 'soft_exudates']
    segmentation_ext = ["HE", "EX", "MA", "SE"]
    
    for seg_type, seg_ext in zip(segmentation_types, segmentation_ext):
        segmentation_path = os.path.join(base_path, set, seg_type, f"{image_id}_{seg_ext}.tif") # path to the segmentation file
        blob = bucket.blob(segmentation_path)
        if blob.exists():
            seg_data = blob.download_as_bytes()
            try: 
                # load the .tif file
                segmentation = tifffile.imread(io.BytesIO(seg_data))
                segmentation_resized = cv2.resize(segmentation.astype('uint8'), (224, 224), interpolation=cv2.INTER_NEAREST)
                # add to masks of all lesions
                combined_segmentation += segmentation_resized
            ## handle errors in tifffile handling; skip file
            except ValueError:
                continue 
    return combined_segmentation

def evaluate_model(model_path: str, client, bucket): 
    """
    End-to-end pipeline for loading GradCAM model and computing IoU on IDRiD dataset

    Input:
        model_path: path to trained model
    """
    vgg = load_model(model_path)
    cam = convert_to_gradcam(vgg)
    iou_scores = []
    base_path = "segmentation/all_segmentation_groundtruths/"
    original_images_path = "segmentation/original_images/"
    dataset_sets = ["training_set", "testing_set"]
    for dataset_set in dataset_sets:
        image_files = list_files_in_gcs_directory(client, f"{original_images_path}{dataset_set}/")
        for file_path in image_files:
            if file_path.endswith(".jpg"):
                image_id = file_path.split("/")[-1].split(".")[0]
                original_image_array = load_image_from_gcs(bucket, file_path)
                combined_segmentation = overlay_segmentations(image_id, bucket, base_path, dataset_set)
                # Example visualization (optional)
                input_tensor = preprocess_image(original_image_array).unsqueeze(0)
                heatmap = cam(input_tensor = input_tensor, targets = None)
                percentile_threshold = np.percentile(heatmap, 90)  # Top 10%
                binary_cam = (heatmap >= percentile_threshold).astype(np.uint8)
                iou_score = compute_iou(combined_segmentation, np.squeeze(binary_cam))
                iou_scores.append(iou_score)
    return sum(iou_scores)/len(iou_scores)

    

if __name__ == "__main__":
    client = storage.Client(project='AIPI540')
    bucket = client.get_bucket('aipi540-cv')
    iou_score = evaluate_model('../vgg16_model.pth', client, bucket) ## change path
    print(f'Final IoU score for validation dataset: {iou_score}')
