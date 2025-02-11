from google.cloud import storage
import os
import pandas as pd
from io import StringIO, BytesIO
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def fetch_labels(bucket, path, N = 2000):
    """
    fetch image labels as dataframe from GCS, subsampling N rows with str 
    """
    blob = bucket.blob(path)
    csv_data = blob.download_as_text()
    df = pd.read_csv(StringIO(csv_data))
    df['level'] = df['level'].replace({2: 1, 3: 2, 4: 2})
    df_sampled, _ = train_test_split(
        df, test_size=len(df) - N, stratify=df["level"], random_state=42
    )
    df_sampled = df_sampled.reset_index().drop(columns='index')
    return df_sampled


def iterative_thresholding(image: np.ndarray):
    """
    perform iterative thresholding on a grayscale image.
    
    Input:
    - image (np.array): a 2D grayscale image with pixel values in [0, 255]
    
    Returns:
    - binary_image (np.ndarray): binary image with pixel values 0 or 255
    """
    # Normalize the intensity values to [0, 1]
    image_normalized = image / 255.0
    
    # Initial threshold: half of the maximum dynamic range (0.5 for normalized image)
    threshold = 0.5
    
    while True:
        # Separate foreground and background based on current threshold
        foreground = image_normalized[image_normalized >= threshold]
        background = image_normalized[image_normalized < threshold]
        
        # Compute the sample means for foreground and background
        mean_foreground = np.mean(foreground) if foreground.size > 0 else 0
        mean_background = np.mean(background) if background.size > 0 else 0
        
        # Compute the new threshold
        new_threshold = (mean_foreground + mean_background) / 2
        
        # Break if threshold convergence is achieved
        if np.abs(new_threshold - threshold) < 1e-5:
            break
        
        threshold = new_threshold

    # Apply the final threshold to generate the binary image
    binary_image = (image_normalized >= threshold).astype(np.uint8) * 255
    
    return binary_image


def remove_small_clusters(binary_image, min_size=20):
    """
    remove small clusters (noise) from the image.
    
    Input:
    - binary_image (np.array): binary image with pixel values 0 or 255    
    """

    # Step 1: Label connected components
    num_labels, labels = cv2.connectedComponents(binary_image)

    # Step 2: Create a new image to store the filtered components
    filtered_image = np.zeros_like(binary_image)

    # Step 3: Iterate through the components and check the size
    for label in range(1, num_labels):  # Start from 1 to ignore the background (label 0)
        # Create a mask for the current component
        component_mask = (labels == label)

        # Count the number of white pixels in the component
        component_size = np.sum(component_mask)

        # Step 4: If the component size is larger than min_size, keep it in the filtered image
        if component_size > min_size:
            filtered_image[component_mask] = 255  # Set the corresponding pixels to white

    return filtered_image


def _apply_preprocessing(bucket, row, N = 8):
    """
    Image Processing + GLCM Feature Extraction pipeline

    Input:
    – bucket: GCS bucket
    – row: row from trainLabels.csv to extract features from
    – bin_size (int): # of groupings of pixels
    """
    img_filename = str(row["image"])
    ## fetch image
    blob_path = f"classification/resized_train/{img_filename}.jpeg"
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    pil_image = Image.open(BytesIO(data)).convert("RGB")
    np_image = np.array(pil_image)
    ## resize to 224 x 224
    img_resized = cv2.resize(np_image, (224,224), interpolation=cv2.INTER_AREA)
    ## convert to grayscale
    gray_image = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    ## contrast enhancement
    hist_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_image)
    ## blurring
    blur_img = cv2.blur(hist_eq, (7, 7))
    ## subtraction 
    subtracted = cv2.subtract(hist_eq, blur_img)
    subtracted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(subtracted)
    ## binary thresholding
    binary_image = iterative_thresholding(subtracted)
    ## noise removal
    noise_removed = remove_small_clusters(binary_image)
    ## complementary image
    complemented_binary = cv2.bitwise_not(noise_removed)
    ## overlaid (final) image
    overlay_image = np.where(complemented_binary == 255, 0, gray_image)
    ## flatten features into 1D for feature matrix
    features.extend(overlay_image.flatten())
    ## quantize pixel intensies into N groups
    bin_size = 256 // N
    quantized = (hist_eq // bin_size).astype(np.uint8)
    ## compute GLCM, using 3 distances and averaging across 4 angles
    P = graycomatrix(quantized, [1,10,50], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=N, normed=True)
    ## extract texture features from GLCM
    features = []
    features.extend(graycoprops(P, 'contrast').mean(axis=1))
    features.extend(graycoprops(P, 'dissimilarity').mean(axis=1))
    features.extend(graycoprops(P, 'homogeneity').mean(axis=1))
    features.extend(graycoprops(P, 'energy').mean(axis=1))
    features.extend(graycoprops(P, 'correlation').mean(axis=1))
    features.extend(graycoprops(P, 'mean').mean(axis=1))
    features.extend(graycoprops(P, 'variance').mean(axis=1))
    features.extend(graycoprops(P, 'entropy').mean(axis=1))
    return features

def train_random_forest(X_train, y_train):
    """
    
    """

    # # of trees in random forest
    n_estimators = [400,431,450]
    # max #s of levels in tree
    max_depth = [90,100,100]
    # min. number of samples required to split a node
    min_samples_split = [2, 3, 4]

    param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

    ## hyperparameter tuning with GridSarchCV
    rf = RandomForestClassifier(max_features='sqrt', min_samples_leaf=1, random_state=42)
    rf_tuned = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', verbose=2, n_jobs = -1)
    rf_tuned.fit(X_train, y_train)
    return rf_tuned.best_estimator_

def evaluate_rf_model(model, X_test, y_test):
    y_preds = model.predict(X_test)
    f1_scores = f1_score(y_test, y_preds, average='weighted')
    return f1_scores
    

if __name__ == "__main__":
    client = storage.Client(project='AIPI540')
    bucket = client.get_bucket('aipi540-cv')
    df_sampled = fetch_labels(bucket, 'classification/trainLabels.csv')
    feature_matrix = np.array(df_sampled.apply(_apply_preprocessing, axis=1).tolist())
    feature_df = pd.DataFrame(feature_matrix)
    feature_df['label'] = df_sampled['level']
    X_train, X_test, y_train, y_test = train_test_split(feature_df.iloc[:,:-1], feature_df.iloc[:,-1], test_size=0.2, stratify=feature_df.iloc[:,-1], random_state=0)
    rf_model = train_random_forest(X_train, y_train)
    f1 = evaluate_rf_model(rf_model, X_test, y_test)
    print(f'Final F1 score for validation dataset: {f1}')

