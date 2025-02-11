# app.py
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from models.model import NaiveModel, ClassicalModel, DeepLearningModel
import joblib

# Bucket name
BUCKET_NAME = "aipi540-cv"
VERTEX_AI_ENDPOINT = ""

# class type
class_names = ["Normal", "Mild Diabetic Retinopathy", "Severe Diabetic Retinopathy"]

# need to change the following code
class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    # Preprocess the image
    def preprocess_image(self, image):
        """Preprocess the image for model input."""
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to(self.device)

def load_model(model_type):
    handler = ModelHandler()
    device = handler.device
    
    # Load model
    if model_type == "Naive approach":
        model = NaiveModel()
        state_dict = torch.load("models/naive_model.pth", map_location=device)
        model.load_state_dict(state_dict)
    elif model_type == "Machine Learning Model":
        model = ClassicalModel()
        model.load_state_dict(torch.load("models/classical_model.pth", map_location=device))
    else:  # Deep Learning Model
        model = DeepLearningModel()
        model.load_state_dict(torch.load("models/vgg16_model.pth", map_location=device))
    
    model = model.to(device)
    if hasattr(model, 'eval'):
        model.eval()
    
    return model, handler.preprocess_image

# Prediction function
def predict(model, image_tensor, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Get class probabilities
        if model_type == "Naive approach":
            probabilities = outputs
        else:
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()
        class_probabilities = probabilities[0].cpu().numpy()
    
    
    return predicted_class, class_probabilities

# the streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Eye Disease Prediction",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("👁️ Diabetic Retinopathy Detection System")
    st.write("Upload a fundus image to detect diabetic retinopathy severity")
    
    # 
    with st.sidebar:
        st.header("Model Selection")
        model_type = st.selectbox(
            "Select Model",
            ["Naive approach", "Machine Learning Model", "Deep Learning Model"]
        )
        
        st.header("About")
        st.markdown("""
        This system aims to detect diabetic retinopathy(DR) from fundus images.
        ### Available Models:
        - **Naive approach**: Based on data distribution
        - **Machine Learning Model**: Traditional ML approach
        - **Deep Learning**: VGG16 based model
        
        ### Classes:
        - Normal (No DR)
        - Mild DR
        - Severe DR
        """)
    
    st.header("Image Upload")
    uploaded_file = st.file_uploader(
        "Choose a fundus image", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Image"):
            try:
                model, preprocess = load_model(model_type)
                processed_image = preprocess(image)
                
                with st.spinner("Analyzing image..."):
                    predicted_class, probabilities = predict(model, processed_image, model_type)
                    
                st.success("Analysis Complete!")
                
                st.header("Prediction Results")
                st.write(f"**Predicted Condition:** {class_names[predicted_class]}")
                st.write("**Class Probabilities:**")
                st.json(probabilities)
                
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()