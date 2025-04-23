from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import logging
from typing import List

from resnet import ResNet18, BasicBlock

app = FastAPI(title="ResNet18 Image Classifier API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class names (replace with your actual class names)
CLASS_NAMES = ["class0", "class1"]  # Modify with your actual class names

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model
try:
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("resnet18_custom.pth", map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.get("/")
async def root():
    return {"message": "ResNet18 Image Classification API", "status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction_idx = torch.argmax(probabilities).item()
            prediction_confidence = probabilities[prediction_idx].item()
        
        # Return prediction results
        return {
            "class_id": prediction_idx,
            "class_name": CLASS_NAMES[prediction_idx],
            "confidence": round(prediction_confidence * 100, 2)
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)