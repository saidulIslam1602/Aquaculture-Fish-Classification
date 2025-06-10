"""
FastAPI-based Inference API for Fish Classification

Provides REST endpoints for real-time fish species classification
with support for single image and batch inference.
"""

import os
import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Any, Optional
import logging
import uvicorn
from pydantic import BaseModel
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.fish_classifier import load_pretrained_model
from utils.config import setup_config
from utils.device import setup_device

logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    species: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_processing_time: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    version: str


class FishClassificationAPI:
    """Main API class for fish classification."""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize the API.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to the configuration file
        """
        self.config = setup_config(config_path)
        self.device = setup_device(self.config.hardware.device)
        self.model = None
        self.class_names = self.config.species.classes
        self.confidence_threshold = self.config.deployment.confidence_threshold
        
        # Load model
        self._load_model(model_path)
        
        # Setup image preprocessing
        self._setup_transforms()
        
        logger.info("Fish Classification API initialized")
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            self.model = load_pretrained_model(model_path, self.config, str(self.device))
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = A.Compose([
            A.Resize(*self.config.data.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy
        image_np = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict_single(self, image: Image.Image) -> PredictionResponse:
        """
        Predict species for a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Create response
            species = self.class_names[predicted_class]
            prob_dict = {
                class_name: float(probabilities[0, i].item())
                for i, class_name in enumerate(self.class_names)
            }
            
            processing_time = time.time() - start_time
            
            return PredictionResponse(
                species=species,
                confidence=confidence,
                probabilities=prob_dict,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, images: List[Image.Image]) -> BatchPredictionResponse:
        """
        Predict species for a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Batch prediction response
        """
        start_time = time.time()
        predictions = []
        
        for image in images:
            pred = self.predict_single(image)
            predictions.append(pred)
        
        total_processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time=total_processing_time
        )


# Initialize FastAPI app
app = FastAPI(
    title="Aquaculture Fish Classification API",
    description="AI-powered fish species classification for aquaculture industry",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API instance
api_instance: Optional[FishClassificationAPI] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    global api_instance
    
    # Get model and config paths from environment or use defaults
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise RuntimeError(f"Model file not found: {model_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise RuntimeError(f"Config file not found: {config_path}")
    
    api_instance = FishClassificationAPI(model_path, config_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if api_instance and api_instance.model else "unhealthy",
        model_loaded=api_instance is not None and api_instance.model is not None,
        device=str(api_instance.device) if api_instance else "unknown",
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict fish species from uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction response
    """
    if not api_instance:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get prediction
        prediction = api_instance.predict_single(image)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Predict fish species from multiple uploaded images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        Batch prediction response
    """
    if not api_instance:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    try:
        images = []
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="All files must be images")
            
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
        
        # Get predictions
        predictions = api_instance.predict_batch(images)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/species")
async def get_species_list():
    """Get list of supported fish species."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    return {
        "species": api_instance.class_names,
        "count": len(api_instance.class_names)
    }


@app.get("/model_info")
async def get_model_info():
    """Get model information."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    return {
        "architecture": api_instance.config.model.architecture,
        "num_classes": api_instance.config.model.num_classes,
        "image_size": api_instance.config.data.image_size,
        "confidence_threshold": api_instance.confidence_threshold,
        "device": str(api_instance.device)
    }


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server.
    
    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload for development
    """
    uvicorn.run("src.api.inference:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the API
    run_api(reload=True) 