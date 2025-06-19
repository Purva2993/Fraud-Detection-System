"""FastAPI server for real-time fraud detection predictions."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.logger import logger
from src.config import config


# Pydantic models for request/response validation
class TransactionRequest(BaseModel):
    """Request model for a single transaction prediction."""
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    class Config:
        # Example values for API documentation
        schema_extra = {
            "example": {
                "Time": 12345.0,
                "Amount": 149.62,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053
            }
        }


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    is_fraud: bool
    fraud_probability: float
    confidence: str
    transaction_id: str


class BatchTransactionRequest(BaseModel):
    """Request model for batch predictions."""
    transactions: List[TransactionRequest]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    normal_count: int


class FraudDetectionAPI:
    """Main API class for fraud detection."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            model_path = Path("models/best_fraud_model.joblib")
            scaler_path = Path("models/scaler.joblib")
            
            if not model_path.exists() or not scaler_path.exists():
                logger.error("❌ Model files not found! Please train the model first.")
                logger.error("Run: python main.py train")
                return False
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_loaded = True
            
            logger.info("✅ Model and scaler loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_transaction(self, transaction: TransactionRequest) -> np.ndarray:
        """Preprocess a single transaction for prediction."""
        # Convert to dictionary and then to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Scale the features using the same scaler from training
        scaled_features = self.scaler.transform(df)
        
        return scaled_features
    
    def predict_single(self, transaction: TransactionRequest) -> PredictionResponse:
        """Make fraud prediction for a single transaction."""
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Preprocess the transaction
            scaled_features = self.preprocess_transaction(transaction)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            probability = self.model.predict_proba(scaled_features)[0, 1]
            
            # Determine confidence level
            if probability > 0.8:
                confidence = "HIGH"
            elif probability > 0.5:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            # Generate transaction ID
            import uuid
            transaction_id = str(uuid.uuid4())[:8]
            
            return PredictionResponse(
                is_fraud=bool(prediction),
                fraud_probability=float(probability),
                confidence=confidence,
                transaction_id=transaction_id
            )
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def predict_batch(self, transactions: List[TransactionRequest]) -> BatchPredictionResponse:
        """Make fraud predictions for multiple transactions."""
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        predictions = []
        fraud_count = 0
        
        for transaction in transactions:
            prediction = self.predict_single(transaction)
            predictions.append(prediction)
            if prediction.is_fraud:
                fraud_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(transactions),
            fraud_count=fraud_count,
            normal_count=len(transactions) - fraud_count
        )


# Initialize the API
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using machine learning",
    version="1.0.0"
)

fraud_api = FraudDetectionAPI()


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("🚀 Starting Fraud Detection API Server...")
    if not fraud_api.model_loaded:
        logger.warning("⚠️ Model not loaded! Please train the model first.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Fraud Detection API is running!",
        "model_loaded": fraud_api.model_loaded,
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if fraud_api.model_loaded else "model_not_loaded",
        "model_loaded": fraud_api.model_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict if a single transaction is fraudulent."""
    logger.info("🔍 Making fraud prediction for single transaction")
    return fraud_api.predict_single(transaction)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(request: BatchTransactionRequest):
    """Predict fraud for multiple transactions."""
    logger.info(f"🔍 Making fraud predictions for {len(request.transactions)} transactions")
    return fraud_api.predict_batch(request.transactions)


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if not fraud_api.model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(fraud_api.model).__name__,
        "model_loaded": True,
        "features_count": 30,
        "scaler_loaded": fraud_api.scaler is not None
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("🌟 Starting Fraud Detection API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")