"""Machine Learning model training for fraud detection."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.logger import logger
from src.config import config


class FraudModelTrainer:
    """Handles training and evaluation of fraud detection models."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def prepare_data(self, df):
        """
        Prepare data for machine learning.
        
        Args:
            df: DataFrame with fraud detection data
            
        Returns:
            X_train, X_test, y_train, y_test: Prepared datasets
        """
        logger.info("🔧 Preparing data for machine learning...")
        
        # Separate features and target
        # Features: everything except 'Class'
        X = df.drop('Class', axis=1)
        # Target: 'Class' column (0=normal, 1=fraud)
        y = df['Class']
        
        logger.info(f"📊 Features shape: {X.shape}")
        logger.info(f"🎯 Target distribution: Normal={sum(y==0)}, Fraud={sum(y==1)}")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.model.test_size,  # 20% for testing
            random_state=config.model.random_state,  # Reproducible results
            stratify=y  # Keep same fraud ratio in train/test
        )
        
        logger.info(f"📚 Training set: {X_train.shape[0]} samples")
        logger.info(f"🧪 Testing set: {X_test.shape[0]} samples")
        
        # Scale features (normalize them)
        # This helps algorithms work better
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("✅ Data preparation completed!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Handle the imbalanced dataset using SMOTE.
        
        Since fraud is rare (2%), we need to balance the classes
        for better model training.
        """
        logger.info("⚖️ Handling class imbalance with SMOTE...")
        
        # Count original distribution
        original_normal = sum(y_train == 0)
        original_fraud = sum(y_train == 1)
        logger.info(f"📊 Original: Normal={original_normal}, Fraud={original_fraud}")
        
        # Apply SMOTE (Synthetic Minority Oversampling Technique)
        smote = SMOTE(random_state=config.model.random_state)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        # Count new distribution
        new_normal = sum(y_balanced == 0)
        new_fraud = sum(y_balanced == 1)
        logger.info(f"📊 Balanced: Normal={new_normal}, Fraud={new_fraud}")
        
        return X_balanced, y_balanced
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare their performance."""
        logger.info("🚀 Training fraud detection models...")
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=config.model.random_state,
                n_jobs=-1  # Use all CPU cores
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=config.model.random_state,
                eval_metric='logloss'  # Suppress warnings
            )
        }
        
        results = {}
        
        # Train each model
        for name, model in models_to_train.items():
            logger.info(f"🎯 Training {name}...")
            
            # Train the model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
            
            # Calculate performance metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"✅ {name} trained! AUC Score: {auc_score:.4f}")
            
            # Print detailed classification report
            logger.info(f"📊 {name} Classification Report:")
            report = classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])
            print(report)  # Print to console for better formatting
        
        # Find best model
        self.best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
        self.best_model = results[self.best_model_name]['model']
        self.best_score = results[self.best_model_name]['auc_score']
        
        logger.info(f"🏆 Best model: {self.best_model_name} (AUC: {self.best_score:.4f})")
        
        return results
    
    def save_model(self):
        """Save the best trained model to disk."""
        if self.best_model is None:
            logger.error("❌ No model trained yet!")
            return
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model and scaler
        model_path = models_dir / "best_fraud_model.joblib"
        scaler_path = models_dir / "scaler.joblib"
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"💾 Scaler saved: {scaler_path}")
        logger.info(f"🎯 Best model: {self.best_model_name}")
        
        return model_path, scaler_path


def train_fraud_model():
    """Main function to train fraud detection model."""
    logger.info("🤖 === FRAUD DETECTION MODEL TRAINING ===")
    
    try:
        # Load data
        from src.data_loader import load_fraud_data
        df = load_fraud_data()
        
        # Initialize trainer
        trainer = FraudModelTrainer()
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        # Train models
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Save best model
        trainer.save_model()
        
        logger.info("🎉 Model training completed successfully!")
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        raise e