"""Data loading and downloading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.logger import logger
from src.config import config


def download_sample_data():
    """
    Create a sample fraud detection dataset that mimics real credit card data.
    This helps us test our system before using real data.
    """
    logger.info("Creating sample fraud detection dataset...")
    
    # Set random seed so we get the same data every time (reproducible)
    np.random.seed(42)
    
    # Dataset parameters
    n_samples = 1000      # Total number of transactions
    n_fraud = 20          # Number of fraudulent transactions (2% fraud rate)
    
    logger.info(f"Generating {n_samples} transactions with {n_fraud} fraudulent ones")
    
    # Create empty dictionary to store our features
    data = {}
    
    # === 1. TIME FEATURE ===
    # Represents when the transaction occurred (in seconds)
    # Real dataset: seconds elapsed between transactions and first transaction
    data['Time'] = np.random.randint(0, 172800, n_samples)  # 2 days = 172800 seconds
    
    # === 2. AMOUNT FEATURE ===
    # Transaction amount in dollars
    # Using log-normal distribution (most transactions small, few large ones)
    data['Amount'] = np.random.lognormal(mean=3, sigma=1, size=n_samples)
    
    # === 3. ANONYMIZED FEATURES V1-V28 ===
    # In real dataset, these are PCA-transformed features (privacy protection)
    # We simulate them as normally distributed random numbers
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(loc=0, scale=1, size=n_samples)
    
    # === 4. CLASS FEATURE (TARGET) ===
    # 0 = Normal transaction, 1 = Fraudulent transaction
    data['Class'] = np.zeros(n_samples)  # Start with all normal
    
    # Randomly select which transactions will be fraudulent
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    data['Class'][fraud_indices] = 1  # Mark them as fraud
    
    # === 5. MAKE FRAUD LOOK DIFFERENT ===
    # Real fraud has patterns - let's simulate them
    for idx in fraud_indices:
        # Fraudulent transactions often have higher amounts
        data['Amount'][idx] *= 3
        
        # Modify some V features to create fraud patterns
        for i in range(1, 10):  # Change first 9 V features
            data['V{}'.format(i)][idx] += np.random.normal(2, 1)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV file
    output_path = Path("data/raw/sample_fraud_data.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Dataset created: {n_samples} transactions ({n_fraud} fraudulent)")
    logger.info(f"✅ Fraud rate: {(n_fraud/n_samples)*100:.1f}%")
    logger.info(f"✅ Saved to: {output_path}")
    
    return df


def load_fraud_data():
    """Load the fraud detection dataset from file or create it."""
    data_path = Path("data/raw/sample_fraud_data.csv")
    
    if not data_path.exists():
        logger.info("📁 Dataset not found. Creating new sample dataset...")
        return download_sample_data()
    
    logger.info(f"📁 Loading existing dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    fraud_count = df['Class'].sum()
    total_count = len(df)
    fraud_rate = (fraud_count / total_count) * 100
    
    logger.info(f"✅ Dataset loaded: {total_count} transactions")
    logger.info(f"✅ Fraud transactions: {fraud_count}")
    logger.info(f"✅ Fraud rate: {fraud_rate:.2f}%")
    
    return df


def explore_data(df):
    """Analyze our dataset to understand what we're working with."""
    logger.info("🔍 === DATASET EXPLORATION ===")
    
    # Basic info
    logger.info(f"📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"📋 Features: {list(df.columns)}")
    logger.info(f"❓ Missing values: {df.isnull().sum().sum()}")
    
    # Class distribution
    normal_transactions = (df['Class'] == 0).sum()
    fraud_transactions = df['Class'].sum()
    fraud_percentage = (fraud_transactions / len(df)) * 100
    
    logger.info(f"✅ Normal transactions: {normal_transactions}")
    logger.info(f"🚨 Fraud transactions: {fraud_transactions}")
    logger.info(f"📈 Fraud percentage: {fraud_percentage:.2f}%")
    
    # Amount statistics
    logger.info("💰 === TRANSACTION AMOUNTS ===")
    logger.info(f"💵 Average amount: ${df['Amount'].mean():.2f}")
    logger.info(f"💵 Median amount: ${df['Amount'].median():.2f}")
    logger.info(f"💵 Max amount: ${df['Amount'].max():.2f}")
    logger.info(f"💵 Min amount: ${df['Amount'].min():.2f}")
    
    # Compare normal vs fraud amounts
    normal_amounts = df[df['Class'] == 0]['Amount']
    fraud_amounts = df[df['Class'] == 1]['Amount']
    
    logger.info(f"💰 Average normal transaction: ${normal_amounts.mean():.2f}")
    logger.info(f"🚨 Average fraud transaction: ${fraud_amounts.mean():.2f}")
    
    return df