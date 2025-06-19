"""
Main entry point for the Fraud Detection System.

This script provides command-line interface to run different components
of the fraud detection system.
"""

import click
from src.config import config
from src.logger import logger


@click.group()
def cli():
    """Fraud Detection System CLI."""
    logger.info("Starting Fraud Detection System")


@cli.command()
def setup():
    """Set up the project environment and download data."""
    logger.info("Setting up project environment...")
    
    # Create directories (already done in config, but let's be explicit)
    config._create_directories()
    logger.info("Created project directories")
    
    # Show what we created
    logger.info("Project setup completed successfully!")
    logger.info("You can now proceed with data download and model training")


@cli.command()
def data():
    """Load and explore the fraud detection dataset."""
    logger.info("Loading and exploring fraud detection dataset...")
    
    try:
        from src.data_loader import load_fraud_data, explore_data
        
        # Load the dataset
        df = load_fraud_data()
        
        # Explore the dataset
        explore_data(df)
        
        logger.info("Dataset ready for model training!")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")


@cli.command()
def train():
    """Train the fraud detection model."""
    logger.info("Starting machine learning model training...")
    
    try:
        from src.model_trainer import train_fraud_model
        
        # Train the model
        trainer = train_fraud_model()
        
        logger.info("🎉 Model training completed successfully!")
        logger.info("You can now use 'python main.py serve' to start the API")
        
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")


@cli.command()
def serve():
    """Start the API server."""
    logger.info("🚀 Starting Fraud Detection API server...")
    
    try:
        import uvicorn
        from src.api_server import app
        
        logger.info("📡 API server will be available at:")
        logger.info("🌐 Main API: http://localhost:8000")
        logger.info("📚 API Documentation: http://localhost:8000/docs")
        logger.info("🔍 Health Check: http://localhost:8000/health")
        logger.info("Press Ctrl+C to stop the server")
        
        # Start the server
        uvicorn.run(
            app, 
            host=config.api.host, 
            port=config.api.port, 
            log_level=config.api.log_level.lower()
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 API server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting API server: {e}")


@cli.command()
def stream():
    """Start the streaming data processor."""
    logger.info("Starting streaming processor...")
    # TODO: We'll implement this in next steps
    logger.info("Streaming processor not implemented yet - coming in next steps!")


@cli.command()
def dashboard():
    """Launch the monitoring dashboard."""
    logger.info("Launching dashboard...")
    # TODO: We'll implement this in next steps
    logger.info("Dashboard not implemented yet - coming in next steps!")


if __name__ == "__main__":
    cli()