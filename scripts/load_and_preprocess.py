#!/usr/bin/env python3
"""
Data Loading and Preprocessing Script for DVC Pipeline
"""
import sys
import os
import logging
import json
import pandas as pd

# Make sure src module is discoverable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import InsuranceDataProcessor

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROCESSED_DATA_PATH = "data/processed/cleaned_data.parquet"
METADATA_PATH = "data/processed/data_metadata.json"

def ensure_directories():
    """Ensure the processed data directory exists"""
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

def main():
    """Main data loading and preprocessing function"""
    logger.info("Starting data loading and preprocessing...")

    ensure_directories()

    try:
        # Initialize data processor
        processor = InsuranceDataProcessor()

        # Load data
        df = processor.load_data()
        if df.empty:
            logger.warning("Loaded DataFrame is empty")
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Preprocess data
        df_processed = processor.preprocess_data(df)
        if df_processed.empty:
            logger.warning("Processed DataFrame is empty")
        logger.info(f"Data preprocessed. New shape: {df_processed.shape}")

        # Save processed data
        df_processed.to_parquet(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Processed data saved successfully at {PROCESSED_DATA_PATH}")

        # Generate metadata
        metadata = {
            'processing_date': str(pd.Timestamp.now()),
            'original_shape': df.shape if df is not None else (0, 0),
            'processed_shape': df_processed.shape if df_processed is not None else (0, 0),
            'missing_values_original': int(df.isnull().sum().sum()) if df is not None else 0,
            'missing_values_processed': int(df_processed.isnull().sum().sum()) if df_processed is not None else 0,
            'columns_processed': list(df_processed.columns) if df_processed is not None else []
        }

        # Save metadata
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved successfully at {METADATA_PATH}")

        return True

    except Exception as e:
        logger.error(f"Error during data loading and preprocessing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)