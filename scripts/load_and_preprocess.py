#!/usr/bin/env python3
"""
Data Loading and Preprocessing Script for DVC Pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import InsuranceDataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main data loading and preprocessing function"""
    logger.info("Starting data loading and preprocessing...")
    
    try:
        # Initialize data processor
        processor = InsuranceDataProcessor()
        
        # Load data
        df = processor.load_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Preprocess data
        df_processed = processor.preprocess_data()
        logger.info(f"Data preprocessed. New shape: {df_processed.shape}")
        
        # Save processed data
        processor.save_processed_data()
        logger.info("Processed data saved successfully")
        
        # Generate and save metadata
        metadata = {
            'processing_date': str(pd.Timestamp.now()),
            'original_shape': processor.metadata.get('original_shape', df.shape),
            'processed_shape': df_processed.shape,
            'missing_values_original': df.isnull().sum().sum(),
            'missing_values_processed': df_processed.isnull().sum().sum(),
            'columns_processed': list(df_processed.columns)
        }
        
        import json
        metadata_path = "data/processed/data_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data loading and preprocessing: {e}")
        return False


if __name__ == "__main__":
    import pandas as pd
    success = main()
    sys.exit(0 if success else 1)