import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate insurance dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data with proper error handling."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def validate_data(self) -> Tuple[bool, dict]:
        """Validate data structure and quality."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        validation_report = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": self.df.dtypes.to_dict(),
            "duplicates": self.df.duplicated().sum()
        }
        
        # Check required columns
        required_columns = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 
                           'Province', 'VehicleType', 'Gender', 'ZipCode']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            validation_report["missing_columns"] = missing_cols
            
        return len(missing_cols) == 0, validation_report