import pandas as pd
import numpy as np
from scipy import stats
import warnings
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """Analyze insurance portfolio statistics."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_loss_ratio(self) -> Dict:
        """Calculate overall and segmented loss ratios."""
        # Overall loss ratio
        total_premium = self.df['TotalPremium'].sum()
        total_claims = self.df['TotalClaims'].sum()
        overall_ratio = total_claims / total_premium if total_premium > 0 else np.nan
        
        # By Province
        province_ratios = self.df.groupby('Province').apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum() 
            if x['TotalPremium'].sum() > 0 else np.nan
        )
        
        # By Vehicle Type
        vehicle_ratios = self.df.groupby('VehicleType').apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum() 
            if x['TotalPremium'].sum() > 0 else np.nan
        )
        
        # By Gender
        gender_ratios = self.df.groupby('Gender').apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum() 
            if x['TotalPremium'].sum() > 0 else np.nan
        )
        
        return {
            "overall_loss_ratio": overall_ratio,
            "by_province": province_ratios.to_dict(),
            "by_vehicle_type": vehicle_ratios.to_dict(),
            "by_gender": gender_ratios.to_dict()
        }
    
    def detect_outliers_iqr(self, column: str) -> Tuple[pd.Series, float, float]:
        """Detect outliers using IQR method."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return outliers[column], lower_bound, upper_bound
    
    def analyze_distributions(self) -> Dict:
        """Fit statistical distributions to key variables."""
        distributions = {}
        
        for column in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']:
            data = self.df[column].dropna()
            
            # Fit normal distribution
            mu, sigma = stats.norm.fit(data)
            
            # Fit exponential distribution
            loc, scale = stats.expon.fit(data)
            
            # Kolmogorov-Smirnov test for normality
            ks_stat, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
            
            distributions[column] = {
                "normal_params": {"mu": mu, "sigma": sigma},
                "exponential_params": {"loc": loc, "scale": scale},
                "normality_test": {"ks_statistic": ks_stat, "p_value": p_value},
                "is_normal": p_value > 0.05
            }
            
        return distributions