import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InsuranceVisualizer:
    """Create insightful visualizations for insurance data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def create_loss_ratio_heatmap(self) -> go.Figure:
        """Create heatmap of loss ratios by Province and VehicleType."""
        # Calculate loss ratios
        pivot_data = self.df.groupby(['Province', 'VehicleType']).apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum() 
            if x['TotalPremium'].sum() > 0 else np.nan
        ).reset_index(name='LossRatio')
        
        # Pivot for heatmap
        heatmap_data = pivot_data.pivot(index='Province', 
                                       columns='VehicleType', 
                                       values='LossRatio')
        
        fig = px.imshow(
            heatmap_data,
            title="Loss Ratio Heatmap by Province and Vehicle Type",
            labels=dict(x="Vehicle Type", y="Province", color="Loss Ratio"),
            color_continuous_scale="RdYlGn_r",  # Red for high loss ratio
            aspect="auto"
        )
        
        fig.update_layout(height=600)
        return fig
    
    def create_premium_claims_scatter(self) -> go.Figure:
        """Interactive scatter plot of Premium vs Claims with clustering."""
        fig = px.scatter(
            self.df,
            x="TotalPremium",
            y="TotalClaims",
            color="Province",
            size="CustomValueEstimate",
            hover_data=["VehicleType", "Gender", "ZipCode"],
            title="Total Premium vs Total Claims by Province",
            trendline="ols",
            trendline_scope="overall"
        )
        
        fig.update_layout(
            xaxis_title="Total Premium",
            yaxis_title="Total Claims",
            height=600
        )
        
        return fig
    
    def create_temporal_trend_plot(self) -> go.Figure:
        """Show temporal trends in claims and premiums."""
        if 'PolicyStartDate' not in self.df.columns:
            # Create synthetic date for demonstration
            dates = pd.date_range(start='2022-01-01', periods=len(self.df), freq='D')
            self.df['PolicyStartDate'] = dates[:len(self.df)]
        
        # Aggregate by month
        self.df['Month'] = pd.to_datetime(self.df['PolicyStartDate']).dt.to_period('M')
        monthly_data = self.df.groupby('Month').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'CustomValueEstimate': 'mean'
        }).reset_index()
        monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
        monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Monthly Premiums", "Monthly Claims", 
                          "Average Vehicle Value", "Loss Ratio Trend"),
            vertical_spacing=0.15
        )
        
        # Premium trend
        fig.add_trace(
            go.Scatter(x=monthly_data['Month'], y=monthly_data['TotalPremium'],
                      mode='lines+markers', name='Premium'),
            row=1, col=1
        )
        
        # Claims trend
        fig.add_trace(
            go.Scatter(x=monthly_data['Month'], y=monthly_data['TotalClaims'],
                      mode='lines+markers', name='Claims'),
            row=1, col=2
        )
        
        # Vehicle value trend
        fig.add_trace(
            go.Scatter(x=monthly_data['Month'], y=monthly_data['CustomValueEstimate'],
                      mode='lines+markers', name='Avg Vehicle Value'),
            row=2, col=1
        )
        
        # Loss ratio trend
        fig.add_trace(
            go.Scatter(x=monthly_data['Month'], y=monthly_data['LossRatio'],
                      mode='lines+markers', name='Loss Ratio'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Temporal Trends in Insurance Portfolio")
        return fig
    
    def create_vehicle_make_analysis(self) -> go.Figure:
        """Analyze top vehicle makes by claim frequency and severity."""
        vehicle_stats = self.df.groupby('VehicleMake').agg({
            'TotalClaims': ['count', 'sum', 'mean'],
            'TotalPremium': 'sum'
        }).round(2)
        
        vehicle_stats.columns = ['Count', 'TotalClaims', 'AvgClaim', 'TotalPremium']
        vehicle_stats['ClaimFrequency'] = vehicle_stats['Count'] / vehicle_stats['Count'].sum()
        vehicle_stats['ClaimSeverity'] = vehicle_stats['TotalClaims'] / vehicle_stats['Count']
        vehicle_stats['LossRatio'] = vehicle_stats['TotalClaims'] / vehicle_stats['TotalPremium']
        
        # Get top 10 by total claims
        top_10 = vehicle_stats.nlargest(10, 'TotalClaims')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Total Claims by Make", "Claim Frequency",
                          "Average Claim Amount", "Loss Ratio"),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Bar chart: Total Claims
        fig.add_trace(
            go.Bar(x=top_10.index, y=top_10['TotalClaims'], name='Total Claims'),
            row=1, col=1
        )
        
        # Pie chart: Claim Frequency
        fig.add_trace(
            go.Pie(labels=top_10.index, values=top_10['ClaimFrequency'],
                   name='Claim Frequency'),
            row=1, col=2
        )
        
        # Bar chart: Average Claim
        fig.add_trace(
            go.Bar(x=top_10.index, y=top_10['AvgClaim'], name='Avg Claim'),
            row=2, col=1
        )
        
        # Bar chart: Loss Ratio
        fig.add_trace(
            go.Bar(x=top_10.index, y=top_10['LossRatio'], name='Loss Ratio'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Vehicle Make Analysis")
        return fig
    
    def create_geographic_distribution(self) -> go.Figure:
        """Create geographic distribution of premiums and claims."""
        # Group by ZipCode (first 3 digits for aggregation)
        self.df['ZipPrefix'] = self.df['ZipCode'].astype(str).str[:3]
        geo_data = self.df.groupby('ZipPrefix').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'CustomValueEstimate': 'mean',
            'Province': 'first'
        }).reset_index()
        
        geo_data['LossRatio'] = geo_data['TotalClaims'] / geo_data['TotalPremium']
        
        fig = px.scatter_geo(
            geo_data,
            lat=np.random.uniform(43, 60, len(geo_data)),  # Canada latitudes
            lon=np.random.uniform(-140, -50, len(geo_data)),  # Canada longitudes
            size="TotalPremium",
            color="LossRatio",
            hover_name="ZipPrefix",
            hover_data=["TotalPremium", "TotalClaims", "CustomValueEstimate"],
            title="Geographic Distribution of Insurance Portfolio",
            color_continuous_scale="RdYlGn_r",
            size_max=50,
            scope="north america"
        )
        
        fig.update_layout(height=600)
        return fig