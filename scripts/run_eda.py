import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from src.utils.data_loader import InsuranceDataProcessor
from src.analysis.eda_analyzer import EDAAnalyzer
from src.visualization.eda_plots import EDAVisualizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_comprehensive_eda():
    """Run comprehensive EDA analysis"""
    logger.info("Starting comprehensive EDA analysis...")
    
    try:
        # Load processed data
        processor = InsuranceDataProcessor()
        processor.load_data()
        df = processor.preprocess_data()
        
        logger.info(f"Data loaded for EDA. Shape: {df.shape}")
        
        # Initialize analyzer and visualizer
        analyzer = EDAAnalyzer(df)
        visualizer = EDAVisualizer(df, save_dir="reports/figures")
        
        # Run comprehensive analysis
        logger.info("Running descriptive statistics...")
        desc_stats = analyzer.compute_descriptive_statistics()
        
        logger.info("Analyzing missing values...")
        missing_df = analyzer.analyze_missing_values()
        
        logger.info("Analyzing distributions...")
        distribution_results = analyzer.analyze_distributions()
        
        logger.info("Analyzing correlations...")
        correlation_matrix = analyzer.analyze_correlations()
        
        logger.info("Detecting outliers...")
        outlier_results = analyzer.detect_outliers()
        
        logger.info("Analyzing temporal trends...")
        temporal_results = analyzer.analyze_temporal_trends()
        
        logger.info("Analyzing by dimensions...")
        dimensions = ['Province', 'VehicleType', 'Gender', 'CoverType', 'Make']
        dimension_results = analyzer.analyze_by_dimensions(dimensions)
        
        logger.info("Generating visualizations...")
        # Create standard visualizations
        visualizer.plot_data_quality_summary(missing_df)
        visualizer.plot_numeric_distributions(analyzer.numeric_cols[:10])
        visualizer.plot_categorical_distributions(analyzer.categorical_cols[:8])
        visualizer.plot_correlation_matrix(correlation_matrix)
        visualizer.plot_loss_ratio_analysis()
        
        if 'monthly_data' in temporal_results:
            visualizer.plot_temporal_trends(temporal_results['monthly_data'])
        
        visualizer.plot_outlier_analysis(outlier_results)
        
        # Create creative visualizations
        logger.info("Creating creative visualizations...")
        visualizer.create_creative_visualization_1()
        visualizer.create_creative_visualization_2()
        visualizer.create_creative_visualization_3()
        
        # Generate comprehensive summary
        logger.info("Generating comprehensive summary report...")
        summary_report = analyzer.generate_summary_report()
        
        # Save summary report
        summary_path = "reports/docs/eda_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        logger.info(f"EDA summary saved to: {summary_path}")
        
        # Generate metrics for DVC
        metrics = {
            'overall_loss_ratio': float(df['LossRatio'].mean()) if 'LossRatio' in df.columns else None,
            'total_premium': float(df['TotalPremium'].sum()) if 'TotalPremium' in df.columns else None,
            'total_claims': float(df['TotalClaims'].sum()) if 'TotalClaims' in df.columns else None,
            'data_quality_score': 100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            'outlier_percentage': outlier_results.get('summary', {}).get('percentage_outlier_records', 0),
            'analysis_timestamp': str(pd.Timestamp.now())
        }
        
        metrics_path = "reports/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Generate HTML report
        generate_html_report(summary_report, metrics)
        
        logger.info("✅ EDA analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in EDA analysis: {e}")
        return False


def generate_html_report(summary_report, metrics):
    """Generate HTML report from EDA results"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Insurance Risk Analysis - EDA Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .insight {{ background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .warning {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Insurance Risk Analysis - Exploratory Data Analysis Report</h1>
    <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric">
        <p><strong>Overall Loss Ratio:</strong> {metrics['overall_loss_ratio']:.3f}</p>
        <p><strong>Total Premium:</strong> ${metrics['total_premium']:,.0f}</p>
        <p><strong>Total Claims:</strong> ${metrics['total_claims']:,.0f}</p>
        <p><strong>Data Quality Score:</strong> {metrics['data_quality_score']:.1f}%</p>
    </div>
    
    <h2>Key Insights</h2>
"""
    
    # Add insights from summary report
    if 'risk_insights' in summary_report:
        insights = summary_report['risk_insights']
        
        if 'top_risky_provinces' in insights:
            html_content += "<div class='insight'><h3>Top Risky Provinces</h3><ul>"
            for province in insights['top_risky_provinces'][:3]:
                html_content += f"<li>{province['Province']}: Loss Ratio = {province['mean']:.3f}</li>"
            html_content += "</ul></div>"
        
        if 'top_risky_vehicles' in insights:
            html_content += "<div class='insight'><h3>Top Risky Vehicle Types</h3><ul>"
            for vehicle in insights['top_risky_vehicles'][:3]:
                html_content += f"<li>{vehicle['VehicleType']}: Loss Ratio = {vehicle['mean']:.3f}</li>"
            html_content += "</ul></div>"
    
    # Add data quality warnings
    if 'data_quality' in summary_report:
        dq = summary_report['data_quality']
        if dq['missing_value_percentage'] > 5:
            html_content += f"""
            <div class='warning'>
                <h3>⚠️ Data Quality Warning</h3>
                <p>Missing values detected: {dq['missing_value_percentage']:.1f}%</p>
                <p>Consider implementing data imputation strategies.</p>
            </div>
            """
    
    html_content += """
    <h2>Detailed Analysis</h2>
    <p>For complete analysis results, please refer to:</p>
    <ul>
        <li>EDA Summary JSON: reports/docs/eda_summary.json</li>
        <li>Visualizations: reports/figures/</li>
        <li>Metrics: reports/metrics.json</li>
    </ul>
    
    <h2>Recommendations</h2>
    <ol>
        <li>Review pricing strategy for high-loss-ratio provinces</li>
        <li>Investigate risk factors for top risky vehicle types</li>
        <li>Monitor temporal trends for seasonal patterns</li>
        <li>Address data quality issues in columns with high missing values</li>
        <li>Consider further analysis on outlier cases</li>
    </ol>
    
    <footer>
        <p>Report generated by Insurance Risk Analysis Pipeline</p>
        <p>Data Version: Tracked with DVC</p>
    </footer>
</body>
</html>
"""
    
    report_path = "reports/docs/eda_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {report_path}")


if __name__ == "__main__":
    
    success = run_comprehensive_eda()
    sys.exit(0 if success else 1)