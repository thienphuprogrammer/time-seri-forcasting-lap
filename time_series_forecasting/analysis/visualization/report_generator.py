"""
Report Generator Module for Time Series Analysis
"""

from typing import Dict, Any, Optional
import pandas as pd

def generate_basic_report(stats: Dict[str, Any], region: str = None) -> str:
    """
    Generate basic analysis report
    
    Args:
        stats: Dictionary containing statistics
        region: Region name
        
    Returns:
        Formatted report string
    """
    report = f"""
TIME SERIES ANALYSIS REPORT
{'='*50}

1. OVERVIEW:
   - {'Region: ' + region if region else ''}
   - Total records: {stats['total_records']:,}
   - Time range: {stats['date_range']['start']} to {stats['date_range']['end']}
   - Duration: {stats['date_range']['duration_years']:.1f} years
   - Missing values: {stats['missing_values']}
   - Duplicate records: {stats['duplicates']}

2. CONSUMPTION STATISTICS (MW):
   - Mean: {stats['statistics']['mean']:.2f}
   - Median: {stats['statistics']['50%']:.2f}
   - Standard deviation: {stats['statistics']['std']:.2f}
   - Minimum: {stats['statistics']['min']:.2f}
   - Maximum: {stats['statistics']['max']:.2f}
   - 25th percentile: {stats['statistics']['25%']:.2f}
   - 75th percentile: {stats['statistics']['75%']:.2f}

3. DATA QUALITY:
   - Completeness: {((stats['total_records'] - stats['missing_values']) / stats['total_records'] * 100):.2f}%
   - Duplicate-free: {'Yes' if stats['duplicates'] == 0 else 'No'}

4. RECOMMENDATIONS:
   - Data is suitable for time series forecasting
   - Consider normalizing before modeling
   - Suggested window size: 24 hours for daily patterns
    """
    
    return report.strip()

def generate_comparison_report(comparison_data: Dict[str, pd.DataFrame], stats_df: pd.DataFrame) -> str:
    """
    Generate comparison report for multiple regions
    
    Args:
        comparison_data: Dictionary of DataFrames for each region
        stats_df: DataFrame containing comparison statistics
        
    Returns:
        Formatted report string
    """
    report = f"""
REGION COMPARISON REPORT
{'='*50}

1. REGIONS ANALYZED:
   - Total regions: {len(comparison_data)}
   - Regions: {', '.join(comparison_data.keys())}

2. CONSUMPTION COMPARISON:
"""
    
    for region in comparison_data.keys():
        stats = stats_df[stats_df['Region'] == region].iloc[0]
        report += f"""
   {region}:
   - Mean consumption: {stats['Mean']:.2f} MW
   - Peak consumption: {stats['Max']:.2f} MW
   - Variability (std): {stats['Std']:.2f} MW
        """
    
    report += """
3. KEY FINDINGS:
   - Region with highest consumption: """ + stats_df.loc[stats_df['Mean'].idxmax(), 'Region'] + """
   - Region with most variability: """ + stats_df.loc[stats_df['Std'].idxmax(), 'Region'] + """
   - Region with highest peak: """ + stats_df.loc[stats_df['Max'].idxmax(), 'Region']
    
    return report.strip()

def generate_seasonal_report(seasonal_stats: Dict[str, pd.Series]) -> str:
    """
    Generate seasonal analysis report
    
    Args:
        seasonal_stats: Dictionary containing seasonal statistics
        
    Returns:
        Formatted report string
    """
    monthly_avg = seasonal_stats['monthly_avg']
    daily_avg = seasonal_stats['daily_avg']
    hourly_avg = seasonal_stats['hourly_avg']
    
    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()
    peak_hour = hourly_avg.idxmax()
    low_hour = hourly_avg.idxmin()
    
    report = f"""
SEASONAL ANALYSIS REPORT
{'='*50}

1. MONTHLY PATTERNS:
   - Peak consumption month: {peak_month} ({monthly_avg[peak_month]:.2f} MW)
   - Lowest consumption month: {low_month} ({monthly_avg[low_month]:.2f} MW)
   - Seasonal variation: {(monthly_avg.max() - monthly_avg.min()):.2f} MW

2. DAILY PATTERNS:
   - Weekday vs Weekend difference: {(daily_avg[:5].mean() - daily_avg[5:].mean()):.2f} MW
   - Most active day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][daily_avg.idxmax()]}
   - Least active day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][daily_avg.idxmin()]}

3. HOURLY PATTERNS:
   - Peak hour: {peak_hour}:00 ({hourly_avg[peak_hour]:.2f} MW)
   - Low hour: {low_hour}:00 ({hourly_avg[low_hour]:.2f} MW)
   - Daily variation: {(hourly_avg.max() - hourly_avg.min()):.2f} MW

4. KEY INSIGHTS:
   - Clear seasonal pattern with peak in {peak_month}
   - Significant daily pattern with peak at {peak_hour}:00
   - Notable weekday vs weekend differences
    """
    
    return report.strip() 