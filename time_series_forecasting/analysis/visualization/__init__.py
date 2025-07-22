"""
Visualization Module
"""

from .plot_generator import (
    plot_time_series,
    plot_seasonal_patterns,
    plot_distribution,
    plot_trends,
    plot_anomalies,
    plot_correlation,
    create_report_plots
)

# Backward-compatibility aliases
create_time_series_plot = plot_time_series
create_seasonal_plots = plot_seasonal_patterns
create_distribution_plots = plot_distribution
create_trend_plots = plot_trends
create_seasonal_patterns = plot_seasonal_patterns
create_anomaly_plots = plot_anomalies
create_correlation_plots = plot_correlation

from .report_generator import (
    generate_basic_report,
    generate_comparison_report,
    generate_seasonal_report
)

__all__ = [
    'plot_time_series',
    'plot_seasonal_patterns',
    'plot_distribution',
    'plot_trends',
    'plot_anomalies',
    'plot_correlation',
    'create_report_plots',
    'create_time_series_plot',
    'create_seasonal_plots',
    'create_distribution_plots',
    'create_trend_plots',
    'create_seasonal_patterns',
    'create_anomaly_plots',
    'create_correlation_plots',
    'generate_basic_report',
    'generate_comparison_report',
    'generate_seasonal_report'
] 