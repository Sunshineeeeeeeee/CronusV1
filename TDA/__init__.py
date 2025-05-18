"""
CronusV1 TDA Module - Topological Data Analysis for Financial Time Series
"""

from .distance_metrics_v2 import (
    FinancialDistanceMetrics,
    create_financial_distance_function,
    compute_distance_matrix,
    configure_tda_logging
)

from .mapper_core_v2 import (
    FinancialMapperConfig,
    FinancialMapper
)

try:
    from .filter_functions_v2 import FinancialLensFactory
except ImportError:
    pass

# Configure TDA logging with minimal verbosity by default
configure_tda_logging()

__all__ = [
    'FinancialDistanceMetrics',
    'FinancialMapperConfig',
    'FinancialMapper',
    'FinancialLensFactory',
    'create_financial_distance_function',
    'compute_distance_matrix',
    'configure_tda_logging'
] 