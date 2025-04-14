"""
Topological Data Analysis (TDA) package for volatility regime identification.

This package provides tools for identifying volatility regimes using TDA techniques,
particularly the Mapper algorithm. It includes functions for filter creation, 
distance metrics, mapper execution, and regime analysis.
"""

from .filter_functions_v2 import FinancialLensFactory
from .distance_metrics_v2 import FinancialDistanceMetrics, create_financial_distance_function, compute_distance_matrix
from .mapper_core_v2 import FinancialMapperConfig, FinancialMapper

__all__ = [
    'FinancialLensFactory',
    'FinancialDistanceMetrics',
    'create_financial_distance_function',
    'compute_distance_matrix',
    'FinancialMapperConfig',
    'FinancialMapper'
] 