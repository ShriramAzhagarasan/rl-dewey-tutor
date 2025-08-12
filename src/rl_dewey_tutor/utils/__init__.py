"""
Utilities module for RL-Dewey-Tutor

This module provides utility functions and classes for robust
error handling, logging, and system management.
"""

from .error_handling import (
    RobustErrorHandler,
    robust_execution,
    RecoveryStrategy,
    ErrorSeverity,
    ErrorReport,
    get_global_error_handler,
    safe_execute
)

__all__ = [
    'RobustErrorHandler',
    'robust_execution',
    'RecoveryStrategy',
    'ErrorSeverity',
    'ErrorReport',
    'get_global_error_handler',
    'safe_execute'
]
