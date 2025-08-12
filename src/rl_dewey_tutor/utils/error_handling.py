"""
Comprehensive Error Handling and Recovery System

This module provides robust error handling, recovery mechanisms,
and graceful degradation for the RL-Dewey-Tutor system.
"""

import sys
import traceback
import logging
import time
import signal
import pickle
import json
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import threading
from pathlib import Path

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ErrorReport:
    """Structured error report"""
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None

class RobustErrorHandler:
    """
    Comprehensive error handler with recovery strategies
    
    Features:
    - Automatic error classification
    - Multiple recovery strategies
    - Graceful degradation
    - Comprehensive logging
    - Emergency shutdown protocols
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 log_file: Optional[str] = None,
                 emergency_save_dir: str = "emergency_saves"):
        """
        Initialize error handler
        
        Args:
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            log_file: Log file path (None for console only)
            emergency_save_dir: Directory for emergency state saves
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.emergency_save_dir = Path(emergency_save_dir)
        self.emergency_save_dir.mkdir(exist_ok=True)
        
        # Error tracking
        self.error_history: List[ErrorReport] = []
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, float] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.fallback_functions: Dict[str, Callable] = {}
        
        # Emergency protocols
        self.emergency_handlers: List[Callable] = []
        self.shutdown_requested = False
        
        # Setup logging
        self._setup_logging(log_file)
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self, log_file: Optional[str]):
        """Setup logging configuration"""
        self.logger = logging.getLogger("RobustErrorHandler")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_format)
            self.logger.addHandler(file_handler)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_requested = True
            self._trigger_emergency_protocol()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def robust_execute(self, 
                      func: Callable, 
                      *args, 
                      error_context: Optional[Dict[str, Any]] = None,
                      recovery_strategy: Optional[RecoveryStrategy] = None,
                      fallback_func: Optional[Callable] = None,
                      **kwargs) -> Any:
        """
        Execute function with robust error handling
        
        Args:
            func: Function to execute
            *args: Function arguments
            error_context: Additional context for error reporting
            recovery_strategy: Specific recovery strategy to use
            fallback_func: Fallback function if main function fails
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        func_name = func.__name__
        context = error_context or {}
        context.update({
            'function_name': func_name,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        })
        
        # Check if we should attempt execution
        if self._should_skip_execution(func_name):
            self.logger.warning(f"Skipping execution of {func_name} due to recent failures")
            if fallback_func:
                return self._execute_fallback(fallback_func, args, kwargs, context)
            return None
        
        # Attempt execution with retries
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for {func_name}")
                    time.sleep(self.retry_delay * attempt)  # Exponential backoff
                
                result = func(*args, **kwargs)
                
                # Success - reset error count
                if func_name in self.error_counts:
                    self.error_counts[func_name] = 0
                
                return result
                
            except Exception as e:
                last_exception = e
                severity = self._classify_error(e, context)
                
                error_report = ErrorReport(
                    timestamp=time.time(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    severity=severity,
                    context=context.copy()
                )
                
                self._log_error(error_report)
                
                # Update error tracking
                self.error_counts[func_name] = self.error_counts.get(func_name, 0) + 1
                self.last_errors[func_name] = time.time()
                
                # Check if we should continue retrying
                if severity == ErrorSeverity.CRITICAL or attempt == self.max_retries:
                    break
                
                # Apply recovery strategy
                if recovery_strategy:
                    recovery_success = self._apply_recovery_strategy(
                        recovery_strategy, error_report, func, args, kwargs
                    )
                    error_report.recovery_attempted = True
                    error_report.recovery_successful = recovery_success
                    error_report.recovery_strategy = recovery_strategy
                    
                    if recovery_success:
                        continue
        
        # All retries failed
        self.logger.error(f"All retry attempts failed for {func_name}")
        
        # Try fallback function
        if fallback_func:
            try:
                return self._execute_fallback(fallback_func, args, kwargs, context)
            except Exception as fallback_error:
                self.logger.error(f"Fallback function also failed: {fallback_error}")
        
        # Final error handling
        if self._should_trigger_emergency(last_exception, context):
            self._trigger_emergency_protocol()
        
        # Re-raise the last exception
        raise last_exception
    
    def _classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Classify error severity based on type and context"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Critical errors
        if error_type in ['SystemExit', 'KeyboardInterrupt', 'MemoryError']:
            return ErrorSeverity.CRITICAL
        
        if 'out of memory' in error_msg or 'cuda' in error_msg:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['RuntimeError', 'AssertionError']:
            return ErrorSeverity.HIGH
        
        if 'training' in context.get('function_name', '').lower():
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (default)
        return ErrorSeverity.LOW
    
    def _log_error(self, error_report: ErrorReport):
        """Log error with appropriate level"""
        self.error_history.append(error_report)
        
        log_msg = f"Error in {error_report.context.get('function_name', 'unknown')}: {error_report.error_message}"
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg)
        elif error_report.severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
        
        # Log full details at debug level
        self.logger.debug(f"Full error details: {asdict(error_report)}")
    
    def _should_skip_execution(self, func_name: str) -> bool:
        """Determine if function execution should be skipped"""
        if func_name not in self.error_counts:
            return False
        
        error_count = self.error_counts[func_name]
        last_error_time = self.last_errors.get(func_name, 0)
        
        # Skip if too many recent errors
        if error_count >= 5 and (time.time() - last_error_time) < 300:  # 5 minutes
            return True
        
        return False
    
    def _apply_recovery_strategy(self, 
                               strategy: RecoveryStrategy,
                               error_report: ErrorReport,
                               func: Callable,
                               args: tuple,
                               kwargs: dict) -> bool:
        """Apply specific recovery strategy"""
        try:
            if strategy == RecoveryStrategy.RETRY:
                # Simple retry (handled by main loop)
                return True
            
            elif strategy == RecoveryStrategy.FALLBACK:
                func_name = func.__name__
                if func_name in self.fallback_functions:
                    fallback_result = self.fallback_functions[func_name](*args, **kwargs)
                    self.logger.info(f"Fallback strategy successful for {func_name}")
                    return True
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Attempt to modify parameters for simpler execution
                simplified_kwargs = self._simplify_parameters(kwargs)
                result = func(*args, **simplified_kwargs)
                self.logger.info(f"Graceful degradation successful for {func.__name__}")
                return True
            
            elif strategy == RecoveryStrategy.EMERGENCY_STOP:
                self._trigger_emergency_protocol()
                return False
        
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy {strategy.value} failed: {recovery_error}")
            return False
        
        return False
    
    def _simplify_parameters(self, kwargs: dict) -> dict:
        """Simplify parameters for graceful degradation"""
        simplified = kwargs.copy()
        
        # Common simplifications
        if 'batch_size' in simplified:
            simplified['batch_size'] = min(simplified['batch_size'], 32)
        
        if 'learning_rate' in simplified:
            simplified['learning_rate'] = min(simplified['learning_rate'], 0.001)
        
        if 'max_steps' in simplified:
            simplified['max_steps'] = min(simplified['max_steps'], 100)
        
        if 'n_topics' in simplified:
            simplified['n_topics'] = min(simplified['n_topics'], 2)
        
        return simplified
    
    def _execute_fallback(self, 
                         fallback_func: Callable,
                         args: tuple,
                         kwargs: dict,
                         context: Dict[str, Any]) -> Any:
        """Execute fallback function with error handling"""
        self.logger.info(f"Executing fallback function for {context.get('function_name', 'unknown')}")
        try:
            return fallback_func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Fallback function failed: {e}")
            raise
    
    def _should_trigger_emergency(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Determine if emergency protocol should be triggered"""
        # Trigger on critical errors
        if self._classify_error(error, context) == ErrorSeverity.CRITICAL:
            return True
        
        # Trigger if too many recent errors across the system
        recent_errors = [
            err for err in self.error_history 
            if time.time() - err.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_errors) > 10:
            return True
        
        return False
    
    def _trigger_emergency_protocol(self):
        """Trigger emergency protocol for system shutdown"""
        self.logger.critical("EMERGENCY PROTOCOL ACTIVATED")
        self.shutdown_requested = True
        
        # Execute emergency handlers
        for handler in self.emergency_handlers:
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Emergency handler failed: {e}")
        
        # Save system state
        self._emergency_save_state()
    
    def _emergency_save_state(self):
        """Save system state for recovery"""
        timestamp = int(time.time())
        save_path = self.emergency_save_dir / f"emergency_state_{timestamp}.json"
        
        try:
            state_data = {
                'timestamp': timestamp,
                'error_history': [asdict(err) for err in self.error_history[-10:]],
                'error_counts': self.error_counts,
                'shutdown_reason': 'emergency_protocol'
            }
            
            with open(save_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self.logger.info(f"Emergency state saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency state: {e}")
    
    def register_emergency_handler(self, handler: Callable):
        """Register emergency shutdown handler"""
        self.emergency_handlers.append(handler)
    
    def register_fallback_function(self, func_name: str, fallback_func: Callable):
        """Register fallback function for a specific function"""
        self.fallback_functions[func_name] = fallback_func
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        recent_errors = [
            err for err in self.error_history 
            if time.time() - err.timestamp < 3600  # Last hour
        ]
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_counts_by_function': dict(self.error_counts),
            'error_counts_by_severity': {
                severity.value: len([
                    err for err in recent_errors 
                    if err.severity == severity
                ])
                for severity in ErrorSeverity
            },
            'shutdown_requested': self.shutdown_requested,
            'last_error_time': max(self.last_errors.values()) if self.last_errors else None
        }
    
    def reset_error_tracking(self):
        """Reset error tracking (use carefully)"""
        self.error_counts.clear()
        self.last_errors.clear()
        self.shutdown_requested = False
        self.logger.info("Error tracking reset")

# Decorator for robust function execution
def robust_execution(max_retries: int = 3,
                    recovery_strategy: Optional[RecoveryStrategy] = None,
                    fallback_func: Optional[Callable] = None,
                    error_handler: Optional[RobustErrorHandler] = None):
    """
    Decorator for robust function execution
    
    Args:
        max_retries: Maximum retry attempts
        recovery_strategy: Recovery strategy to use
        fallback_func: Fallback function
        error_handler: Error handler instance (creates new if None)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or RobustErrorHandler(max_retries=max_retries)
            return handler.robust_execute(
                func, *args,
                recovery_strategy=recovery_strategy,
                fallback_func=fallback_func,
                **kwargs
            )
        return wrapper
    return decorator

# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> RobustErrorHandler:
    """Get or create global error handler"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
    return _global_error_handler

def safe_execute(func: Callable, *args, **kwargs) -> Any:
    """Execute function with global error handler"""
    handler = get_global_error_handler()
    return handler.robust_execute(func, *args, **kwargs)
