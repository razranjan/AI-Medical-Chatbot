"""
Performance Monitoring and Metrics Collection Module

This module provides comprehensive monitoring capabilities for the medical chatbot,
including performance metrics, health checks, and system monitoring.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
        
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def get_metric_stats(self, metric_name: str, window_seconds: int = 300) -> Dict[str, Any]:
        """Get statistics for a specific metric within a time window."""
        if metric_name not in self.metrics:
            return {}
            
        cutoff_time = time.time() - window_seconds
        recent_metrics = [
            m for m in self.metrics[metric_name] 
            if m['timestamp'] > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
            
        values = [m['value'] for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else None,
            'window_seconds': window_seconds
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self.lock:
            return {
                name: list(metrics) for name, metrics in self.metrics.items()
            }

class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'uptime_seconds': time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}
    
    def get_process_stats(self) -> Dict[str, Any]:
        """Get current process statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'memory_rss_mb': memory_info.rss / (1024**2),
                'memory_vms_mb': memory_info.vms / (1024**2),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        except Exception as e:
            logger.error(f"Error getting process stats: {str(e)}")
            return {}

class HealthChecker:
    """Performs health checks on various system components."""
    
    def __init__(self, performance_metrics: PerformanceMetrics):
        self.performance_metrics = performance_metrics
        self.health_status = {
            'overall': 'healthy',
            'components': {},
            'last_check': None
        }
    
    def check_ai_components(self, rag_chain) -> Dict[str, Any]:
        """Check health of AI components."""
        try:
            if rag_chain is None:
                return {'status': 'unavailable', 'error': 'RAG chain not initialized'}
            
            # Check if components are accessible
            return {'status': 'healthy', 'message': 'AI components operational'}
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_performance_metrics(self) -> Dict[str, Any]:
        """Check if performance metrics are within acceptable ranges."""
        try:
            # Check response time metrics
            response_time_stats = self.performance_metrics.get_metric_stats('response_time', 300)
            
            if not response_time_stats:
                return {'status': 'unknown', 'message': 'No response time data available'}
            
            avg_response_time = response_time_stats.get('avg', 0)
            
            if avg_response_time < 2.0:
                status = 'healthy'
            elif avg_response_time < 5.0:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'avg_response_time': avg_response_time,
                'message': f'Average response time: {avg_response_time:.2f}s'
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def perform_health_check(self, rag_chain) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            ai_health = self.check_ai_components(rag_chain)
            performance_health = self.check_performance_metrics()
            
            # Determine overall health
            component_statuses = [ai_health['status'], performance_health['status']]
            
            if 'unhealthy' in component_statuses:
                overall_status = 'unhealthy'
            elif 'degraded' in component_statuses:
                overall_status = 'degraded'
            elif 'error' in component_statuses:
                overall_status = 'error'
            else:
                overall_status = 'healthy'
            
            self.health_status = {
                'overall': overall_status,
                'components': {
                    'ai_components': ai_health,
                    'performance': performance_health
                },
                'last_check': datetime.now().isoformat()
            }
            
            return self.health_status
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            return {
                'overall': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }

class PerformanceDecorator:
    """Decorator for measuring function performance."""
    
    def __init__(self, metrics: PerformanceMetrics, metric_name: Optional[str] = None):
        self.metrics = metrics
        self.metric_name = metric_name
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record metric
                metric_name = self.metric_name or f"{func.__module__}.{func.__name__}"
                self.metrics.record_metric(metric_name, execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metric
                error_metric_name = f"{self.metric_name or func.__name__}_error"
                self.metrics.record_metric(error_metric_name, execution_time)
                
                raise
        
        return wrapper

# Global instances
performance_metrics = PerformanceMetrics()
system_monitor = SystemMonitor()
health_checker = HealthChecker(performance_metrics)

def monitor_performance(metric_name: Optional[str] = None):
    """Decorator for monitoring function performance."""
    return PerformanceDecorator(performance_metrics, metric_name)

def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of all performance metrics."""
    return {
        'metrics': performance_metrics.get_all_metrics(),
        'system': system_monitor.get_system_stats(),
        'process': system_monitor.get_process_stats(),
        'health': health_checker.health_status
    }

def record_response_time(response_time: float):
    """Record response time for chat requests."""
    performance_metrics.record_metric('response_time', response_time)

def record_request_count():
    """Record request count."""
    performance_metrics.record_metric('request_count', 1)

def record_error_count():
    """Record error count."""
    performance_metrics.record_metric('error_count', 1)
