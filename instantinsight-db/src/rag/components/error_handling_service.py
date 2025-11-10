"""
Error Handling & Monitoring Service for V2 Architecture.

This component centralizes error handling, logging, performance monitoring,
and observability functionality that was previously scattered across the monolithic VannaRAGEngine.
"""

import threading
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from loguru import logger


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    DATABASE_CONNECTION = "database_connection"
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    TRAINING_DATA = "training_data"
    SCHEMA_MANAGEMENT = "schema_management"
    AGENT_EXECUTION = "agent_execution"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling and monitoring."""

    enable_error_tracking: bool = True
    enable_performance_monitoring: bool = True
    enable_health_checks: bool = True
    max_error_history: int = 1000
    max_performance_history: int = 500
    error_aggregation_window_minutes: int = 5
    alert_threshold_errors_per_minute: int = 10
    enable_error_recovery: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    enable_metrics_export: bool = True
    metrics_export_interval_seconds: int = 60


@dataclass
class ErrorRecord:
    """Represents a single error occurrence."""

    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    error_type: str
    error_message: str
    stack_trace: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    recovery_attempted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "component": self.component,
            "operation": self.operation,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "resolved": self.resolved,
            "recovery_attempted": self.recovery_attempted,
        }


@dataclass
class PerformanceMetric:
    """Represents a performance measurement."""

    metric_id: str
    timestamp: datetime
    component: str
    operation: str
    duration_seconds: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "metric_id": self.metric_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    component: str
    healthy: bool
    response_time_ms: float
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""

    component: str
    state: str  # 'closed', 'open', 'half_open'
    failure_count: int = 0
    last_failure_time: datetime | None = None
    next_attempt_time: datetime | None = None


class ErrorHandlingService:
    """
    Service for centralized error handling, monitoring, and observability.

    Features:
    - Comprehensive error tracking and classification
    - Performance monitoring and metrics collection
    - Health checks and component status monitoring
    - Circuit breaker pattern for fault tolerance
    - Error recovery and retry mechanisms
    - Real-time alerting and notification
    - Metrics export and observability
    """

    def __init__(self, config: ErrorHandlingConfig | None = None):
        """
        Initialize error handling service.

        Args:
            config: Configuration for error handling and monitoring

        """
        self.config = config or ErrorHandlingConfig()

        # Error tracking
        self.error_history: deque = deque(maxlen=self.config.max_error_history)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.error_rates: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Performance monitoring
        self.performance_history: deque = deque(
            maxlen=self.config.max_performance_history
        )
        self.performance_stats: dict[str, dict[str, Any]] = defaultdict(dict)

        # Health checks
        self.health_check_registry: dict[str, Callable] = {}
        self.last_health_check: dict[str, HealthCheckResult] = {}

        # Circuit breakers
        self.circuit_breakers: dict[str, CircuitBreakerState] = {}

        # Recovery handlers
        self.recovery_handlers: dict[str, Callable] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Error aggregation
        self.error_aggregation_window = timedelta(
            minutes=self.config.error_aggregation_window_minutes
        )

        logger.info("ErrorHandlingService initialized")

    def record_error(
        self,
        component: str,
        operation: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: dict[str, Any] | None = None,
        attempt_recovery: bool = True,
    ) -> str:
        """
        Record an error occurrence.

        Args:
            component: Component where error occurred
            operation: Operation that failed
            error: The exception that occurred
            severity: Severity level of the error
            category: Category classification
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            Error ID for tracking

        """
        if not self.config.enable_error_tracking:
            return ""

        with self._lock:
            error_id = f"{component}_{operation}_{int(time.time() * 1000)}"

            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                component=component,
                operation=operation,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                context=context or {},
            )

            # Add to history
            self.error_history.append(error_record)

            # Update error counts and rates
            error_key = f"{component}:{category.value}"
            self.error_counts[error_key] += 1
            self.error_rates[error_key].append(datetime.now())

            # Update circuit breaker
            self._update_circuit_breaker(component, False)

            # Attempt recovery if enabled
            if attempt_recovery and self.config.enable_error_recovery:
                recovery_success = self._attempt_error_recovery(
                    component, operation, error, context
                )
                error_record.recovery_attempted = True
                if recovery_success:
                    error_record.resolved = True

            # Check for alert conditions
            self._check_alert_conditions(component, category)

            # Log the error
            logger.error(
                f"Error in {component}.{operation}: {error_record.error_message}",
                extra={
                    "error_id": error_id,
                    "severity": severity.value,
                    "category": category.value,
                    "component": component,
                    "operation": operation,
                },
            )

            return error_id

    def record_performance_metric(
        self,
        component: str,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record a performance metric.

        Args:
            component: Component that performed the operation
            operation: Operation that was performed
            duration_seconds: Duration of the operation
            success: Whether the operation was successful
            metadata: Additional metadata

        Returns:
            Metric ID for tracking

        """
        if not self.config.enable_performance_monitoring:
            return ""

        with self._lock:
            metric_id = f"{component}_{operation}_{int(time.time() * 1000)}"

            # Create performance metric
            metric = PerformanceMetric(
                metric_id=metric_id,
                timestamp=datetime.now(),
                component=component,
                operation=operation,
                duration_seconds=duration_seconds,
                success=success,
                metadata=metadata or {},
            )

            # Add to history
            self.performance_history.append(metric)

            # Update performance stats
            stats_key = f"{component}:{operation}"
            if stats_key not in self.performance_stats:
                self.performance_stats[stats_key] = {
                    "count": 0,
                    "total_duration": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "min_duration": float("inf"),
                    "max_duration": 0,
                    "last_updated": datetime.now(),
                }

            stats = self.performance_stats[stats_key]
            stats["count"] += 1
            stats["total_duration"] += duration_seconds
            stats["min_duration"] = min(stats["min_duration"], duration_seconds)
            stats["max_duration"] = max(stats["max_duration"], duration_seconds)
            stats["last_updated"] = datetime.now()

            if success:
                stats["success_count"] += 1
                # Update circuit breaker on success
                self._update_circuit_breaker(component, True)
            else:
                stats["failure_count"] += 1

            return metric_id

    def register_health_check(
        self, component: str, health_check_func: Callable[[], bool]
    ):
        """
        Register a health check function for a component.

        Args:
            component: Name of the component
            health_check_func: Function that returns True if healthy

        """
        if self.config.enable_health_checks:
            self.health_check_registry[component] = health_check_func
            logger.debug(f"Registered health check for {component}")

    def run_health_checks(self) -> dict[str, HealthCheckResult]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of health check results by component

        """
        if not self.config.enable_health_checks:
            return {}

        results = {}

        for component, health_check_func in self.health_check_registry.items():
            start_time = time.time()

            try:
                is_healthy = health_check_func()
                response_time = (time.time() - start_time) * 1000  # Convert to ms

                result = HealthCheckResult(
                    component=component,
                    healthy=is_healthy,
                    response_time_ms=response_time,
                )

            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                result = HealthCheckResult(
                    component=component,
                    healthy=False,
                    response_time_ms=response_time,
                    error_message=str(e),
                )

                # Record the health check failure as an error
                self.record_error(
                    component=component,
                    operation="health_check",
                    error=e,
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.CONFIGURATION,
                )

            results[component] = result
            self.last_health_check[component] = result

        return results

    def register_recovery_handler(
        self,
        component: str,
        recovery_func: Callable[[str, Exception, dict[str, Any] | None], bool],
    ):
        """
        Register an error recovery handler for a component.

        Args:
            component: Name of the component
            recovery_func: Function that attempts to recover from errors

        """
        if self.config.enable_error_recovery:
            self.recovery_handlers[component] = recovery_func
            logger.debug(f"Registered recovery handler for {component}")

    def get_error_summary(self, time_window_minutes: int = 60) -> dict[str, Any]:
        """
        Get summary of errors within a time window.

        Args:
            time_window_minutes: Time window to analyze

        Returns:
            Error summary statistics

        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        recent_errors = [
            error for error in self.error_history if error.timestamp >= cutoff_time
        ]

        # Group by severity
        severity_counts = defaultdict(int)
        for error in recent_errors:
            severity_counts[error.severity.value] += 1

        # Group by category
        category_counts = defaultdict(int)
        for error in recent_errors:
            category_counts[error.category.value] += 1

        # Group by component
        component_counts = defaultdict(int)
        for error in recent_errors:
            component_counts[error.component] += 1

        return {
            "time_window_minutes": time_window_minutes,
            "total_errors": len(recent_errors),
            "severity_breakdown": dict(severity_counts),
            "category_breakdown": dict(category_counts),
            "component_breakdown": dict(component_counts),
            "error_rate_per_minute": len(recent_errors) / max(1, time_window_minutes),
        }

    def get_performance_summary(self, component: str | None = None) -> dict[str, Any]:
        """
        Get performance summary statistics.

        Args:
            component: Specific component to analyze, or None for all

        Returns:
            Performance summary statistics

        """
        summary = {}

        for stats_key, stats in self.performance_stats.items():
            comp, operation = stats_key.split(":", 1)

            if component and comp != component:
                continue

            avg_duration = (
                stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
            )
            success_rate = (
                stats["success_count"] / stats["count"] if stats["count"] > 0 else 0
            )

            summary[stats_key] = {
                "component": comp,
                "operation": operation,
                "total_calls": stats["count"],
                "success_count": stats["success_count"],
                "failure_count": stats["failure_count"],
                "success_rate": success_rate,
                "avg_duration_seconds": avg_duration,
                "min_duration_seconds": (
                    stats["min_duration"]
                    if stats["min_duration"] != float("inf")
                    else 0
                ),
                "max_duration_seconds": stats["max_duration"],
                "last_updated": stats["last_updated"].isoformat(),
            }

        return summary

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """
        Get status of all circuit breakers.

        Returns:
            Circuit breaker status information

        """
        if not self.config.enable_circuit_breaker:
            return {}

        status = {}

        for component, breaker in self.circuit_breakers.items():
            status[component] = {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "last_failure_time": (
                    breaker.last_failure_time.isoformat()
                    if breaker.last_failure_time
                    else None
                ),
                "next_attempt_time": (
                    breaker.next_attempt_time.isoformat()
                    if breaker.next_attempt_time
                    else None
                ),
            }

        return status

    def is_circuit_breaker_open(self, component: str) -> bool:
        """
        Check if circuit breaker is open for a component.

        Args:
            component: Component to check

        Returns:
            True if circuit breaker is open (calls should be blocked)

        """
        if not self.config.enable_circuit_breaker:
            return False

        if component not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[component]

        if breaker.state == "open":
            # Check if we should try half-open
            if (
                breaker.next_attempt_time
                and datetime.now() >= breaker.next_attempt_time
            ):
                breaker.state = "half_open"
                logger.info(f"Circuit breaker for {component} moved to half-open state")
                return False
            return True

        return False

    def _update_circuit_breaker(self, component: str, success: bool):
        """Update circuit breaker state based on operation result."""
        if not self.config.enable_circuit_breaker:
            return

        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerState(
                component=component, state="closed"
            )

        breaker = self.circuit_breakers[component]

        if success:
            if breaker.state == "half_open":
                # Success in half-open state, close the breaker
                breaker.state = "closed"
                breaker.failure_count = 0
                logger.info(
                    f"Circuit breaker for {component} closed after successful operation"
                )
            elif breaker.state == "closed":
                # Reset failure count on success
                breaker.failure_count = max(0, breaker.failure_count - 1)
        else:
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()

            if breaker.failure_count >= self.config.circuit_breaker_failure_threshold:
                breaker.state = "open"
                breaker.next_attempt_time = datetime.now() + timedelta(
                    seconds=self.config.circuit_breaker_timeout_seconds
                )
                logger.warning(
                    f"Circuit breaker for {component} opened after {breaker.failure_count} failures"
                )

    def _attempt_error_recovery(
        self,
        component: str,
        operation: str,
        error: Exception,
        context: dict[str, Any] | None,
    ) -> bool:
        """Attempt to recover from an error using registered handlers."""
        if component not in self.recovery_handlers:
            return False

        try:
            recovery_func = self.recovery_handlers[component]
            success = recovery_func(operation, error, context)

            if success:
                logger.info(
                    f"Successfully recovered from error in {component}.{operation}"
                )
            else:
                logger.warning(f"Recovery attempt failed for {component}.{operation}")

            return success

        except Exception as recovery_error:
            logger.error(
                f"Error during recovery attempt for {component}.{operation}: {recovery_error}"
            )
            return False

    def _check_alert_conditions(self, component: str, category: ErrorCategory):
        """Check if alert conditions are met and trigger alerts."""
        # Simple rate-based alerting
        error_key = f"{component}:{category.value}"
        recent_errors = self.error_rates[error_key]

        # Count errors in the last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_count = sum(
            1 for timestamp in recent_errors if timestamp >= one_minute_ago
        )

        if recent_count >= self.config.alert_threshold_errors_per_minute:
            logger.warning(
                f"ALERT: High error rate detected for {component}:{category.value} "
                f"({recent_count} errors in last minute)"
            )

    def export_metrics(self) -> dict[str, Any]:
        """
        Export all metrics for external monitoring systems.

        Returns:
            Comprehensive metrics data

        """
        if not self.config.enable_metrics_export:
            return {}

        return {
            "timestamp": datetime.now().isoformat(),
            "error_summary": self.get_error_summary(),
            "performance_summary": self.get_performance_summary(),
            "circuit_breaker_status": self.get_circuit_breaker_status(),
            "health_check_results": {
                component: result.__dict__
                for component, result in self.last_health_check.items()
            },
            "system_stats": {
                "total_errors_tracked": len(self.error_history),
                "total_metrics_tracked": len(self.performance_history),
                "active_circuit_breakers": len(self.circuit_breakers),
                "registered_health_checks": len(self.health_check_registry),
                "registered_recovery_handlers": len(self.recovery_handlers),
            },
        }

    def reset_stats(self):
        """Reset all statistics and history."""
        with self._lock:
            self.error_history.clear()
            self.error_counts.clear()
            self.error_rates.clear()
            self.performance_history.clear()
            self.performance_stats.clear()
            self.last_health_check.clear()
            self.circuit_breakers.clear()

        logger.info("ErrorHandlingService stats reset")

    def get_component_health(self, component: str) -> dict[str, Any]:
        """
        Get comprehensive health status for a specific component.

        Args:
            component: Component to check

        Returns:
            Health status information

        """
        health_info = {
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "issues": [],
        }

        # Check recent errors
        recent_errors = [
            error
            for error in self.error_history
            if error.component == component
            and error.timestamp >= datetime.now() - timedelta(minutes=5)
        ]

        if recent_errors:
            health_info["overall_healthy"] = False
            health_info["issues"].append(
                f"{len(recent_errors)} errors in last 5 minutes"
            )

        # Check circuit breaker
        if component in self.circuit_breakers:
            breaker = self.circuit_breakers[component]
            if breaker.state == "open":
                health_info["overall_healthy"] = False
                health_info["issues"].append("Circuit breaker is open")

        # Check health check result
        if component in self.last_health_check:
            result = self.last_health_check[component]
            if not result.healthy:
                health_info["overall_healthy"] = False
                health_info["issues"].append(
                    f"Health check failed: {result.error_message}"
                )

        # Add performance stats
        performance_stats = self.get_performance_summary(component)
        health_info["performance"] = performance_stats

        return health_info
