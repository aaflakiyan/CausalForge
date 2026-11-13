"""RobotGuard: deterministic kinematic pre-flight checks for robot actions."""
from .core import ConfigurationError, Result, RobotSpec, StreamGuard, Violation, validate_trace

__all__ = ["ConfigurationError", "Result", "RobotSpec", "StreamGuard", "Violation", "validate_trace"]
__version__ = "0.1.0"
