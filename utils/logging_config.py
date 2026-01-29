"""
Logging Configuration for Face Authentication Attendance System
Centralized logging with file and console output
"""

import logging
import os
from datetime import datetime
from pathlib import Path
import config

# Create logs directory
LOGS_DIR = config.BASE_DIR / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file with date
LOG_FILE = LOGS_DIR / f"attendance_system_{datetime.now().strftime('%Y%m%d')}.log"


def setup_logger(name: str = "attendance_system", level: str = None) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    level = level or config.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Global logger
logger = setup_logger()


def log_registration(employee_id: str, name: str, success: bool, message: str = ""):
    """Log user registration attempt"""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"REGISTRATION {status} - Employee: {employee_id}, Name: {name}, Message: {message}")


def log_authentication(user_id: int = None, user_name: str = None, 
                       success: bool = False, distance: float = None,
                       confidence: str = None, message: str = ""):
    """Log authentication attempt"""
    status = "SUCCESS" if success else "FAILED"
    details = f"User: {user_name or 'Unknown'}, Distance: {distance:.3f if distance else 'N/A'}, Confidence: {confidence or 'N/A'}"
    logger.info(f"AUTHENTICATION {status} - {details}, Message: {message}")


def log_attendance(action: str, user_id: int, user_name: str, 
                  success: bool, message: str = ""):
    """Log attendance marking (punch-in/out)"""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"ATTENDANCE {action.upper()} {status} - User: {user_name} (ID: {user_id}), Message: {message}")


def log_error(module: str, error: Exception, context: str = ""):
    """Log error with traceback"""
    logger.error(f"ERROR in {module} - {context}: {str(error)}", exc_info=True)


def log_performance(operation: str, duration_ms: float, details: str = ""):
    """Log performance metrics"""
    logger.debug(f"PERFORMANCE - {operation}: {duration_ms:.2f}ms, {details}")


def log_database_operation(operation: str, table: str, success: bool, details: str = ""):
    """Log database operations"""
    status = "SUCCESS" if success else "FAILED"
    logger.debug(f"DATABASE {operation.upper()} {status} - Table: {table}, {details}")


if __name__ == "__main__":
    # Test logging
    logger.info("Logging system initialized")
    logger.debug("Debug message test")
    logger.warning("Warning message test")
    logger.error("Error message test")
    
    print(f"\nLog file created at: {LOG_FILE}")
