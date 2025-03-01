"""
Module for configuring and setting up logging for the AI trading agent.
"""
import os
import yaml
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from typing import Dict, Optional

def setup_logger(config_path: str = 'config/settings.yaml') -> Dict[str, logging.Logger]:
    """
    Set up logging configuration for the application.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, logging.Logger]: Dictionary of configured loggers
    """
    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging_config = config.get('logging', {})
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        logging_config = {
            'level': 'INFO',
            'file': 'logs/trading.log',
            'rotation': '1 day',
            'retention': '30 days'
        }
    
    # Create logs directory if it doesn't exist
    log_file = logging_config.get('file', 'logs/trading.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Set log level
    level_str = logging_config.get('level', 'INFO')
    level = getattr(logging, level_str)
    root_logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler with rotation
    rotation = logging_config.get('rotation', '1 day')
    retention_count = int(logging_config.get('retention', '30 days').split()[0])
    
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='D',  # Daily rotation
        interval=1,
        backupCount=retention_count
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Create specific loggers
    loggers = {
        'main': logging.getLogger('main'),
        'data_ingestion': logging.getLogger('data_ingestion'),
        'data_processing': logging.getLogger('data_processing'),
        'ai_module': logging.getLogger('ai_module'),
        'execution': logging.getLogger('execution')
    }
    
    # Log startup message
    root_logger.info("Logging system initialized")
    
    return loggers 