"""
Main entry point for the AI trading agent application.
"""
import os
import sys
import time
import yaml
import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import argparse

# Import utility modules
from utils.logger import setup_logger
from utils.db_handler import DatabaseHandler

# Import data collection modules
from data_ingestion.liquidation_collector import LiquidationCollector
from data_ingestion.funding_rate_collector import FundingRateCollector
from data_ingestion.open_interest_collector import OpenInterestCollector
from data_ingestion.token_launches_collector import TokenLaunchesCollector
from data_ingestion.transactions_collector import TransactionsCollector

# Import data processing modules
from data_processing.data_cleaner import DataCleaner
from data_processing.feature_engineering import FeatureEngineer

# Import AI modules
from ai_module.model_trainer import ModelTrainer
from ai_module.prediction_engine import PredictionEngine

# Import execution modules
from execution.order_manager import OrderManager

class AITradingAgent:
    """Main orchestrator for the AI trading agent."""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """
        Initialize the AI trading agent.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Set up logging
        self.loggers = setup_logger(config_path)
        self.logger = self.loggers['main']
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Initialize components
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        
        # Initialize database
        self.db_handler = DatabaseHandler(config_path)
        
        # Initialize data collectors
        self._init_data_collectors()
        
        # Initialize AI components
        self._init_ai_components()
        
        # Initialize order manager
        self._init_order_manager()
        
        # Data storage
        self.liquidation_data = []
        self.funding_rate_data = []
        self.open_interest_data = []
        self.token_launches_data = []
        self.transaction_data = {}
        
        self.logger.info("AI Trading Agent initialized")
    
    def _init_data_collectors(self) -> None:
        """Initialize data collection components."""
        # Liquidation collector
        liq_config = self.config.get('data_collection', {}).get('liquidation', {})
        if liq_config.get('enabled', False):
            self.liquidation_collector = LiquidationCollector(
                exchanges=liq_config.get('exchanges', []),
                config_path=self.config_path
            )
            self.liquidation_collector.register_callback(self._on_liquidation_data)
        else:
            self.liquidation_collector = None
        
        # Funding rate collector
        fr_config = self.config.get('data_collection', {}).get('funding_rate', {})
        if fr_config.get('enabled', False):
            self.funding_rate_collector = FundingRateCollector(
                exchanges=fr_config.get('exchanges', []),
                config_path=self.config_path
            )
        else:
            self.funding_rate_collector = None
        
        # Open interest collector
        oi_config = self.config.get('data_collection', {}).get('open_interest', {})
        if oi_config.get('enabled', False):
            self.open_interest_collector = OpenInterestCollector(
                exchanges=oi_config.get('exchanges', []),
                config_path=self.config_path
            )
        else:
            self.open_interest_collector = None
        
        # Token launches collector
        tl_config = self.config.get('data_collection', {}).get('token_launches', {})
        if tl_config.get('enabled', False):
            self.token_launches_collector = TokenLaunchesCollector(
                sources=tl_config.get('sources', []),
                config_path=self.config_path
            )
        else:
            self.token_launches_collector = None
        
        # Transactions collector
        tx_config = self.config.get('data_collection', {}).get('transactions', {})
        if tx_config.get('enabled', False):
            self.transactions_collector = TransactionsCollector(
                exchanges=tx_config.get('exchanges', []),
                symbols=tx_config.get('symbols', []),
                config_path=self.config_path
            )
            self.transactions_collector.register_callback(self._on_transaction_data)
        else:
            self.transactions_collector = None
        
        self.logger.info("Data collectors initialized")
    
    def _init_ai_components(self) -> None:
        """Initialize AI components."""
        # Model trainer
        self.model_trainer = ModelTrainer(config_path=self.config_path)
        
        # Prediction engine - initialize only if a model exists
        model_path = 'models/lstm_model'
        if os.path.exists(f"{model_path}.keras"):
            self.prediction_engine = PredictionEngine(
                model_path=model_path,
                config_path=self.config_path
            )
            self.logger.info("Prediction engine initialized with existing model")
        else:
            self.prediction_engine = None
            self.logger.warning("No trained model found. Prediction engine not initialized.")
    
    def _init_order_manager(self) -> None:
        """Initialize order manager."""
        # Use the first exchange from transactions as the execution exchange
        tx_exchanges = self.config.get('data_collection', {}).get('transactions', {}).get('exchanges', [])
        if tx_exchanges and self.config.get('execution', {}).get('enabled', False):
            self.order_manager = OrderManager(
                exchange_id=tx_exchanges[0],
                config_path=self.config_path
            )
            self.logger.info(f"Order manager initialized for {tx_exchanges[0]}")
        else:
            self.order_manager = None
            self.logger.info("Order manager not initialized (execution disabled)")
    
    def _on_liquidation_data(self, data: Dict[str, Any]) -> None:
        """
        Callback for handling liquidation data.
        
        Args:
            data (Dict[str, Any]): Liquidation data
        """
        self.liquidation_data.append(data)
        
        # Store to database
        self.db_handler.store_liquidation(data)
        
        # Keep only the latest 1000 events
        if len(self.liquidation_data) > 1000:
            self.liquidation_data = self.liquidation_data[-1000:]
    
    def _on_transaction_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Callback for handling transaction data.
        
        Args:
            symbol (str): Trading symbol
            data (Dict[str, Any]): Transaction data
        """
        if symbol not in self.transaction_data:
            self.transaction_data[symbol] = []
        
        self.transaction_data[symbol].append(data)
        
        # Store to database
        self.db_handler.store_transaction(symbol, data)
        
        # Keep only the latest 1000 transactions per symbol
        if len(self.transaction_data[symbol]) > 1000:
            self.transaction_data[symbol] = self.transaction_data[symbol][-1000:]
    
    async def start(self) -> None:
        """Start the AI trading agent."""
        self.logger.info("Starting AI Trading Agent...")
        
        # Start data collectors
        data_collection_tasks = []
        
        # Start liquidation collector
        liq_config = self.config.get('data_collection', {}).get('liquidation', {})
        if self.liquidation_collector and liq_config.get('enabled', False):
            data_collection_tasks.append(
                asyncio.create_task(self.liquidation_collector.start_collection())
            )
        
        # Start funding rate collector
        fr_config = self.config.get('data_collection', {}).get('funding_rate', {})
        if self.funding_rate_collector and fr_config.get('enabled', False):
            interval = fr_config.get('interval', 3600)
            data_collection_tasks.append(
                asyncio.create_task(self.funding_rate_collector.start_collection(interval=interval))
            )
        
        # Start open interest collector
        oi_config = self.config.get('data_collection', {}).get('open_interest', {})
        if self.open_interest_collector and oi_config.get('enabled', False):
            interval = oi_config.get('interval', 300)
            data_collection_tasks.append(
                asyncio.create_task(self.open_interest_collector.start_collection(interval=interval))
            )
        
        # Start token launches collector
        tl_config = self.config.get('data_collection', {}).get('token_launches', {})
        if self.token_launches_collector and tl_config.get('enabled', False):
            interval = tl_config.get('interval', 3600)
            data_collection_tasks.append(
                asyncio.create_task(self.token_launches_collector.start_collection(interval=interval))
            )
        
        # Start transactions collector
        if self.transactions_collector:
            data_collection_tasks.append(
                asyncio.create_task(self.transactions_collector.start_collection())
            )
        
        # Set up signal processing
        self.logger.info("Setting up signal processing pipeline...")
        
        # For now, we're focusing on data collection and processing
        # Skip the model training and trade execution parts
        
        # Wait for all data collection tasks
        if data_collection_tasks:
            self.logger.info(f"Started {len(data_collection_tasks)} data collection tasks")
            await asyncio.gather(*data_collection_tasks)
        else:
            self.logger.warning("No data collection tasks started. Check configuration.")
            
            # Keep the program running
            while True:
                await asyncio.sleep(60)

async def main():
    """Main entry point for the application."""
    # Initialize AI trading agent
    agent = AITradingAgent()
    
    # Start the agent
    await agent.start()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Trading Agent")
    parser.add_argument("--config", type=str, default="config/settings.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    parser.add_argument("--backtest", action="store_true", 
                        help="Run in backtesting mode")
    parser.add_argument("--start-date", type=str, 
                        help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, 
                        help="End date for backtesting (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Run the application
    asyncio.run(main()) 