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
        self.logger.info("Starting AI Trading Agent")
        
        # Start data collectors
        data_collection_tasks = []
        
        if self.liquidation_collector:
            data_collection_tasks.append(asyncio.create_task(self.liquidation_collector.start_collection()))
        
        if self.transactions_collector:
            data_collection_tasks.append(asyncio.create_task(self.transactions_collector.start_collection()))
        
        # Start processing loop
        process_task = asyncio.create_task(self._processing_loop())
        
        # Wait for tasks - we'll only wait for the processing loop since the collectors run indefinitely
        try:
            await process_task
        except asyncio.CancelledError:
            self.logger.info("Processing loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            # Stop collectors
            if self.liquidation_collector:
                self.liquidation_collector.stop_collection()
            
            if self.transactions_collector:
                self.transactions_collector.stop_collection()
            
            # Cancel all tasks
            for task in data_collection_tasks:
                task.cancel()
    
    async def _processing_loop(self) -> None:
        """Main processing loop for the agent."""
        self.logger.info("Starting processing loop")
        
        while True:
            try:
                # Extract features from collected data
                features = await self._extract_features()
                
                if not features.empty:
                    # Generate trading signals
                    if self.prediction_engine:
                        signal = await self.generate_signals(features)
                        self.logger.info(f"Generated signal: {signal}")
                        
                        # Execute trades based on signals
                        await self.execute_trades(signal)
                    else:
                        self.logger.debug("No prediction engine available")
                
                # Periodic tasks
                # TODO: Implement periodic tasks like model retraining
                
                # Sleep to control loop frequency
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _extract_features(self) -> pd.DataFrame:
        """
        Extract features from collected data.
        
        Returns:
            pd.DataFrame: Extracted features
        """
        # Process liquidation data if available
        liquidation_features = pd.DataFrame()
        if self.liquidation_data:
            cleaned_liquidations = self.data_cleaner.clean_liquidation_data(self.liquidation_data)
            liquidation_features = self.feature_engineer.extract_liquidation_features(cleaned_liquidations)
        
        # Process funding rate data if available
        funding_features = pd.DataFrame()
        if self.funding_rate_data:
            cleaned_funding = self.data_cleaner.clean_funding_rate_data(self.funding_rate_data)
            funding_features = self.feature_engineer.extract_funding_rate_features(cleaned_funding)
        
        # Process open interest data if available
        oi_features = pd.DataFrame()
        if self.open_interest_data:
            cleaned_oi = self.data_cleaner.clean_open_interest_data(self.open_interest_data)
            oi_features = self.feature_engineer.extract_open_interest_features(cleaned_oi)
        
        # Process transaction data if available
        tx_features = {}
        for symbol, tx_data in self.transaction_data.items():
            if tx_data:
                cleaned_tx = self.data_cleaner.clean_transaction_data(tx_data)
                tx_features[symbol] = self.feature_engineer.extract_market_transaction_features(cleaned_tx)
        
        # For now, we'll return features for the first configured symbol
        # A more sophisticated implementation would handle multiple symbols
        symbols = self.config.get('data_collection', {}).get('transactions', {}).get('symbols', [])
        if symbols and symbols[0] in tx_features:
            return tx_features[symbols[0]]
        
        # If we don't have transaction features, return an empty DataFrame
        return pd.DataFrame()
    
    async def generate_signals(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals from processed features.
        
        Args:
            features (pd.DataFrame): Processed features
            
        Returns:
            Dict[str, Any]: Trading signals
        """
        if self.prediction_engine is None:
            self.logger.warning("Prediction engine not initialized")
            return {'signal': 'neutral', 'confidence': 0.0}
        
        if features.empty:
            self.logger.warning("No features available for prediction")
            return {'signal': 'neutral', 'confidence': 0.0}
        
        return self.prediction_engine.generate_signal(features)
    
    async def execute_trades(self, signal: Dict[str, Any]) -> None:
        """
        Execute trades based on generated signals.
        
        Args:
            signal (Dict[str, Any]): Trading signal
        """
        if not self.config.get('execution', {}).get('enabled', False) or self.order_manager is None:
            self.logger.debug("Trade execution disabled")
            return
        
        # Get symbol to trade
        symbols = self.config.get('data_collection', {}).get('transactions', {}).get('symbols', [])
        if not symbols:
            self.logger.warning("No symbols configured for trading")
            return
        
        symbol = symbols[0]  # Use the first symbol for now
        
        # Check if signal is actionable
        if signal.get('signal') == 'buy':
            self.logger.info(f"Executing BUY for {symbol} with confidence {signal.get('confidence')}")
            
            # Calculate position size (simplified)
            position_size = self.order_manager.calculate_position_size(symbol)
            
            # Execute the order
            order_result = self.order_manager.execute_order(
                symbol=symbol,
                side='buy',
                quantity=position_size
            )
            
            self.logger.info(f"Order execution result: {order_result}")
            
        elif signal.get('signal') == 'sell':
            self.logger.info(f"Executing SELL for {symbol} with confidence {signal.get('confidence')}")
            
            # Check if we have an open position
            if symbol in self.order_manager.open_positions:
                position = self.order_manager.open_positions[symbol]
                
                # Close the position
                order_result = self.order_manager.execute_order(
                    symbol=symbol,
                    side='sell',
                    quantity=position['quantity']
                )
                
                self.logger.info(f"Order execution result: {order_result}")
            else:
                self.logger.info(f"No open position for {symbol} to sell")

async def main():
    """Main entry point for the application."""
    # Initialize AI trading agent
    agent = AITradingAgent()
    
    # Start the agent
    await agent.start()

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main()) 