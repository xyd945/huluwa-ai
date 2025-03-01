"""
Module for managing order execution on cryptocurrency exchanges.
"""
import logging
import yaml
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from data_ingestion.exchange_connector import ExchangeConnector

class OrderManager:
    """Manages order execution and trade lifecycle."""
    
    def __init__(self, exchange_id: str, config_path: str = 'config/settings.yaml'):
        """
        Initialize the order manager.
        
        Args:
            exchange_id (str): ID of the exchange to use
            config_path (str): Path to configuration file
        """
        self.exchange_id = exchange_id
        self.logger = logging.getLogger(f"OrderManager.{exchange_id}")
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.execution_config = config.get('execution', {})
                
                # Set execution parameters
                self.enabled = self.execution_config.get('enabled', False)
                self.max_position_size = self.execution_config.get('max_position_size', 1000)
                self.stop_loss_pct = self.execution_config.get('stop_loss_pct', 2.0)
                self.take_profit_pct = self.execution_config.get('take_profit_pct', 5.0)
                self.max_open_positions = self.execution_config.get('max_open_positions', 3)
                
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            self.enabled = False
            self.max_position_size = 1000
            self.stop_loss_pct = 2.0
            self.take_profit_pct = 5.0
            self.max_open_positions = 3
        
        # Initialize exchange connector
        try:
            self.connector = ExchangeConnector(exchange_id, config_path)
            self.logger.info(f"Initialized connector for {exchange_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize connector: {str(e)}")
            self.connector = None
        
        # Track open positions
        self.open_positions = {}
    
    def is_enabled(self) -> bool:
        """
        Check if trading is enabled.
        
        Returns:
            bool: True if trading is enabled, False otherwise
        """
        return self.enabled and self.connector is not None
    
    def execute_signal(self, signal: Dict[str, Any], symbol: str, 
                      position_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a trading signal by placing an order.
        
        Args:
            signal (Dict[str, Any]): Trading signal with action and confidence
            symbol (str): Trading pair symbol
            position_size (Optional[float]): Size of position in USD (uses max_position_size * confidence if None)
            
        Returns:
            Dict[str, Any]: Order result info
        """
        if not self.is_enabled():
            self.logger.warning("Trading is not enabled, skipping order execution")
            return {'status': 'skipped', 'reason': 'trading_disabled'}
        
        # Extract signal details
        action = signal.get('signal', 'neutral')
        confidence = signal.get('confidence', 0.0)
        
        # Check if signal is actionable
        if action == 'neutral':
            return {'status': 'skipped', 'reason': 'neutral_signal'}
        
        try:
            # Get current market price
            ticker = self.connector.fetch_ticker(symbol)
            if not ticker or 'last' not in ticker:
                self.logger.error(f"Could not get price for {symbol}")
                return {'status': 'error', 'reason': 'price_unavailable'}
            
            current_price = ticker['last']
            
            # Calculate position size based on confidence if not provided
            if position_size is None:
                position_size = self.max_position_size * confidence
            
            # Calculate quantity in base currency
            quantity = position_size / current_price
            
            # Check if we've reached max open positions
            if len(self.open_positions) >= self.max_open_positions and action == 'buy':
                self.logger.warning(f"Max open positions reached ({self.max_open_positions}), skipping buy order")
                return {'status': 'skipped', 'reason': 'max_positions_reached'}
            
            # Execute order
            order_type = 'market'  # Using market orders for simplicity
            order = self.connector.create_order(
                symbol=symbol,
                order_type=order_type,
                side=action,
                amount=quantity
            )
            
            # Record the position
            if action == 'buy':
                self.open_positions[symbol] = {
                    'entry_price': current_price,
                    'quantity': quantity,
                    'entry_time': datetime.now(),
                    'stop_loss': current_price * (1 - self.stop_loss_pct / 100),
                    'take_profit': current_price * (1 + self.take_profit_pct / 100)
                }
            elif action == 'sell' and symbol in self.open_positions:
                del self.open_positions[symbol]
            
            self.logger.info(f"Executed {action} order for {quantity} {symbol} at {current_price}")
            return {
                'status': 'executed',
                'order': order,
                'price': current_price,
                'quantity': quantity,
                'action': action,
                'timestamp': datetime.now().timestamp() * 1000
            }
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return {'status': 'error', 'reason': str(e)}
    
    def check_stop_loss_take_profit(self, symbol: str) -> Dict[str, Any]:
        """
        Check if stop loss or take profit levels have been reached for a position.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict[str, Any]: Result of check and any actions taken
        """
        if not self.is_enabled() or symbol not in self.open_positions:
            return {'status': 'no_position'}
        
        try:
            # Get current market price
            ticker = self.connector.fetch_ticker(symbol)
            if not ticker or 'last' not in ticker:
                return {'status': 'error', 'reason': 'price_unavailable'}
            
            current_price = ticker['last']
            position = self.open_positions[symbol]
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                # Execute sell order
                order = self.connector.create_order(
                    symbol=symbol,
                    order_type='market',
                    side='sell',
                    amount=position['quantity']
                )
                
                # Remove position
                del self.open_positions[symbol]
                
                self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                return {
                    'status': 'stop_loss_triggered',
                    'order': order,
                    'price': current_price,
                    'timestamp': datetime.now().timestamp() * 1000
                }
            
            # Check take profit
            if current_price >= position['take_profit']:
                # Execute sell order
                order = self.connector.create_order(
                    symbol=symbol,
                    order_type='market',
                    side='sell',
                    amount=position['quantity']
                )
                
                # Remove position
                del self.open_positions[symbol]
                
                self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                return {
                    'status': 'take_profit_triggered',
                    'order': order,
                    'price': current_price,
                    'timestamp': datetime.now().timestamp() * 1000
                }
            
            # No triggers hit
            return {
                'status': 'monitoring',
                'current_price': current_price,
                'entry_price': position['entry_price'],
                'pnl_pct': (current_price - position['entry_price']) / position['entry_price'] * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error checking SL/TP: {str(e)}")
            return {'status': 'error', 'reason': str(e)}
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance.
        
        Returns:
            Dict[str, Any]: Account balance information
        """
        if not self.is_enabled():
            return {'status': 'disabled'}
        
        try:
            return self.connector.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            return {'status': 'error', 'reason': str(e)} 