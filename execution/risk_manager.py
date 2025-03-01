"""
Module for managing trading risk and position sizing.
"""
import logging
import yaml
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class RiskManager:
    """Manages risk controls and position sizing for trades."""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """
        Initialize the risk manager.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger("RiskManager")
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.execution_config = config.get('execution', {})
                
                # Set risk parameters
                self.max_position_size = self.execution_config.get('max_position_size', 1000)
                self.stop_loss_pct = self.execution_config.get('stop_loss_pct', 2.0)
                self.take_profit_pct = self.execution_config.get('take_profit_pct', 5.0)
                self.max_open_positions = self.execution_config.get('max_open_positions', 3)
                
                # Additional risk parameters
                self.max_daily_loss_pct = self.execution_config.get('max_daily_loss_pct', 5.0)
                self.position_sizing_model = self.execution_config.get('position_sizing_model', 'fixed')
                self.risk_per_trade_pct = self.execution_config.get('risk_per_trade_pct', 1.0)
                
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            # Default values
            self.max_position_size = 1000
            self.stop_loss_pct = 2.0
            self.take_profit_pct = 5.0
            self.max_open_positions = 3
            self.max_daily_loss_pct = 5.0
            self.position_sizing_model = 'fixed'
            self.risk_per_trade_pct = 1.0
        
        # Track current positions and performance
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.start_balance = 0.0
    
    def set_account_balance(self, balance: float) -> None:
        """
        Set the current account balance for risk calculations.
        
        Args:
            balance (float): Current account balance in USD
        """
        self.start_balance = balance
        self.logger.info(f"Set account balance to ${balance:.2f}")
    
    def calculate_position_size(self, symbol: str, price: float, signal_confidence: float = 0.5) -> float:
        """
        Calculate position size based on risk parameters and current market conditions.
        
        Args:
            symbol (str): Trading symbol
            price (float): Current market price
            signal_confidence (float): Confidence level of the trading signal (0.0-1.0)
            
        Returns:
            float: Position size in base currency units
        """
        if len(self.open_positions) >= self.max_open_positions:
            self.logger.warning(f"Maximum number of open positions reached ({self.max_open_positions})")
            return 0.0
        
        # Base position sizing on the model
        if self.position_sizing_model == 'fixed':
            # Simple fixed position size
            position_size_usd = self.max_position_size
            
        elif self.position_sizing_model == 'risk_percentage':
            # Risk a percentage of the balance per trade
            risk_amount = self.start_balance * (self.risk_per_trade_pct / 100)
            position_size_usd = risk_amount / (self.stop_loss_pct / 100)
            
        elif self.position_sizing_model == 'kelly':
            # Simplified Kelly criterion
            # For Kelly, we need win rate and win/loss ratio from historical data
            # Using a simplified approach with signal confidence
            win_rate = 0.5 + (signal_confidence - 0.5) * 0.5  # Scale confidence to win rate
            win_loss_ratio = self.take_profit_pct / self.stop_loss_pct
            
            kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
            kelly_fraction = max(0, min(kelly_fraction, 0.2))  # Limit to 20% of capital
            
            position_size_usd = self.start_balance * kelly_fraction
            
        else:
            # Default to fixed size
            position_size_usd = self.max_position_size
        
        # Cap to max position size
        position_size_usd = min(position_size_usd, self.max_position_size)
        
        # Convert to base currency units
        position_size = position_size_usd / price
        
        self.logger.info(f"Calculated position size for {symbol}: {position_size:.6f} units (${position_size_usd:.2f})")
        return position_size
    
    def validate_trade(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """
        Validate a potential trade against risk parameters.
        
        Args:
            symbol (str): Trading symbol
            side (str): Trade side (buy/sell)
            quantity (float): Trade quantity
            price (float): Trade price
            
        Returns:
            Dict[str, Any]: Validation result with status and reason
        """
        # Check if we're within daily loss limit
        if self.daily_pnl < -(self.start_balance * (self.max_daily_loss_pct / 100)):
            return {
                'valid': False,
                'reason': f'Daily loss limit of {self.max_daily_loss_pct}% reached'
            }
        
        # Check max open positions
        if side == 'buy' and len(self.open_positions) >= self.max_open_positions:
            return {
                'valid': False,
                'reason': f'Maximum number of open positions reached ({self.max_open_positions})'
            }
        
        # Check position size
        trade_value = quantity * price
        if trade_value > self.max_position_size:
            return {
                'valid': False,
                'reason': f'Position size ${trade_value:.2f} exceeds maximum ${self.max_position_size:.2f}'
            }
        
        # All checks passed
        return {
            'valid': True,
            'reason': 'Trade validated'
        }
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record a completed trade for risk tracking.
        
        Args:
            trade (Dict[str, Any]): Trade details
        """
        self.daily_trades.append(trade)
        
        # Update P&L
        if trade.get('realized_pnl'):
            self.daily_pnl += trade['realized_pnl']
            
        # Update open positions
        symbol = trade.get('symbol')
        side = trade.get('side')
        
        if side == 'buy':
            self.open_positions[symbol] = {
                'entry_price': trade.get('price', 0),
                'quantity': trade.get('quantity', 0),
                'timestamp': trade.get('timestamp', datetime.now().timestamp() * 1000)
            }
        elif side == 'sell' and symbol in self.open_positions:
            # Position closed
            del self.open_positions[symbol]
        
        self.logger.info(f"Recorded trade: {symbol} {side} at ${trade.get('price', 0):.2f}")
        self.logger.info(f"Daily P&L: ${self.daily_pnl:.2f}, Open positions: {len(self.open_positions)}")
    
    def reset_daily_metrics(self) -> None:
        """Reset daily performance metrics."""
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.logger.info("Reset daily risk metrics") 