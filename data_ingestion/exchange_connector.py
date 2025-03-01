"""
Base exchange connector using CCXT to interact with cryptocurrency exchanges.
"""
import os
import yaml
import ccxt
import logging
from typing import Dict, List, Optional, Any, Union

class ExchangeConnector:
    """Base class for connecting to cryptocurrency exchanges using CCXT library."""
    
    def __init__(self, exchange_id: str, config_path: str = 'config/settings.yaml'):
        """
        Initialize exchange connector with configured API credentials.
        
        Args:
            exchange_id (str): ID of the exchange (e.g., 'binance', 'bitmex')
            config_path (str): Path to the configuration file
        """
        self.exchange_id = exchange_id
        self.logger = logging.getLogger(f"ExchangeConnector.{exchange_id}")
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            exchange_config = config.get('exchanges', {}).get(exchange_id, {})
            if not exchange_config:
                self.logger.warning(f"No configuration found for exchange {exchange_id}")
            
            # Initialize CCXT exchange
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': exchange_config.get('api_key', ''),
                'secret': exchange_config.get('api_secret', ''),
                'enableRateLimit': True,
                'options': {
                    'testnet': exchange_config.get('testnet', True)
                }
            })
            
            self.logger.info(f"Successfully initialized {exchange_id} connector")
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize {exchange_id} connector: {str(e)}")
            raise
    
    def fetch_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch all available markets from the exchange.
        
        Returns:
            List[Dict[str, Any]]: List of markets
        """
        try:
            return self.exchange.fetch_markets()
        except Exception as e:
            self.logger.error(f"Error fetching markets: {str(e)}")
            return []
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch ticker data for a symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dict[str, Any]: Ticker data
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            return {}
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                   since: Optional[int] = None, limit: Optional[int] = None) -> List[List[float]]:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe (e.g., '1m', '1h', '1d')
            since (Optional[int]): Timestamp in ms for start time
            limit (Optional[int]): Maximum number of candles to fetch
            
        Returns:
            List[List[float]]: OHLCV data
        """
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return []
    
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch order book for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            limit (Optional[int]): Limit the number of orders returned
            
        Returns:
            Dict[str, Any]: Order book data
        """
        try:
            return self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            return {}
    
    def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create an order on the exchange.
        
        Args:
            symbol (str): Trading pair symbol
            order_type (str): Type of order ('limit', 'market')
            side (str): 'buy' or 'sell'
            amount (float): Amount of base currency to trade
            price (Optional[float]): Price for limit orders
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            return self.exchange.create_order(symbol, order_type, side, amount, price)
        except Exception as e:
            self.logger.error(f"Error creating {order_type} {side} order for {symbol}: {str(e)}")
            raise
    
    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
            Dict[str, Any]: Account balance information
        """
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            return {}
    
    def get_websocket_endpoint(self) -> str:
        """
        Get WebSocket endpoint for the exchange.
        
        Returns:
            str: WebSocket endpoint URL
        """
        # This is a basic implementation, as CCXT doesn't directly handle WebSockets
        # Many exchanges use different WebSocket URLs than their REST API endpoints
        ws_endpoints = {
            'binance': 'wss://stream.binance.com:9443/ws',
            'bitmex': 'wss://www.bitmex.com/realtime',
            'deribit': 'wss://www.deribit.com/ws/api/v2',
            # Add more exchanges as needed
        }
        
        return ws_endpoints.get(self.exchange_id, '') 