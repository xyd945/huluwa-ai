"""
Module for collecting real-time market transaction data.
"""
import time
import json
import asyncio
import websockets
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from data_ingestion.exchange_connector import ExchangeConnector

class TransactionsCollector:
    """Collector for market transactions from various exchanges."""
    
    def __init__(self, symbols: List[str], exchanges: List[str], config_path: str = 'config/settings.yaml'):
        """
        Initialize the transactions collector.
        
        Args:
            symbols (List[str]): List of trading pairs to monitor
            exchanges (List[str]): List of exchange IDs to collect data from
            config_path (str): Path to configuration file
        """
        self.symbols = symbols
        self.exchanges = exchanges
        self.config_path = config_path
        self.connectors = {}
        self.websocket_connections = {}
        self.running = False
        self.callbacks = []
        self.logger = logging.getLogger("TransactionsCollector")
        
        # Initialize exchange connectors
        for exchange_id in exchanges:
            try:
                self.connectors[exchange_id] = ExchangeConnector(exchange_id, config_path)
                self.logger.info(f"Initialized connector for {exchange_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize connector for {exchange_id}: {str(e)}")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to process transaction data.
        
        Args:
            callback (Callable): Function that takes transaction data dict as input
        """
        self.callbacks.append(callback)
        self.logger.debug(f"Registered new callback, total callbacks: {len(self.callbacks)}")
    
    async def _binance_trades_handler(self, symbols: List[str]) -> None:
        """
        Handle Binance real-time trade data via WebSocket.
        
        Args:
            symbols (List[str]): List of symbols to monitor
        """
        # Convert symbols to Binance format (lowercase, remove /)
        formatted_symbols = [s.lower().replace('/', '') for s in symbols]
        
        # Create stream names for each symbol
        streams = [f"{symbol}@trade" for symbol in formatted_symbols]
        endpoint = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        async def on_message(websocket):
            async for message in websocket:
                data = json.loads(message)
                if 'data' in data and 'e' in data['data'] and data['data']['e'] == 'trade':
                    trade_data = data['data']
                    
                    # Extract symbol in standard format
                    raw_symbol = trade_data['s']
                    standard_symbol = f"{raw_symbol[:-4]}/{raw_symbol[-4:]}" if raw_symbol.endswith(('USDT', 'USDC')) else raw_symbol
                    
                    transaction_data = {
                        'exchange': 'binance',
                        'symbol': standard_symbol,
                        'price': float(trade_data['p']),
                        'amount': float(trade_data['q']),
                        'side': 'buy' if trade_data['m'] else 'sell',  # m = is_buyer_maker
                        'timestamp': trade_data['T'],
                        'trade_id': trade_data['t'],
                        'raw_data': trade_data
                    }
                    
                    for callback in self.callbacks:
                        callback(transaction_data)
        
        while self.running:
            try:
                self.logger.info(f"Connecting to Binance trades stream for {len(symbols)} symbols...")
                async with websockets.connect(endpoint) as websocket:
                    self.websocket_connections['binance'] = websocket
                    await on_message(websocket)
            except Exception as e:
                self.logger.error(f"Binance WebSocket error: {str(e)}")
                await asyncio.sleep(5)  # Reconnect after delay
    
    async def start_collection(self) -> None:
        """Start collecting real-time market transaction data."""
        self.running = True
        tasks = []
        
        # Start collectors for each exchange
        for exchange in self.exchanges:
            if exchange == 'binance':
                # Filter symbols for this exchange
                exchange_symbols = self.symbols
                if exchange_symbols:
                    tasks.append(asyncio.create_task(self._binance_trades_handler(exchange_symbols)))
            # Add more exchanges as needed
        
        if tasks:
            self.logger.info(f"Started transaction collectors for: {', '.join(self.exchanges)}")
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
        else:
            self.logger.warning("No transaction collectors started. Check symbols and exchange configuration.")
    
    def stop_collection(self) -> None:
        """Stop all running collectors."""
        self.running = False
        self.logger.info("Stopping all transaction collectors...")
        
        # Close all websocket connections
        for connection in self.websocket_connections.values():
            asyncio.create_task(connection.close())
        
        self.websocket_connections = {} 