"""
Module for collecting liquidation data from cryptocurrency exchanges.
"""
import time
import json
import asyncio
import websockets
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from data_ingestion.exchange_connector import ExchangeConnector

class LiquidationCollector:
    """Collector for liquidation data from various exchanges."""
    
    def __init__(self, exchanges: List[str], config_path: str = 'config/settings.yaml'):
        """
        Initialize the liquidation collector.
        
        Args:
            exchanges (List[str]): List of exchange IDs to collect data from
            config_path (str): Path to configuration file
        """
        self.exchanges = exchanges
        self.config_path = config_path
        self.connectors = {}
        self.websocket_connections = {}
        self.running = False
        self.callbacks = []
        self.logger = logging.getLogger("LiquidationCollector")
        
        # Initialize exchange connectors
        for exchange_id in exchanges:
            try:
                self.connectors[exchange_id] = ExchangeConnector(exchange_id, config_path)
                self.logger.info(f"Initialized connector for {exchange_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize connector for {exchange_id}: {str(e)}")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to process liquidation data.
        
        Args:
            callback (Callable): Function that takes liquidation data dict as input
        """
        self.callbacks.append(callback)
        self.logger.debug(f"Registered new callback, total callbacks: {len(self.callbacks)}")
    
    async def _binance_liquidation_handler(self) -> None:
        """Handle Binance liquidation data via WebSocket."""
        endpoint = "wss://fstream.binance.com/ws/!forceOrder@arr"
        
        async def on_message(websocket):
            async for message in websocket:
                data = json.loads(message)
                if 'e' in data and data['e'] == 'forceOrder':
                    liquidation_data = {
                        'exchange': 'binance',
                        'symbol': data['o']['s'],
                        'side': data['o']['S'].lower(),
                        'quantity': float(data['o']['q']),
                        'price': float(data['o']['p']),
                        'timestamp': data['E'],
                        'order_type': data['o']['o'],
                        'raw_data': data
                    }
                    
                    for callback in self.callbacks:
                        callback(liquidation_data)
        
        while self.running:
            try:
                self.logger.info("Connecting to Binance liquidation stream...")
                async with websockets.connect(endpoint) as websocket:
                    self.websocket_connections['binance'] = websocket
                    await on_message(websocket)
            except Exception as e:
                self.logger.error(f"Binance WebSocket error: {str(e)}")
                await asyncio.sleep(5)  # Reconnect after delay
    
    async def _bitmex_liquidation_handler(self) -> None:
        """Handle BitMEX liquidation data via WebSocket."""
        endpoint = "wss://www.bitmex.com/realtime"
        
        async def on_message(websocket):
            # Subscribe to liquidation feed
            await websocket.send(json.dumps({
                "op": "subscribe",
                "args": ["liquidation"]
            }))
            
            async for message in websocket:
                data = json.loads(message)
                if 'table' in data and data['table'] == 'liquidation' and 'data' in data:
                    for item in data['data']:
                        liquidation_data = {
                            'exchange': 'bitmex',
                            'symbol': item['symbol'],
                            'side': 'sell' if item['side'] == 'Buy' else 'buy',  # BitMEX side is from taker perspective
                            'quantity': item['leavesQty'],
                            'price': item['price'],
                            'timestamp': int(datetime.now().timestamp() * 1000),  # BitMEX doesn't include timestamp
                            'order_type': item['ordType'],
                            'raw_data': item
                        }
                        
                        for callback in self.callbacks:
                            callback(liquidation_data)
        
        while self.running:
            try:
                self.logger.info("Connecting to BitMEX liquidation stream...")
                async with websockets.connect(endpoint) as websocket:
                    self.websocket_connections['bitmex'] = websocket
                    await on_message(websocket)
            except Exception as e:
                self.logger.error(f"BitMEX WebSocket error: {str(e)}")
                await asyncio.sleep(5)  # Reconnect after delay
    
    async def start_collection(self) -> None:
        """Start collecting liquidation data from all configured exchanges."""
        self.running = True
        tasks = []
        
        # Start collectors for each exchange
        for exchange in self.exchanges:
            if exchange == 'binance':
                tasks.append(asyncio.create_task(self._binance_liquidation_handler()))
            elif exchange == 'bitmex':
                tasks.append(asyncio.create_task(self._bitmex_liquidation_handler()))
            # Add more exchanges as needed
        
        self.logger.info(f"Started liquidation collectors for: {', '.join(self.exchanges)}")
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
    
    def stop_collection(self) -> None:
        """Stop all running collectors."""
        self.running = False
        self.logger.info("Stopping all liquidation collectors...")
        
        # Close all websocket connections
        for connection in self.websocket_connections.values():
            asyncio.create_task(connection.close())
        
        self.websocket_connections = {} 