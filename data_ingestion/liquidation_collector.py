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
        self.recent_liquidations = []
        
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
        # Use the futures WebSocket endpoint for liquidations
        endpoint = "wss://fstream.binance.com/ws"
        
        async def on_message(websocket):
            # Subscribe to liquidation feed
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": ["!forceOrder@arr"],  # This subscribes to all liquidation events
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            
            # Process messages
            while self.running:
                try:
                    message = await websocket.recv()
                    
                    try:
                        data = json.loads(message)
                        
                        # Check if it's a subscription confirmation message
                        if "result" in data:
                            self.logger.info(f"Binance subscription result: {data}")
                            continue
                        
                        # Check if it's a liquidation event (forceOrder)
                        if "e" in data and data["e"] == "forceOrder":
                            # This is a single liquidation event
                            self._process_binance_liquidation(data)
                        elif "data" in data and isinstance(data["data"], dict) and data["data"].get("e") == "forceOrder":
                            # This might be a wrapped liquidation event
                            self._process_binance_liquidation(data["data"])
                        
                    except json.JSONDecodeError:
                        self.logger.error(f"Invalid JSON in message: {message}")
                    except Exception as e:
                        self.logger.error(f"Error processing liquidation message: {str(e)}")
                
                except Exception as e:
                    self.logger.error(f"Error receiving WebSocket message: {str(e)}")
                    break
        
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
    
    async def start_collection(self, interval: int = 60) -> None:
        """
        Start collecting liquidation data from all configured exchanges.
        
        Args:
            interval (int): Collection interval in seconds (used for reconnection logic)
        """
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
    
    def _convert_to_ccxt_symbol(self, symbol_raw: str) -> str:
        """
        Convert exchange symbol format to CCXT format.
        
        Args:
            symbol_raw (str): Raw symbol from the exchange (e.g., 'BTCUSDT')
            
        Returns:
            str: Symbol in CCXT format (e.g., 'BTC/USDT')
        """
        # Return if already in CCXT format
        if "/" in symbol_raw:
            return symbol_raw
        
        # Handle Binance futures symbols
        if symbol_raw.endswith("USDT"):
            return f"{symbol_raw[:-4]}/USDT"
        elif symbol_raw.endswith("BUSD"):
            return f"{symbol_raw[:-4]}/BUSD"
        elif symbol_raw.endswith("BTC"):
            return f"{symbol_raw[:-3]}/BTC"
        elif symbol_raw.endswith("ETH"):
            return f"{symbol_raw[:-3]}/ETH"
        elif symbol_raw.endswith("USD"):
            # Might be a USD-settled futures contract
            return f"{symbol_raw[:-3]}/USD" 
        
        # If we can't parse the symbol, log a warning and return as-is
        self.logger.warning(f"Could not convert symbol to CCXT format: {symbol_raw}")
        return symbol_raw 

    def _process_binance_liquidation(self, data: Dict[str, Any]) -> None:
        """
        Process a Binance liquidation event.
        
        Args:
            data (Dict[str, Any]): Liquidation event data
        """
        try:
            # Extract liquidation data
            if "o" in data:  # Structure used by Binance futures
                order_data = data["o"]
                symbol_raw = order_data.get("s", "")
                
                # Convert to CCXT symbol format (BTC/USDT)
                symbol = self._convert_to_ccxt_symbol(symbol_raw)
                
                # Create liquidation record
                liquidation = {
                    "exchange": "binance",
                    "symbol": symbol,
                    "side": "sell" if order_data.get("S") == "BUY" else "buy",  # Flip side (liquidations are forced)
                    "quantity": float(order_data.get("q", 0)),
                    "price": float(order_data.get("p", 0)),
                    "timestamp": data.get("E", int(time.time() * 1000)),
                    "id": str(order_data.get("i", "")),
                    "raw_data": data
                }
                
                # Store and notify
                self._store_liquidation(liquidation)
                
                # Call any registered callbacks
                for callback in self.callbacks:
                    callback(liquidation)
                
                self.logger.info(f"Processed Binance liquidation: {symbol} {liquidation['side']} {liquidation['quantity']} @ {liquidation['price']}")
                
        except Exception as e:
            self.logger.error(f"Error processing Binance liquidation: {str(e)}") 

    def _store_liquidation(self, liquidation: Dict[str, Any]) -> None:
        """
        Store liquidation data in memory and/or database.
        
        Args:
            liquidation (Dict[str, Any]): Liquidation data to store
        """
        # Store in memory
        self.recent_liquidations.append(liquidation)
        
        # Keep only the most recent liquidations (e.g., last 1000)
        if len(self.recent_liquidations) > 1000:
            self.recent_liquidations = self.recent_liquidations[-1000:]
        
        # If you have a database connection, store there as well
        # For example:
        # if hasattr(self, 'db_connector'):
        #    self.db_connector.insert_liquidation(liquidation)
        
        # Log a summary
        symbol = liquidation.get('symbol', 'Unknown')
        side = liquidation.get('side', 'Unknown')
        quantity = liquidation.get('quantity', 0)
        price = liquidation.get('price', 0)
        
        self.logger.debug(f"Stored liquidation: {symbol} {side} {quantity} @ {price}") 