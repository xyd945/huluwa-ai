"""
Module for collecting open interest data from cryptocurrency exchanges.
"""
import time
import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from data_ingestion.exchange_connector import ExchangeConnector

class OpenInterestCollector:
    """Collector for open interest data from various exchanges."""
    
    def __init__(self, exchanges: List[str], config_path: str = 'config/settings.yaml'):
        """
        Initialize the open interest collector.
        
        Args:
            exchanges (List[str]): List of exchange IDs to collect data from
            config_path (str): Path to configuration file
        """
        self.exchanges = exchanges
        self.config_path = config_path
        self.connectors = {}
        self.running = False
        self.callbacks = []
        self.logger = logging.getLogger("OpenInterestCollector")
        
        # Initialize exchange connectors
        for exchange_id in exchanges:
            try:
                self.connectors[exchange_id] = ExchangeConnector(exchange_id, config_path)
                self.logger.info(f"Initialized connector for {exchange_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize connector for {exchange_id}: {str(e)}")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to process open interest data.
        
        Args:
            callback (Callable): Function that takes open interest data dict as input
        """
        self.callbacks.append(callback)
        self.logger.debug(f"Registered new callback, total callbacks: {len(self.callbacks)}")
    
    async def _collect_binance_open_interest(self) -> List[Dict[str, Any]]:
        """
        Collect open interest data from Binance.
        
        Returns:
            List[Dict[str, Any]]: List of open interest data
        """
        try:
            connector = self.connectors.get('binance')
            if not connector:
                return []
            
            # Get all futures markets - ensuring we get full market objects
            markets = await asyncio.to_thread(connector.exchange.fetch_markets)
            
            # Properly filter futures markets
            futures_markets = [m for m in markets if 
                               (m.get('future', False) or 
                               'future' in m.get('type', '').lower() or
                               'swap' in m.get('type', '').lower()) and
                               m.get('active', False)]
            
            # Get symbols in the correct format (BTC/USDT:USDT)
            futures_symbols = [m['symbol'] for m in futures_markets]
            
            # Debug log to verify we have proper symbols
            self.logger.info(f"Found {len(futures_symbols)} futures symbols. Examples: {futures_symbols[:3] if futures_symbols else []}")
            
            # Limit to top 10 by volume if available
            if len(futures_symbols) > 10:
                futures_symbols = futures_symbols[:10]
            
            self.logger.info(f"Fetching open interest for {len(futures_symbols)} Binance futures symbols")
            
            open_interest_data = []
            
            # Fetch open interest for each symbol
            for symbol in futures_symbols:
                try:
                    open_interest_info = await asyncio.to_thread(connector.exchange.fetch_open_interest, symbol)
                    if open_interest_info:
                        open_interest_data.append({
                            'exchange': 'binance',
                            'symbol': symbol,
                            'open_interest': open_interest_info.get('openInterestAmount', 0),
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'raw_data': open_interest_info
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error fetching Binance open interest for {symbol}: {str(e)}")
            
            self.logger.info(f"Collected open interest for {len(open_interest_data)} Binance symbols")
            return open_interest_data
            
        except Exception as e:
            self.logger.error(f"Error collecting Binance open interest: {str(e)}")
            return []
    
    async def _collect_bitmex_open_interest(self) -> List[Dict[str, Any]]:
        """
        Collect open interest data from BitMEX.
        
        Returns:
            List[Dict[str, Any]]: List of open interest data
        """
        try:
            connector = self.connectors.get('bitmex')
            if not connector:
                self.logger.error("BitMEX connector not initialized")
                return []
            
            # Fetch list of futures symbols
            markets = connector.exchange.fetch_markets()
            future_symbols = [market['symbol'] for market in markets if 
                             'swap' in market.get('type', '').lower()]
            
            # Fetch open interest for each symbol
            result = []
            for symbol in future_symbols:
                try:
                    # BitMEX requires specific API endpoints for open interest
                    endpoint = f"instrument?symbol={symbol.split(':')[0]}&count=1"
                    response = connector.exchange.request('GET', endpoint)
                    
                    if response and isinstance(response, list) and len(response) > 0:
                        instrument_data = response[0]
                        
                        data = {
                            'exchange': 'bitmex',
                            'symbol': symbol,
                            'open_interest': float(instrument_data.get('openInterest', 0)),
                            'open_interest_usd': float(instrument_data.get('openValue', 0)) / 100000000,  # Convert from Satoshis
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'raw_data': instrument_data
                        }
                        
                        result.append(data)
                        
                        # Process callbacks
                        for callback in self.callbacks:
                            callback(data)
                    
                    # Avoid rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching BitMEX open interest for {symbol}: {str(e)}")
            
            self.logger.info(f"Collected open interest for {len(result)} BitMEX symbols")
            return result
            
        except Exception as e:
            self.logger.error(f"Error collecting BitMEX open interest: {str(e)}")
            return []
    
    async def collect_open_interest(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect open interest data from all configured exchanges.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of open interest data by exchange
        """
        tasks = []
        
        # Add tasks for each exchange
        if 'binance' in self.exchanges:
            tasks.append(self._collect_binance_open_interest())
        
        if 'bitmex' in self.exchanges:
            tasks.append(self._collect_bitmex_open_interest())
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        open_interest_data = {}
        exchange_idx = 0
        
        for exchange in self.exchanges:
            if exchange in ['binance', 'bitmex']:
                if isinstance(results[exchange_idx], list):
                    open_interest_data[exchange] = results[exchange_idx]
                else:
                    self.logger.error(f"Error collecting open interest from {exchange}: {results[exchange_idx]}")
                    open_interest_data[exchange] = []
                exchange_idx += 1
        
        return open_interest_data
    
    async def _collection_loop(self, interval: int) -> None:
        """
        Run the open interest collection loop at specified intervals.
        
        Args:
            interval (int): Collection interval in seconds
        """
        while self.running:
            try:
                self.logger.info("Collecting open interest data...")
                await self.collect_open_interest()
                
                # Wait for next collection
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in open interest collection loop: {str(e)}")
                await asyncio.sleep(min(interval, 60))  # Wait at most 60 seconds on error
    
    async def start_collection(self, interval: int = 300) -> None:
        """
        Start collecting open interest data at specified intervals.
        
        Args:
            interval (int): Collection interval in seconds
        """
        self.running = True
        self.logger.info(f"Starting open interest collection with interval {interval}s")
        await self._collection_loop(interval)
    
    def stop_collection(self) -> None:
        """Stop the open interest collection."""
        self.running = False
        self.logger.info("Stopping open interest collection") 