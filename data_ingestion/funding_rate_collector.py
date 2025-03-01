"""
Module for collecting funding rate data from cryptocurrency exchanges.
"""
import time
import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from data_ingestion.exchange_connector import ExchangeConnector

class FundingRateCollector:
    """Collector for funding rate data from various exchanges."""
    
    def __init__(self, exchanges: List[str], config_path: str = 'config/settings.yaml'):
        """
        Initialize the funding rate collector.
        
        Args:
            exchanges (List[str]): List of exchange IDs to collect data from
            config_path (str): Path to configuration file
        """
        self.exchanges = exchanges
        self.config_path = config_path
        self.connectors = {}
        self.running = False
        self.callbacks = []
        self.logger = logging.getLogger("FundingRateCollector")
        
        # Initialize exchange connectors
        for exchange_id in exchanges:
            try:
                self.connectors[exchange_id] = ExchangeConnector(exchange_id, config_path)
                self.logger.info(f"Initialized connector for {exchange_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize connector for {exchange_id}: {str(e)}")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to process funding rate data.
        
        Args:
            callback (Callable): Function that takes funding rate data dict as input
        """
        self.callbacks.append(callback)
        self.logger.debug(f"Registered new callback, total callbacks: {len(self.callbacks)}")
    
    async def _collect_binance_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Collect funding rate data from Binance.
        
        Returns:
            List[Dict[str, Any]]: List of funding rate data
        """
        try:
            connector = self.connectors.get('binance')
            if not connector:
                return []
                
            # Get all futures markets
            markets = await asyncio.to_thread(connector.exchange.fetch_markets)
            futures_markets = [m for m in markets if m.get('linear') and m.get('active')]
            
            funding_rates = []
            for market in futures_markets[:50]:  # Limit to avoid rate limits
                try:
                    symbol = market['symbol']
                    # Fetch funding rate information
                    funding_info = await asyncio.to_thread(
                        connector.exchange.fetch_funding_rate, 
                        symbol
                    )
                    
                    if funding_info:
                        funding_rates.append({
                            'exchange': 'binance',
                            'symbol': symbol,
                            'funding_rate': funding_info.get('fundingRate', 0),
                            'next_funding_time': funding_info.get('nextFundingTime', 0),
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'raw_data': funding_info
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error fetching Binance funding rate for {market.get('symbol')}: {str(e)}")
            
            self.logger.info(f"Collected {len(funding_rates)} funding rates from Binance")
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Error collecting Binance funding rates: {str(e)}")
            return []
    
    async def _collect_bitmex_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Collect funding rate data from BitMEX.
        
        Returns:
            List[Dict[str, Any]]: List of funding rate data
        """
        try:
            connector = self.connectors.get('bitmex')
            if not connector:
                return []
                
            # Get all active markets
            markets = await asyncio.to_thread(connector.exchange.fetch_markets)
            active_markets = [m for m in markets if m.get('active')]
            
            funding_rates = []
            for market in active_markets[:30]:  # Limit to avoid rate limits
                try:
                    symbol = market['symbol']
                    # Fetch funding rate information
                    funding_info = await asyncio.to_thread(
                        connector.exchange.fetch_funding_rate, 
                        symbol
                    )
                    
                    if funding_info:
                        funding_rates.append({
                            'exchange': 'bitmex',
                            'symbol': symbol,
                            'funding_rate': funding_info.get('fundingRate', 0),
                            'next_funding_time': funding_info.get('nextFundingTime', 0),
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'raw_data': funding_info
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error fetching BitMEX funding rate for {market.get('symbol')}: {str(e)}")
            
            self.logger.info(f"Collected {len(funding_rates)} funding rates from BitMEX")
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Error collecting BitMEX funding rates: {str(e)}")
            return []
    
    async def _collect_deribit_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Collect funding rate data from Deribit.
        
        Returns:
            List[Dict[str, Any]]: List of funding rate data
        """
        try:
            connector = self.connectors.get('deribit')
            if not connector:
                return []
                
            # Get all active markets
            markets = await asyncio.to_thread(connector.exchange.fetch_markets)
            perpetual_markets = [m for m in markets if m.get('active') and 'PERPETUAL' in m.get('symbol', '')]
            
            funding_rates = []
            for market in perpetual_markets:
                try:
                    symbol = market['symbol']
                    # Fetch funding rate information (this may need adjustment for Deribit API)
                    funding_info = await asyncio.to_thread(
                        connector.exchange.fetch_funding_rate, 
                        symbol
                    )
                    
                    if funding_info:
                        funding_rates.append({
                            'exchange': 'deribit',
                            'symbol': symbol,
                            'funding_rate': funding_info.get('fundingRate', 0),
                            'next_funding_time': funding_info.get('nextFundingTime', 0),
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'raw_data': funding_info
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error fetching Deribit funding rate for {market.get('symbol')}: {str(e)}")
            
            self.logger.info(f"Collected {len(funding_rates)} funding rates from Deribit")
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Error collecting Deribit funding rates: {str(e)}")
            return []
    
    async def collect_funding_rates(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect funding rates from all configured exchanges.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with exchange IDs as keys and lists of funding rates as values
        """
        tasks = []
        
        # Add tasks for each exchange
        for exchange in self.exchanges:
            if exchange == 'binance':
                tasks.append(self._collect_binance_funding_rates())
            elif exchange == 'bitmex':
                tasks.append(self._collect_bitmex_funding_rates())
            elif exchange == 'deribit':
                tasks.append(self._collect_deribit_funding_rates())
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        funding_rates = {}
        exchange_idx = 0
        
        for exchange in self.exchanges:
            if exchange in ['binance', 'bitmex', 'deribit']:
                if isinstance(results[exchange_idx], list):
                    # Process the data through callbacks
                    for rate_data in results[exchange_idx]:
                        for callback in self.callbacks:
                            callback(rate_data)
                    
                    funding_rates[exchange] = results[exchange_idx]
                else:
                    self.logger.error(f"Error collecting funding rates from {exchange}: {results[exchange_idx]}")
                    funding_rates[exchange] = []
                exchange_idx += 1
        
        return funding_rates
    
    async def _collection_loop(self, interval: int) -> None:
        """
        Run the funding rate collection loop at specified intervals.
        
        Args:
            interval (int): Collection interval in seconds
        """
        while self.running:
            try:
                self.logger.info("Collecting funding rates...")
                await self.collect_funding_rates()
                
                # Wait for next collection
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in funding rate collection loop: {str(e)}")
                await asyncio.sleep(min(interval, 60))  # Wait at most 60 seconds on error
    
    async def start_collection(self, interval: int = 3600) -> None:
        """
        Start collecting funding rate data at specified intervals.
        
        Args:
            interval (int): Collection interval in seconds
        """
        self.running = True
        self.logger.info(f"Starting funding rate collection with interval {interval}s")
        await self._collection_loop(interval)
    
    def stop_collection(self) -> None:
        """Stop the funding rate collection."""
        self.running = False
        self.logger.info("Stopping funding rate collection") 