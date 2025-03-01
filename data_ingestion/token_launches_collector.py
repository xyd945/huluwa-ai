"""
Module for collecting information about new token launches.
"""
import time
import asyncio
import logging
import requests
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd

class TokenLaunchesCollector:
    """Collector for new token launches from various sources."""
    
    def __init__(self, sources: List[str], config_path: str = 'config/settings.yaml'):
        """
        Initialize the token launches collector.
        
        Args:
            sources (List[str]): List of sources to collect data from
            config_path (str): Path to configuration file
        """
        self.sources = sources
        self.config_path = config_path
        self.running = False
        self.callbacks = []
        self.logger = logging.getLogger("TokenLaunchesCollector")
        
        # CoinGecko API settings
        self.coingecko_api_url = "https://api.coingecko.com/api/v3"
        
        # Binance API settings
        self.binance_announcements_url = "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query"
        
        # Initialize tracked tokens
        self.tracked_tokens = set()
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to process token launch data.
        
        Args:
            callback (Callable): Function that takes token launch data dict as input
        """
        self.callbacks.append(callback)
        self.logger.debug(f"Registered new callback, total callbacks: {len(self.callbacks)}")
    
    async def _collect_coingecko_launches(self) -> List[Dict[str, Any]]:
        """
        Collect new token launches from CoinGecko.
        
        Returns:
            List[Dict[str, Any]]: List of token launch data
        """
        try:
            # Get recently added coins (past 14 days)
            url = f"{self.coingecko_api_url}/coins/list?include_platform=false"
            response = requests.get(url)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching CoinGecko tokens: {response.status_code}")
                return []
            
            all_tokens = response.json()
            
            # Get more details for potential new tokens
            new_tokens = []
            for token in all_tokens[:50]:  # Limit to avoid rate limits
                token_id = token.get('id')
                
                # Skip if already tracked
                if token_id in self.tracked_tokens:
                    continue
                
                # Get detailed info
                detail_url = f"{self.coingecko_api_url}/coins/{token_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
                try:
                    detail_response = requests.get(detail_url)
                    
                    if detail_response.status_code == 200:
                        token_detail = detail_response.json()
                        
                        # Check if token was added in last 14 days
                        genesis_date = token_detail.get('genesis_date')
                        
                        if genesis_date:
                            genesis_datetime = datetime.strptime(genesis_date, "%Y-%m-%d")
                            if (datetime.now() - genesis_datetime).days <= 14:
                                # This is a new token
                                launch_data = {
                                    'source': 'coingecko',
                                    'token_id': token_id,
                                    'name': token_detail.get('name', ''),
                                    'symbol': token_detail.get('symbol', '').upper(),
                                    'launch_date': genesis_date,
                                    'current_price_usd': token_detail.get('market_data', {}).get('current_price', {}).get('usd', 0),
                                    'market_cap_usd': token_detail.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                                    'timestamp': int(datetime.now().timestamp() * 1000),
                                    'raw_data': token_detail
                                }
                                
                                # Add to tracked tokens
                                self.tracked_tokens.add(token_id)
                                
                                # Add to results
                                new_tokens.append(launch_data)
                                
                                # Process callbacks
                                for callback in self.callbacks:
                                    callback(launch_data)
                    
                    # Avoid rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching details for {token_id}: {str(e)}")
            
            self.logger.info(f"Found {len(new_tokens)} new token launches from CoinGecko")
            return new_tokens
            
        except Exception as e:
            self.logger.error(f"Error collecting token launches from CoinGecko: {str(e)}")
            return []
    
    async def _collect_binance_announcements(self) -> List[Dict[str, Any]]:
        """
        Collect new token launches from Binance announcements.
        
        Returns:
            List[Dict[str, Any]]: List of token launch data
        """
        try:
            # Query parameters for new listings
            payload = {
                "catalogId": "48",  # New Listings category
                "pageNo": 1,
                "pageSize": 20
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.binance_announcements_url, 
                data=json.dumps(payload),
                headers=headers
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching Binance announcements: {response.status_code}")
                return []
            
            announcements = response.json().get('data', {}).get('articles', [])
            
            # Filter for token listings
            launch_announcements = []
            for announcement in announcements:
                title = announcement.get('title', '').lower()
                
                # Check if it's a new listing announcement
                if 'binance will list' in title or 'binance lists' in title:
                    # Extract token details
                    code = announcement.get('code', '')
                    
                    # Skip if already tracked
                    if code in self.tracked_tokens:
                        continue
                    
                    # Extract token symbol from title
                    symbols = []
                    for word in title.split():
                        # Clean the word
                        word = word.strip('(),.!?')
                        # Check if it looks like a token symbol (all caps, 2-10 chars)
                        if word.isupper() and 2 <= len(word) <= 10:
                            symbols.append(word)
                    
                    token_symbol = symbols[0] if symbols else ""
                    
                    launch_data = {
                        'source': 'binance_announcements',
                        'token_id': code,
                        'name': announcement.get('title', ''),
                        'symbol': token_symbol,
                        'launch_date': datetime.fromtimestamp(announcement.get('releaseDate', 0)/1000).strftime('%Y-%m-%d'),
                        'announcement_url': f"https://www.binance.com/en/support/announcement/{code}",
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'raw_data': announcement
                    }
                    
                    # Add to tracked tokens
                    self.tracked_tokens.add(code)
                    
                    # Add to results
                    launch_announcements.append(launch_data)
                    
                    # Process callbacks
                    for callback in self.callbacks:
                        callback(launch_data)
            
            self.logger.info(f"Found {len(launch_announcements)} new token launches from Binance announcements")
            return launch_announcements
            
        except Exception as e:
            self.logger.error(f"Error collecting token launches from Binance: {str(e)}")
            return []
    
    async def collect_token_launches(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect token launch data from all configured sources.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of token launch data by source
        """
        tasks = []
        
        # Add tasks for each source
        if 'coingecko' in self.sources:
            tasks.append(self._collect_coingecko_launches())
        
        if 'binance_announcements' in self.sources:
            tasks.append(self._collect_binance_announcements())
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        launch_data = {}
        source_idx = 0
        
        for source in self.sources:
            if source in ['coingecko', 'binance_announcements']:
                if isinstance(results[source_idx], list):
                    launch_data[source] = results[source_idx]
                else:
                    self.logger.error(f"Error collecting token launches from {source}: {results[source_idx]}")
                    launch_data[source] = []
                source_idx += 1
        
        # Flatten the results and store them
        all_launches = []
        for source_launches in launch_data.values():
            all_launches.extend(source_launches)
        
        if all_launches:
            self._store_token_launches(all_launches)
        
        return launch_data
    
    async def _collection_loop(self, interval: int) -> None:
        """
        Run the token launch collection loop at specified intervals.
        
        Args:
            interval (int): Collection interval in seconds
        """
        while self.running:
            try:
                self.logger.info("Collecting token launch data...")
                await self.collect_token_launches()
                
                # Wait for next collection
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in token launch collection loop: {str(e)}")
                await asyncio.sleep(min(interval, 60))  # Wait at most 60 seconds on error
    
    async def start_collection(self, interval: int = 3600) -> None:
        """
        Start collecting token launch data at specified intervals.
        
        Args:
            interval (int): Collection interval in seconds
        """
        self.running = True
        self.logger.info(f"Starting token launch collection with interval {interval}s")
        await self._collection_loop(interval)
    
    def stop_collection(self) -> None:
        """Stop the token launch collection."""
        self.running = False
        self.logger.info("Stopping token launch collection")
    
    def _store_token_launches(self, token_launches: List[Dict[str, Any]]) -> None:
        """
        Store token launch data in the database.
        
        Args:
            token_launches (List[Dict[str, Any]]): Token launch data to store
        """
        try:
            # If you have a database handler
            if hasattr(self, 'db_handler'):
                for item in token_launches:
                    self.db_handler.store_token_launch(item)
            
            self.logger.info(f"Stored {len(token_launches)} token launch records in database")
        except Exception as e:
            self.logger.error(f"Error storing token launch data: {str(e)}") 