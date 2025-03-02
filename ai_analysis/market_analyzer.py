"""
Module for analyzing market data using LLM insights.
"""
import time
import asyncio
import logging
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from utils.db_handler import DatabaseHandler
from ai_analysis.llm_connector import get_llm_connector

class MarketAnalyzer:
    """Analyzes market data using LLMs to generate trading insights."""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """
        Initialize the market analyzer.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger("MarketAnalyzer")
        self.config_path = config_path
        self.db_handler = None
        self.llm_connector = None
        self.running = False
        self.last_analysis_time = 0
        
        # Initialize components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize database and LLM connector components."""
        try:
            import yaml
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Initialize database handler
            self.db_handler = DatabaseHandler(config.get('database', {}))
            
            # Initialize LLM connector based on config
            provider = config.get('ai_analysis', {}).get('llm', {}).get('provider', 'openai')
            self.llm_connector = get_llm_connector(provider, self.config_path)
            
            self.logger.info(f"Initialized MarketAnalyzer with {provider} LLM")
        
        except Exception as e:
            self.logger.error(f"Error initializing MarketAnalyzer: {str(e)}")
    
    async def fetch_last_hour_data(self) -> Dict[str, Any]:
        """
        Fetch data from the past hour from the database.
        
        Returns:
            Dict[str, Any]: Market data from the last hour
        """
        try:
            # Calculate time range
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(hours=1)).timestamp() * 1000)
            
            # Fetch liquidation data
            liquidations = self.db_handler.fetch_liquidations(start_time, end_time)
            
            # Fetch funding rate data
            funding_rates = self.db_handler.fetch_funding_rates(start_time, end_time)
            
            # Fetch open interest data
            open_interest = self.db_handler.fetch_open_interest(start_time, end_time)
            
            # Fetch transaction data
            transactions = self.db_handler.fetch_transactions(start_time, end_time)
            
            # Process data for better analysis
            processed_data = self._process_data_for_analysis(
                liquidations, funding_rates, open_interest, transactions
            )
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error fetching last hour data: {str(e)}")
            return {}
    
    def _process_data_for_analysis(self, 
                                  liquidations: List[Dict], 
                                  funding_rates: List[Dict],
                                  open_interest: List[Dict],
                                  transactions: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Process raw data into a format better suited for LLM analysis.
        
        Args:
            liquidations (List[Dict]): Raw liquidation data
            funding_rates (List[Dict]): Raw funding rate data
            open_interest (List[Dict]): Raw open interest data
            transactions (Dict[str, List[Dict]]): Raw transaction data
            
        Returns:
            Dict[str, Any]: Processed data suitable for LLM analysis
        """
        result = {
            'timestamp': int(datetime.now().timestamp() * 1000),
            'time_range': {
                'start': int((datetime.now() - timedelta(hours=1)).timestamp() * 1000),
                'end': int(datetime.now().timestamp() * 1000),
                'duration': '1 hour'
            },
            'summary': {},
            'liquidations': {},
            'funding_rates': {},
            'open_interest': {},
            'price_movements': {}
        }
        
        # Process liquidations
        if liquidations:
            # Convert to DataFrame for easier analysis
            liq_df = pd.DataFrame(liquidations)
            
            # Total liquidation volume
            total_liq_volume = liq_df['quantity'].sum() if 'quantity' in liq_df.columns else 0
            
            # Count by side (long/short)
            liq_by_side = {}
            if 'side' in liq_df.columns:
                side_counts = liq_df['side'].value_counts().to_dict()
                liq_by_side = {
                    'long': side_counts.get('buy', 0),  # Liquidated longs show as "buy" liquidations
                    'short': side_counts.get('sell', 0)  # Liquidated shorts show as "sell" liquidations
                }
            
            # Top liquidated symbols
            top_liquidated = {}
            if 'symbol' in liq_df.columns and 'quantity' in liq_df.columns:
                symbol_volume = liq_df.groupby('symbol')['quantity'].sum().sort_values(ascending=False)
                top_liquidated = symbol_volume.head(5).to_dict()
            
            result['liquidations'] = {
                'count': len(liquidations),
                'total_volume': float(total_liq_volume),
                'by_side': liq_by_side,
                'top_liquidated_symbols': top_liquidated
            }
            
            # Add summary
            long_liq_pct = liq_by_side.get('long', 0) / max(1, len(liquidations)) * 100
            result['summary']['liquidation_bias'] = 'long' if long_liq_pct > 60 else ('short' if long_liq_pct < 40 else 'neutral')
        
        # Process funding rates
        if funding_rates:
            fr_df = pd.DataFrame(funding_rates)
            
            # Average funding rate
            avg_funding = fr_df['rate'].mean() if 'rate' in fr_df.columns else 0
            
            # Extreme funding rates
            extreme_positive = {}
            extreme_negative = {}
            if 'symbol' in fr_df.columns and 'rate' in fr_df.columns:
                extreme_pos = fr_df[fr_df['rate'] > 0.001].sort_values('rate', ascending=False)
                extreme_neg = fr_df[fr_df['rate'] < -0.001].sort_values('rate')
                
                extreme_positive = {row['symbol']: float(row['rate']) for _, row in extreme_pos.head(5).iterrows()}
                extreme_negative = {row['symbol']: float(row['rate']) for _, row in extreme_neg.head(5).iterrows()}
            
            result['funding_rates'] = {
                'count': len(funding_rates),
                'average_rate': float(avg_funding),
                'extreme_positive': extreme_positive,
                'extreme_negative': extreme_negative
            }
            
            # Add summary
            result['summary']['funding_bias'] = 'long' if avg_funding < 0 else ('short' if avg_funding > 0 else 'neutral')
        
        # Process open interest
        if open_interest:
            oi_df = pd.DataFrame(open_interest)
            
            # Significant OI changes
            significant_changes = {}
            if 'symbol' in oi_df.columns and 'open_interest' in oi_df.columns:
                # Group by symbol and get latest OI
                latest_oi = oi_df.sort_values('timestamp').groupby('symbol').last()
                
                # For proper change calculation, we'd need historical data beyond just the last hour
                # For now, just report the latest values
                significant_changes = {row.name: float(row['open_interest']) for idx, row in latest_oi.iterrows()}
            
            result['open_interest'] = {
                'count': len(open_interest),
                'latest_values': significant_changes
            }
        
        # Process transactions/price movements
        if transactions:
            price_movements = {}
            
            for symbol, txs in transactions.items():
                if not txs:
                    continue
                
                # Convert to DataFrame
                tx_df = pd.DataFrame(txs)
                if 'price' not in tx_df.columns or len(tx_df) < 2:
                    continue
                
                # Calculate price movement
                tx_df = tx_df.sort_values('timestamp')
                start_price = tx_df['price'].iloc[0]
                end_price = tx_df['price'].iloc[-1]
                price_change = end_price - start_price
                price_change_pct = price_change / start_price * 100
                
                price_movements[symbol] = {
                    'start_price': float(start_price),
                    'end_price': float(end_price),
                    'change': float(price_change),
                    'change_percent': float(price_change_pct)
                }
            
            result['price_movements'] = price_movements
            
            # Add overall market direction to summary
            if price_movements:
                # Check BTC and ETH movements first if available
                btc_change = price_movements.get('BTC/USDT', {}).get('change_percent', 0)
                eth_change = price_movements.get('ETH/USDT', {}).get('change_percent', 0)
                
                # Use BTC and ETH as indicators of overall market direction
                if btc_change != 0 or eth_change != 0:
                    avg_major_change = (btc_change + eth_change) / (2 if eth_change != 0 else 1)
                    result['summary']['market_direction'] = 'bullish' if avg_major_change > 0.5 else ('bearish' if avg_major_change < -0.5 else 'neutral')
                else:
                    # Calculate average of all price movements
                    changes = [v.get('change_percent', 0) for v in price_movements.values()]
                    avg_change = sum(changes) / len(changes) if changes else 0
                    result['summary']['market_direction'] = 'bullish' if avg_change > 0.5 else ('bearish' if avg_change < -0.5 else 'neutral')
        
        return result
    
    async def analyze_market(self) -> Dict[str, Any]:
        """
        Analyze current market data using LLM.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Fetch data
            self.logger.info("Fetching last hour market data for analysis...")
            market_data = await self.fetch_last_hour_data()
            
            if not market_data:
                self.logger.warning("No market data available for analysis")
                return {
                    'error': 'No market data available',
                    'timestamp': int(time.time() * 1000)
                }
            
            # Send to LLM for analysis
            self.logger.info("Sending market data to LLM for analysis...")
            analysis_result = self.llm_connector.analyze_market_data(market_data)
            
            # Store analysis result
            self._store_analysis(analysis_result)
            
            # Update last analysis time
            self.last_analysis_time = int(time.time())
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data: {str(e)}")
            return {
                'error': str(e),
                'timestamp': int(time.time() * 1000)
            }
    
    def _store_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Store the analysis result in the database.
        
        Args:
            analysis (Dict[str, Any]): Analysis results to store
        """
        try:
            if hasattr(self.db_handler, 'store_market_analysis'):
                self.db_handler.store_market_analysis(analysis)
                self.logger.info("Stored market analysis in database")
            else:
                self.logger.warning("Database handler doesn't support storing market analysis")
        except Exception as e:
            self.logger.error(f"Error storing market analysis: {str(e)}")
    
    async def _analysis_loop(self, interval: int) -> None:
        """
        Run the market analysis loop at specified intervals.
        
        Args:
            interval (int): Analysis interval in seconds
        """
        while self.running:
            try:
                # Check if it's time for a new analysis
                current_time = int(time.time())
                if current_time - self.last_analysis_time >= interval:
                    self.logger.info(f"Running scheduled market analysis (interval: {interval}s)")
                    await self.analyze_market()
                
                # Sleep for a shorter time to be responsive
                await asyncio.sleep(min(interval / 10, 60))
                
            except Exception as e:
                self.logger.error(f"Error in market analysis loop: {str(e)}")
                await asyncio.sleep(min(interval / 10, 60))
    
    async def start_analysis(self, interval: int = 3600) -> None:
        """
        Start periodic market analysis.
        
        Args:
            interval (int): Analysis interval in seconds (default: 1 hour)
        """
        if self.running:
            self.logger.warning("Market analyzer is already running")
            return
        
        self.running = True
        self.logger.info(f"Starting market analysis with interval {interval}s")
        await self._analysis_loop(interval)
    
    def stop_analysis(self) -> None:
        """Stop the market analysis."""
        self.running = False
        self.logger.info("Stopping market analysis") 