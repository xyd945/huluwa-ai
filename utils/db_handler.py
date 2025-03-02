"""
Module for handling database operations for the AI trading agent.
"""
import os
import yaml
import logging
import sqlite3
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class DatabaseHandler:
    """Handles database operations for storing and retrieving trading data."""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """
        Initialize the database handler.
        
        Args:
            config (Union[str, Dict[str, Any]]): Database configuration path or direct config dict
        """
        self.logger = logging.getLogger("DatabaseHandler")
        
        # Load configuration
        try:
            if isinstance(config, str):
                # It's a path to a config file
                with open(config, 'r') as file:
                    config_data = yaml.safe_load(file).get('database', {})
            else:
                # It's already a config dictionary
                config_data = config
            
            # Get database path from config
            db_path = config_data.get('path', 'data/trading.db')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Connect to SQLite database
            self.conn = sqlite3.connect(db_path)
            self.logger.info(f"Connected to SQLite database at {db_path}")
            
            # Create tables if they don't exist
            self._create_tables()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create liquidations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS liquidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            order_type TEXT,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create funding_rates table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS funding_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            rate REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            next_funding_time INTEGER,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create open_interest table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS open_interest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open_interest REAL NOT NULL,
            open_interest_usd REAL,
            timestamp INTEGER NOT NULL,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create token_launches table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS token_launches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT,
            token_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            launch_date INTEGER,
            initial_price REAL,
            description TEXT,
            url TEXT,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create transactions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            cost REAL,
            timestamp INTEGER NOT NULL,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create signals table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL NOT NULL,
            raw_prediction REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create orders table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            order_id TEXT NOT NULL,
            side TEXT NOT NULL,
            type TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL,
            status TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create market_analysis table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            analysis TEXT NOT NULL,
            error TEXT,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Commit the changes
        self.conn.commit()
        self.logger.info("Database tables created")
    
    def store_liquidation(self, data: Dict[str, Any]) -> bool:
        """
        Store liquidation data in the database.
        
        Args:
            data (Dict[str, Any]): Liquidation data
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO liquidations 
            (exchange, symbol, side, quantity, price, timestamp, order_type, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('exchange'),
                data.get('symbol'),
                data.get('side'),
                data.get('quantity'),
                data.get('price'),
                data.get('timestamp'),
                data.get('order_type'),
                str(data.get('raw_data', ''))
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing liquidation data: {str(e)}")
            return False
    
    def store_funding_rate(self, data: Dict[str, Any]) -> bool:
        """
        Store funding rate data in the database.
        
        Args:
            data (Dict[str, Any]): Funding rate data
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO funding_rates 
            (exchange, symbol, rate, next_funding_time, timestamp, raw_data)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data.get('exchange'),
                data.get('symbol'),
                data.get('funding_rate'),
                data.get('next_funding_time'),
                data.get('timestamp'),
                str(data.get('raw_data', ''))
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing funding rate data: {str(e)}")
            return False
    
    def store_open_interest(self, data: Dict[str, Any]) -> bool:
        """
        Store open interest data in the database.
        
        Args:
            data (Dict[str, Any]): Open interest data
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO open_interest 
            (exchange, symbol, open_interest, timestamp, raw_data)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                data.get('exchange'),
                data.get('symbol'),
                data.get('open_interest'),
                data.get('timestamp'),
                str(data.get('raw_data', ''))
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing open interest data: {str(e)}")
            return False
    
    def store_token_launch(self, data: Dict[str, Any]) -> bool:
        """
        Store token launch data in the database.
        
        Args:
            data (Dict[str, Any]): Token launch data
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO token_launches 
            (exchange, token_name, symbol, launch_date, initial_price, description, url, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('exchange', ''),
                data.get('token_name', ''),
                data.get('symbol', ''),
                data.get('launch_date', 0),
                data.get('initial_price', 0.0),
                data.get('description', ''),
                data.get('url', ''),
                str(data.get('raw_data', ''))
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing token launch data: {str(e)}")
            return False
    
    def store_transaction(self, data: Dict[str, Any]) -> bool:
        """
        Store transaction data in the database.
        
        Args:
            data (Dict[str, Any]): Transaction data
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO transactions 
            (exchange, symbol, side, quantity, price, timestamp, trade_id, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('exchange'),
                data.get('symbol'),
                data.get('side'),
                data.get('quantity'),
                data.get('price'),
                data.get('timestamp'),
                data.get('trade_id'),
                str(data.get('raw_data', ''))
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing transaction data: {str(e)}")
            return False
    
    def store_signal(self, symbol: str, signal: Dict[str, Any], features: Optional[str] = None) -> bool:
        """
        Store trading signal in the database.
        
        Args:
            symbol (str): Trading symbol
            signal (Dict[str, Any]): Signal data
            features (Optional[str]): Serialized features used for the signal
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO signals 
            (symbol, signal, confidence, raw_prediction, timestamp, features)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                signal.get('signal'),
                signal.get('confidence'),
                signal.get('raw_prediction'),
                signal.get('timestamp'),
                features
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing signal data: {str(e)}")
            return False
    
    def store_order(self, order: Dict[str, Any]) -> bool:
        """
        Store order data in the database.
        
        Args:
            order (Dict[str, Any]): Order data
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO orders 
            (exchange, symbol, order_id, side, type, quantity, price, status, timestamp, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.get('exchange'),
                order.get('symbol'),
                order.get('id'),
                order.get('side'),
                order.get('type'),
                order.get('amount'),
                order.get('price'),
                order.get('status'),
                order.get('timestamp'),
                str(order.get('info', ''))
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing order data: {str(e)}")
            return False
    
    def get_liquidations(self, symbol: Optional[str] = None, 
                       start_time: Optional[int] = None, 
                       end_time: Optional[int] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """
        Retrieve liquidation data from the database.
        
        Args:
            symbol (Optional[str]): Filter by symbol
            start_time (Optional[int]): Start timestamp in milliseconds
            end_time (Optional[int]): End timestamp in milliseconds
            limit (int): Maximum number of records to return
            
        Returns:
            pd.DataFrame: Liquidation data
        """
        query = "SELECT * FROM liquidations WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            return pd.read_sql_query(query, self.conn, params=params)
        except Exception as e:
            self.logger.error(f"Error retrieving liquidation data: {str(e)}")
            return pd.DataFrame()
    
    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
            self.logger.info("Database connection closed")

    def fetch_liquidations(self, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """
        Fetch liquidation data within a time range.
        
        Args:
            start_time (int): Start timestamp (milliseconds)
            end_time (int): End timestamp (milliseconds)
            
        Returns:
            List[Dict[str, Any]]: Liquidation data
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT * FROM liquidations
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            ''', (start_time, end_time))
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch all results
            liquidations = []
            for row in cursor.fetchall():
                liquidation = {columns[i]: row[i] for i in range(len(columns))}
                liquidations.append(liquidation)
            
            self.logger.info(f"Fetched {len(liquidations)} liquidations between {start_time} and {end_time}")
            return liquidations
            
        except Exception as e:
            self.logger.error(f"Error fetching liquidations: {str(e)}")
            return []

    def fetch_funding_rates(self, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """
        Fetch funding rate data within a time range.
        
        Args:
            start_time (int): Start timestamp (milliseconds)
            end_time (int): End timestamp (milliseconds)
            
        Returns:
            List[Dict[str, Any]]: Funding rate data
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT * FROM funding_rates
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            ''', (start_time, end_time))
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch all results
            funding_rates = []
            for row in cursor.fetchall():
                funding_rate = {columns[i]: row[i] for i in range(len(columns))}
                funding_rates.append(funding_rate)
            
            self.logger.info(f"Fetched {len(funding_rates)} funding rates between {start_time} and {end_time}")
            return funding_rates
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rates: {str(e)}")
            return []

    def fetch_open_interest(self, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """
        Fetch open interest data within a time range.
        
        Args:
            start_time (int): Start timestamp (milliseconds)
            end_time (int): End timestamp (milliseconds)
            
        Returns:
            List[Dict[str, Any]]: Open interest data
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT * FROM open_interest
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            ''', (start_time, end_time))
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch all results
            open_interest_data = []
            for row in cursor.fetchall():
                oi_entry = {columns[i]: row[i] for i in range(len(columns))}
                open_interest_data.append(oi_entry)
            
            self.logger.info(f"Fetched {len(open_interest_data)} open interest entries between {start_time} and {end_time}")
            return open_interest_data
            
        except Exception as e:
            self.logger.error(f"Error fetching open interest data: {str(e)}")
            return []

    def fetch_transactions(self, start_time: int, end_time: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch transaction data within a time range.
        
        Args:
            start_time (int): Start timestamp (milliseconds)
            end_time (int): End timestamp (milliseconds)
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Transaction data by symbol
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT * FROM transactions
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            ''', (start_time, end_time))
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch all results and organize by symbol
            transactions_by_symbol = {}
            for row in cursor.fetchall():
                transaction = {columns[i]: row[i] for i in range(len(columns))}
                symbol = transaction.get('symbol', 'unknown')
                
                if symbol not in transactions_by_symbol:
                    transactions_by_symbol[symbol] = []
                
                transactions_by_symbol[symbol].append(transaction)
            
            total_transactions = sum(len(txs) for txs in transactions_by_symbol.values())
            self.logger.info(f"Fetched {total_transactions} transactions for {len(transactions_by_symbol)} symbols between {start_time} and {end_time}")
            return transactions_by_symbol
            
        except Exception as e:
            self.logger.error(f"Error fetching transactions: {str(e)}")
            return {}

    def create_market_analysis_table(self) -> None:
        """Create table for storing market analysis results."""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                analysis TEXT NOT NULL,
                error TEXT,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            self.conn.commit()
            self.logger.info("Created market_analysis table")
            
        except Exception as e:
            self.logger.error(f"Error creating market_analysis table: {str(e)}")

    def store_market_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        Store market analysis in the database.
        
        Args:
            analysis (Dict[str, Any]): Analysis data
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            # Create table if it doesn't exist
            self.create_market_analysis_table()
            
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO market_analysis 
            (timestamp, provider, model, analysis, error, raw_data)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                analysis.get('timestamp', int(time.time() * 1000)),
                analysis.get('provider', 'unknown'),
                analysis.get('model', 'unknown'),
                analysis.get('analysis', ''),
                analysis.get('error', None),
                json.dumps(analysis)
            ))
            
            self.conn.commit()
            self.logger.info(f"Stored market analysis from {analysis.get('provider')} in database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing market analysis: {str(e)}")
            return False

    def fetch_latest_analysis(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch the latest market analysis results.
        
        Args:
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Analysis results
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT * FROM market_analysis
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch all results
            analysis_results = []
            for row in cursor.fetchall():
                analysis = {columns[i]: row[i] for i in range(len(columns))}
                
                # Parse raw_data if available
                if 'raw_data' in analysis and analysis['raw_data']:
                    try:
                        analysis['raw_data'] = json.loads(analysis['raw_data'])
                    except:
                        pass
                
                analysis_results.append(analysis)
            
            self.logger.info(f"Fetched {len(analysis_results)} latest market analysis results")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error fetching latest analysis: {str(e)}")
            return [] 