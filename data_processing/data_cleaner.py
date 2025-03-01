"""
Module for cleaning and preprocessing market data.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

class DataCleaner:
    """Class for cleaning and preprocessing market data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.logger = logging.getLogger("DataCleaner")
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns in a dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            method (str): Method to use ('zscore' or 'iqr')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        df_clean = df.copy()
        total_outliers = 0
        
        for column in columns:
            if column not in df.columns:
                self.logger.warning(f"Column {column} not found in dataframe")
                continue
                
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = z_scores > threshold
                
            elif method == 'iqr':
                # IQR method
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))
                
            else:
                self.logger.error(f"Unknown outlier detection method: {method}")
                return df_clean
            
            # Count outliers
            outlier_count = outliers.sum()
            total_outliers += outlier_count
            
            if outlier_count > 0:
                df_clean = df_clean[~outliers]
                self.logger.info(f"Removed {outlier_count} outliers from column {column}")
        
        self.logger.info(f"Total removed outliers: {total_outliers} ({total_outliers/len(df)*100:.2f}%)")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, 
                              numeric_method: str = 'interpolate', 
                              categorical_method: str = 'mode',
                              max_gap: int = 5) -> pd.DataFrame:
        """
        Handle missing values in dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numeric_method (str): Method for numeric columns ('interpolate', 'mean', 'median', 'drop')
            categorical_method (str): Method for categorical columns ('mode', 'ffill', 'drop')
            max_gap (int): Maximum gap size for interpolation
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_clean = df.copy()
        
        # Count initial missing values
        initial_missing = df_clean.isna().sum().sum()
        
        # Handle missing values by column type
        for column in df_clean.columns:
            missing_count = df_clean[column].isna().sum()
            
            if missing_count == 0:
                continue
                
            # Handle numeric columns
            if np.issubdtype(df_clean[column].dtype, np.number):
                if numeric_method == 'interpolate':
                    # Check if gaps are not too large
                    if df_clean[column].isna().astype(int).sum() > max_gap:
                        self.logger.warning(f"Column {column} has gaps larger than {max_gap}")
                    
                    df_clean[column] = df_clean[column].interpolate(method='linear', limit=max_gap)
                    
                elif numeric_method == 'mean':
                    df_clean[column].fillna(df_clean[column].mean(), inplace=True)
                    
                elif numeric_method == 'median':
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                    
                elif numeric_method == 'drop':
                    df_clean = df_clean.dropna(subset=[column])
                    
            # Handle categorical/object columns
            else:
                if categorical_method == 'mode':
                    mode_value = df_clean[column].mode()[0]
                    df_clean[column].fillna(mode_value, inplace=True)
                    
                elif categorical_method == 'ffill':
                    df_clean[column].fillna(method='ffill', inplace=True)
                    # For any remaining NaNs at the start, fill backward
                    df_clean[column].fillna(method='bfill', inplace=True)
                    
                elif categorical_method == 'drop':
                    df_clean = df_clean.dropna(subset=[column])
        
        # Count remaining missing values
        remaining_missing = df_clean.isna().sum().sum()
        
        self.logger.info(f"Missing values: {initial_missing} -> {remaining_missing}")
        return df_clean
    
    def normalize_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'timestamp', 
                             unit: str = 'ms', tz: str = 'UTC') -> pd.DataFrame:
        """
        Normalize timestamps to a consistent format.
        
        Args:
            df (pd.DataFrame): Input dataframe
            timestamp_col (str): Name of timestamp column
            unit (str): Timestamp unit ('ms', 's', 'ns')
            tz (str): Timezone for timestamps
            
        Returns:
            pd.DataFrame: Dataframe with normalized timestamps
        """
        df_clean = df.copy()
        
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column {timestamp_col} not found in dataframe")
            return df_clean
        
        # Convert timestamps to pandas datetime
        try:
            if unit == 'ms':
                df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], unit='ms')
            elif unit == 's':
                df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], unit='s')
            elif unit == 'ns':
                df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col])
            
            # Set timezone if specified
            if tz:
                df_clean[timestamp_col] = df_clean[timestamp_col].dt.tz_localize(tz)
                
            # Sort by timestamp
            df_clean = df_clean.sort_values(timestamp_col)
            
            self.logger.info(f"Normalized timestamps in column {timestamp_col}")
            
        except Exception as e:
            self.logger.error(f"Error normalizing timestamps: {str(e)}")
        
        return df_clean
    
    def aggregate_to_ohlcv(self, df: pd.DataFrame, timestamp_col: str = 'timestamp',
                          price_col: str = 'price', amount_col: str = 'amount',
                          timeframe: str = '1min') -> pd.DataFrame:
        """
        Aggregate transaction data to OHLCV format.
        
        Args:
            df (pd.DataFrame): Input dataframe with transaction data
            timestamp_col (str): Name of timestamp column
            price_col (str): Name of price column
            amount_col (str): Name of amount/volume column
            timeframe (str): Aggregation timeframe ('1min', '5min', '1h', etc.)
            
        Returns:
            pd.DataFrame: OHLCV dataframe
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(df[timestamp_col]):
            df = self.normalize_timestamps(df, timestamp_col)
        
        # Set timestamp as index
        df_indexed = df.set_index(timestamp_col)
        
        # Perform resampling
        ohlcv = df_indexed.resample(timeframe).agg({
            price_col: ['first', 'max', 'min', 'last'],
            amount_col: 'sum'
        })
        
        # Flatten multi-level columns
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Reset index to get timestamp as column
        ohlcv = ohlcv.reset_index()
        
        self.logger.info(f"Aggregated {len(df)} transactions to {len(ohlcv)} {timeframe} OHLCV candles")
        return ohlcv
    
    def clean_transactions_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a standard cleaning pipeline to transaction data.
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Cleaned transaction data
        """
        try:
            # Remove outliers from price
            df_clean = self.remove_outliers(df, ['price'], method='zscore', threshold=3.0)
            
            # Handle missing values
            df_clean = self.handle_missing_values(df_clean)
            
            # Normalize timestamps
            df_clean = self.normalize_timestamps(df_clean)
            
            self.logger.info(f"Applied standard cleaning pipeline: {len(df)} -> {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error in transaction data cleaning pipeline: {str(e)}")
            return df 