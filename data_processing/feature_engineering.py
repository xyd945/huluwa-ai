"""
Module for creating features from market data for AI model training.
"""
import pandas as pd
import numpy as np
import logging
import ta
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

class FeatureEngineer:
    """Class for generating features from market data."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.logger = logging.getLogger("FeatureEngineer")
    
    def add_price_features(self, df: pd.DataFrame, 
                          price_cols: List[str] = ['open', 'high', 'low', 'close']) -> pd.DataFrame:
        """
        Add basic price-based features.
        
        Args:
            df (pd.DataFrame): Input OHLCV dataframe
            price_cols (List[str]): List of price column names
            
        Returns:
            pd.DataFrame: Dataframe with added features
        """
        df_features = df.copy()
        
        # Check if required columns exist
        missing_cols = [col for col in price_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing price columns: {', '.join(missing_cols)}")
            return df_features
        
        try:
            # Price changes
            df_features['price_change'] = df_features['close'].diff()
            df_features['price_change_pct'] = df_features['close'].pct_change() * 100
            
            # Price range features
            df_features['candle_range'] = df_features['high'] - df_features['low']
            df_features['candle_body'] = abs(df_features['close'] - df_features['open'])
            df_features['candle_upper_wick'] = df_features['high'] - df_features[['open', 'close']].max(axis=1)
            df_features['candle_lower_wick'] = df_features[['open', 'close']].min(axis=1) - df_features['low']
            
            # Candle classification
            df_features['candle_type'] = np.where(df_features['close'] >= df_features['open'], 'bullish', 'bearish')
            
            # Typical price
            df_features['typical_price'] = (df_features['high'] + df_features['low'] + df_features['close']) / 3
            
            self.logger.info(f"Added {len(df_features.columns) - len(df.columns)} price-based features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error adding price features: {str(e)}")
            return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            df (pd.DataFrame): Input OHLCV dataframe
            
        Returns:
            pd.DataFrame: Dataframe with added features
        """
        df_features = df.copy()
        
        if 'volume' not in df.columns:
            self.logger.warning("Volume column not found in dataframe")
            return df_features
        
        try:
            # Volume changes
            df_features['volume_change'] = df_features['volume'].diff()
            df_features['volume_change_pct'] = df_features['volume'].pct_change() * 100
            
            # Rolling volume features
            for window in [5, 10, 20]:
                df_features[f'volume_ma_{window}'] = df_features['volume'].rolling(window).mean()
                df_features[f'volume_std_{window}'] = df_features['volume'].rolling(window).std()
                df_features[f'volume_ratio_{window}'] = df_features['volume'] / df_features[f'volume_ma_{window}']
            
            # Volume and price relationship
            df_features['price_volume_change_ratio'] = df_features['price_change_pct'] / df_features['volume_change_pct'].replace(0, np.nan)
            
            # Fill NaN values
            df_features = df_features.fillna(method='bfill')
            
            self.logger.info(f"Added {len(df_features.columns) - len(df.columns)} volume-based features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error adding volume features: {str(e)}")
            return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using the TA-Lib library.
        
        Args:
            df (pd.DataFrame): Input OHLCV dataframe
            
        Returns:
            pd.DataFrame: Dataframe with added features
        """
        df_features = df.copy()
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns for technical indicators: {', '.join(missing_cols)}")
            return df_features
        
        try:
            # Add momentum indicators
            df_features['rsi'] = ta.momentum.RSIIndicator(df_features['close']).rsi()
            df_features['stoch_k'] = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['close']).stoch()
            df_features['stoch_d'] = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['close']).stoch_signal()
            df_features['macd'] = ta.trend.MACD(df_features['close']).macd()
            df_features['macd_signal'] = ta.trend.MACD(df_features['close']).macd_signal()
            df_features['macd_diff'] = ta.trend.MACD(df_features['close']).macd_diff()
            
            # Add trend indicators
            df_features['sma_20'] = ta.trend.SMAIndicator(df_features['close'], window=20).sma_indicator()
            df_features['sma_50'] = ta.trend.SMAIndicator(df_features['close'], window=50).sma_indicator()
            df_features['sma_200'] = ta.trend.SMAIndicator(df_features['close'], window=200).sma_indicator()
            df_features['ema_20'] = ta.trend.EMAIndicator(df_features['close'], window=20).ema_indicator()
            
            # Add volatility indicators
            df_features['bbands_upper'] = ta.volatility.BollingerBands(df_features['close']).bollinger_hband()
            df_features['bbands_lower'] = ta.volatility.BollingerBands(df_features['close']).bollinger_lband()
            df_features['bbands_width'] = (df_features['bbands_upper'] - df_features['bbands_lower']) / df_features['close']
            df_features['atr'] = ta.volatility.AverageTrueRange(df_features['high'], df_features['low'], df_features['close']).average_true_range()
            
            # Add volume indicators
            df_features['obv'] = ta.volume.OnBalanceVolumeIndicator(df_features['close'], df_features['volume']).on_balance_volume()
            df_features['volume_cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df_features['high'], df_features['low'], df_features['close'], df_features['volume']).chaikin_money_flow()
            
            # Fill NaN values
            df_features = df_features.fillna(method='bfill')
            
            self.logger.info(f"Added {len(df_features.columns) - len(df.columns)} technical indicators")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def merge_external_features(self, price_df: pd.DataFrame, 
                               liquidation_df: Optional[pd.DataFrame] = None,
                               funding_df: Optional[pd.DataFrame] = None,
                               open_interest_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge external features with price data.
        
        Args:
            price_df (pd.DataFrame): Price data with timestamp/datetime column
            liquidation_df (pd.DataFrame): Liquidation features
            funding_df (pd.DataFrame): Funding rate features
            open_interest_df (pd.DataFrame): Open interest features
            
        Returns:
            pd.DataFrame: Merged dataframe
        """
        merged_df = price_df.copy()
        time_col = 'datetime' if 'datetime' in price_df.columns else 'timestamp'
        
        try:
            # Merge liquidation features if available
            if liquidation_df is not None and not liquidation_df.empty:
                liquidation_time_col = 'datetime' if 'datetime' in liquidation_df.columns else 'timestamp'
                liquidation_features = liquidation_df.drop_duplicates(subset=[liquidation_time_col])
                
                merged_df = pd.merge_asof(
                    merged_df.sort_values(time_col),
                    liquidation_features.sort_values(liquidation_time_col),
                    left_on=time_col,
                    right_on=liquidation_time_col,
                    direction='nearest',
                    tolerance=pd.Timedelta('5 minutes')
                )
                
            # Merge funding rate features if available
            if funding_df is not None and not funding_df.empty:
                funding_time_col = 'datetime' if 'datetime' in funding_df.columns else 'timestamp'
                funding_features = funding_df.drop_duplicates(subset=[funding_time_col])
                
                merged_df = pd.merge_asof(
                    merged_df.sort_values(time_col),
                    funding_features.sort_values(funding_time_col),
                    left_on=time_col,
                    right_on=funding_time_col,
                    direction='backward',
                    tolerance=pd.Timedelta('1 day')
                )
                
            # Merge open interest features if available
            if open_interest_df is not None and not open_interest_df.empty:
                oi_time_col = 'datetime' if 'datetime' in open_interest_df.columns else 'timestamp'
                oi_features = open_interest_df.drop_duplicates(subset=[oi_time_col])
                
                merged_df = pd.merge_asof(
                    merged_df.sort_values(time_col),
                    oi_features.sort_values(oi_time_col),
                    left_on=time_col,
                    right_on=oi_time_col,
                    direction='nearest',
                    tolerance=pd.Timedelta('15 minutes')
                )
            
            # Clean up duplicate columns
            cols_to_drop = [col for col in merged_df.columns if col.endswith('_y')]
            merged_df = merged_df.drop(columns=cols_to_drop)
            
            # Rename remaining _x columns
            merged_df.columns = [col[:-2] if col.endswith('_x') else col for col in merged_df.columns]
            
            self.logger.info(f"Successfully merged external features. Final dataframe has {len(merged_df.columns)} columns")
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging external features: {str(e)}")
            return merged_df
    
    def create_target_variable(self, df: pd.DataFrame, look_ahead: int = 24, threshold_pct: float = 1.0) -> pd.DataFrame:
        """
        Create target variable for supervised learning.
        
        Args:
            df (pd.DataFrame): Feature dataframe with price data
            look_ahead (int): Number of periods to look ahead
            threshold_pct (float): Threshold percentage for classification
            
        Returns:
            pd.DataFrame: Dataframe with target variables
        """
        df_with_target = df.copy()
        
        try:
            # Future price
            df_with_target['future_price'] = df_with_target['close'].shift(-look_ahead)
            
            # Price movement
            df_with_target['price_movement'] = df_with_target['future_price'] - df_with_target['close']
            df_with_target['price_movement_pct'] = (df_with_target['price_movement'] / df_with_target['close']) * 100
            
            # Create target classes
            conditions = [
                df_with_target['price_movement_pct'] > threshold_pct,
                df_with_target['price_movement_pct'] < -threshold_pct
            ]
            choices = ['up', 'down']
            df_with_target['target'] = np.select(conditions, choices, default='neutral')
            
            # Create numeric target for regression
            df_with_target['target_regression'] = df_with_target['price_movement_pct']
            
            # Create binary target for classification
            df_with_target['target_binary'] = np.where(df_with_target['price_movement_pct'] > 0, 1, 0)
            
            # Drop rows with NaN targets (the last look_ahead rows)
            df_with_target = df_with_target.dropna(subset=['future_price'])
            
            self.logger.info(f"Created target variable with {look_ahead} period look-ahead and {threshold_pct}% threshold")
            
            # Count target distribution
            target_counts = df_with_target['target'].value_counts()
            for target, count in target_counts.items():
                self.logger.info(f"Target '{target}': {count} instances ({count/len(df_with_target)*100:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error creating target variable: {str(e)}")
        
        return df_with_target
    
    def apply_feature_pipeline(self, 
                              ohlcv_df: pd.DataFrame, 
                              liquidation_df: Optional[pd.DataFrame] = None,
                              funding_df: Optional[pd.DataFrame] = None,
                              open_interest_df: Optional[pd.DataFrame] = None,
                              create_target: bool = True) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.
        
        Args:
            ohlcv_df (pd.DataFrame): OHLCV price data
            liquidation_df (pd.DataFrame): Liquidation data (optional)
            funding_df (pd.DataFrame): Funding rate data (optional)
            open_interest_df (pd.DataFrame): Open interest data (optional)
            create_target (bool): Whether to create target variable
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        try:
            # Start with price features
            df_features = self.add_price_features(ohlcv_df)
            
            # Add volume features
            df_features = self.add_volume_features(df_features)
            
            # Add technical indicators
            df_features = self.add_technical_indicators(df_features)
            
            # Merge external features
            df_features = self.merge_external_features(
                df_features, 
                liquidation_df, 
                funding_df,
                open_interest_df
            )
            
            # Create target if requested
            if create_target:
                df_features = self.create_target_variable(df_features)
            
            self.logger.info(f"Applied full feature engineering pipeline. Final feature count: {len(df_features.columns)}")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            return ohlcv_df 