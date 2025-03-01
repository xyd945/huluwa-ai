"""
Module for generating trading signals using trained models.
"""
import numpy as np
import pandas as pd
import logging
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import tensorflow as tf
import pickle

class PredictionEngine:
    """Generates trading signals using trained AI models."""
    
    def __init__(self, model_path: str, config_path: str = 'config/settings.yaml'):
        """
        Initialize the prediction engine.
        
        Args:
            model_path (str): Path to the trained model
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger("PredictionEngine")
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.sequence_length = 10  # Default value
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.model_config = config.get('ai_module', {})
                
                # Set prediction parameters
                self.threshold = self.model_config.get('prediction', {}).get('threshold', 0.75)
                self.model_type = self.model_config.get('model_type', 'lstm')
                
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            self.threshold = 0.75
            self.model_type = 'lstm'
        
        # Load model and scaler
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the trained model and scaler.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Load Keras model
            if self.model_type == 'lstm':
                self.model = tf.keras.models.load_model(f"{self.model_path}.keras")
            
            # Load scaler
            with open(f"{self.model_path}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.logger.info(f"Model and scaler loaded from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            features (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Processed features ready for model prediction
        """
        if self.scaler is None:
            self.logger.error("Scaler not loaded")
            return np.array([])
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # For LSTM models, reshape to sequence format
            if self.model_type == 'lstm':
                # Make sure we have enough data for a sequence
                if len(features_scaled) < self.sequence_length:
                    self.logger.warning(f"Not enough data points for sequence (needed {self.sequence_length}, got {len(features_scaled)})")
                    return np.array([])
                
                # Use the most recent sequence
                sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, features_scaled.shape[1])
                return sequence
            else:
                # For non-sequence models, just use the latest data point
                return features_scaled[-1:] 
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return np.array([])
    
    def generate_signal(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a trading signal from the input features.
        
        Args:
            features (pd.DataFrame): Input features
            
        Returns:
            Dict[str, Any]: Trading signal with prediction and confidence
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return {'signal': 'neutral', 'confidence': 0.0}
        
        try:
            # Prepare features for prediction
            X = self.prepare_features(features)
            if len(X) == 0:
                return {'signal': 'neutral', 'confidence': 0.0}
            
            # Make prediction
            prediction = self.model.predict(X)[0][0]
            
            # Determine signal based on prediction and threshold
            if prediction > self.threshold:
                signal = 'buy'
                confidence = float(prediction)
            elif prediction < 1 - self.threshold:
                signal = 'sell'
                confidence = float(1 - prediction)
            else:
                signal = 'neutral'
                confidence = float(max(prediction, 1 - prediction))
            
            signal_data = {
                'signal': signal,
                'confidence': confidence,
                'raw_prediction': float(prediction),
                'timestamp': pd.Timestamp.now().timestamp() * 1000
            }
            
            self.logger.info(f"Generated {signal} signal with confidence {confidence:.4f}")
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return {'signal': 'neutral', 'confidence': 0.0} 