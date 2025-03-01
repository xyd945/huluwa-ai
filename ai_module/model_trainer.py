"""
Module for training AI models on processed market data.
"""
import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import yaml

class ModelTrainer:
    """Trains machine learning models on processed market data."""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """
        Initialize the model trainer.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger("ModelTrainer")
        self.model = None
        self.scaler = StandardScaler()
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.model_config = config.get('ai_module', {})
                
                # Set model parameters from config
                self.batch_size = self.model_config.get('training', {}).get('batch_size', 64)
                self.epochs = self.model_config.get('training', {}).get('epochs', 100)
                self.test_size = self.model_config.get('training', {}).get('test_size', 0.2)
                self.model_type = self.model_config.get('model_type', 'lstm')
                
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            # Set default values
            self.batch_size = 64
            self.epochs = 100
            self.test_size = 0.2
            self.model_type = 'lstm'
    
    def prepare_data(self, data: pd.DataFrame, target_col: str, 
                   sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training by creating sequences.
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column for prediction
            sequence_length (int): Number of time steps in each sequence
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Make a copy of the data
        df = data.copy()
        
        # Extract target
        y = df[target_col].values
        
        # Drop target from features
        df = df.drop(columns=[target_col])
        
        # Scale features
        df_scaled = self.scaler.fit_transform(df)
        
        # Create sequences
        X = []
        y_seq = []
        
        for i in range(len(df_scaled) - sequence_length):
            X.append(df_scaled[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        X = np.array(X)
        y_seq = np.array(y_seq)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_seq, test_size=self.test_size, shuffle=False
        )
        
        self.logger.info(f"Prepared data with shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, input_shape: Tuple[int, int], output_shape: int = 1) -> Sequential:
        """
        Build an LSTM model.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (sequence_length, features)
            output_shape (int): Number of output units
            
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info(f"Built LSTM model with input shape: {input_shape}")
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train the model on prepared data.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        if self.model_type == 'lstm':
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_lstm_model(input_shape)
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Evaluate on test data
            y_pred = (self.model.predict(X_test) > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary'),
                'history': history.history
            }
            
            self.logger.info(f"Model training completed with test accuracy: {metrics['accuracy']:.4f}")
            return metrics
        
        else:
            self.logger.error(f"Unsupported model type: {self.model_type}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model and scaler to files.
        
        Args:
            filepath (str): Base filepath for saving the model
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.model is None:
            self.logger.error("No model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save Keras model
            if self.model_type == 'lstm':
                self.model.save(f"{filepath}.keras")
            
            # Save scaler
            with open(f"{filepath}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.logger.info(f"Model and scaler saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model and scaler from files.
        
        Args:
            filepath (str): Base filepath for loading the model
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Load Keras model
            if self.model_type == 'lstm':
                self.model = tf.keras.models.load_model(f"{filepath}.keras")
            
            # Load scaler
            with open(f"{filepath}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.logger.info(f"Model and scaler loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False 