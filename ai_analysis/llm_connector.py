"""
Module for connecting to LLM services like OpenAI, DeepSeek, Google Gemini, etc.
"""
import os
import logging
import yaml
import json
import time
from typing import Dict, List, Optional, Any, Union
import requests
from concurrent.futures import ThreadPoolExecutor

class LLMConnector:
    """Base class for LLM service connectors."""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """
        Initialize the LLM connector.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger("LLMConnector")
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.llm_config = config.get('ai_analysis', {}).get('llm', {})
                
                # Set API keys and other settings
                self.provider = self.llm_config.get('provider', 'openai')
                self.api_key = self.llm_config.get('api_key', os.getenv(f"{self.provider.upper()}_API_KEY", ""))
                self.model = self.llm_config.get('model', 'gpt-4')
                self.temperature = self.llm_config.get('temperature', 0.0)
                self.max_tokens = self.llm_config.get('max_tokens', 1000)
                
                if not self.api_key:
                    self.logger.warning(f"No API key found for {self.provider}. Set it in config or environment variable.")
        
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            self.provider = 'openai'
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            self.model = 'gpt-4'
            self.temperature = 0.0
            self.max_tokens = 1000
        
        # Set provider-specific configuration
        self.api_url = None
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Set up provider-specific configuration
        self._setup_provider_config()
    
    def _setup_provider_config(self):
        """Set up provider-specific configuration (to be implemented by subclasses)."""
        pass
    
    def _prepare_market_analysis_prompt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the prompt for market analysis.
        
        Args:
            data (Dict[str, Any]): Market data
            
        Returns:
            Dict[str, Any]: Formatted prompt
        """
        # Format the prompt
        system_prompt = """You are a cryptocurrency market analysis AI. Analyze the provided data and extract key insights:
1. Market trends: Overall direction and notable trends in the market
2. Anomalies: Unusual patterns or events in the data
3. Trading opportunities: Potential profitable trades based on the data
4. Risk assessment: Identify potential risks in the current market"""
        
        # Format data as text
        data_text = json.dumps(data, indent=2)
        user_prompt = f"""Analyze the following cryptocurrency market data from the past hour:

{data_text}

Provide a concise analysis focusing on:
1. Key market movements
2. Unusual liquidation events
3. Significant funding rate changes
4. Open interest trends
5. Your trading insight for the next 1-4 hours

Format your analysis in clear sections."""
        
        # Return messages array
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call LLM API with formatted messages.
        
        Args:
            messages (List[Dict[str, str]]): Formatted messages
            
        Returns:
            Dict[str, Any]: API response
        """
        if not self.api_url:
            raise ValueError(f"API URL not set for {self.provider}")
            
        # Prepare payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Make API call
        start_time = time.time()
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response_time = time.time() - start_time
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            
            self.logger.info(f"{self.provider.capitalize()} analysis completed in {response_time:.2f}s")
            
            return {
                'analysis': analysis,
                'timestamp': int(time.time() * 1000),
                'provider': self.provider,
                'model': self.model,
            }
        else:
            self.logger.error(f"{self.provider.capitalize()} API error: {response.status_code} - {response.text}")
            return {
                'error': f"API error: {response.status_code}",
                'timestamp': int(time.time() * 1000),
            }
    
    def analyze_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send market data to LLM for analysis.
        
        Args:
            data (Dict[str, Any]): Formatted market data
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Prepare the prompt
            messages = self._prepare_market_analysis_prompt(data)
            
            # Call the API
            return self._call_api(messages)
            
        except Exception as e:
            self.logger.error(f"Error in {self.provider} analysis: {str(e)}")
            return {
                'error': str(e),
                'timestamp': int(time.time() * 1000),
                'provider': self.provider
            }


class OpenAIConnector(LLMConnector):
    """Connector for OpenAI's API."""
    
    def _setup_provider_config(self):
        """Set up OpenAI-specific configuration."""
        self.provider = 'openai'
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        # Use default model if not specifically set for OpenAI
        if self.model == 'gpt-4' or 'gpt' not in self.model:
            # Default to GPT-4 or fallback to GPT-3.5-turbo if GPT-4 wasn't specified
            self.model = self.llm_config.get('model', 'gpt-4')


class DeepSeekConnector(LLMConnector):
    """Connector for DeepSeek's API."""
    
    def _setup_provider_config(self):
        """Set up DeepSeek-specific configuration."""
        self.provider = 'deepseek'
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Use default model if not specifically set for DeepSeek
        if 'deepseek' not in self.model:
            # Default to deepseek-chat if no deepseek model was specified
            self.model = self.llm_config.get('model', 'deepseek-chat')


class GeminiConnector(LLMConnector):
    """Connector for Google's Gemini AI API."""
    
    def _setup_provider_config(self):
        """Set up Gemini-specific configuration."""
        self.provider = 'gemini'
        
        # Set the model name based on configuration
        if 'gemini' not in self.model:
            # Default to gemini-1.5-pro if no gemini model was specified
            self.model = self.llm_config.get('model', 'gemini-1.5-pro')
        
        # Gemini API requires the API key as a URL parameter
        # The base URL format includes the model name
        model_name = self.model
        api_version = self.llm_config.get('api_version', 'v1beta')
        
        # Format the API URL with the model and API key
        self.api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
        
        # For Gemini, we don't use Bearer auth in headers, we add the API key as a parameter
        self.headers = {"Content-Type": "application/json"}
    
    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call Gemini API with formatted messages.
        
        Args:
            messages (List[Dict[str, str]]): Formatted messages
            
        Returns:
            Dict[str, Any]: API response
        """
        if not self.api_url:
            raise ValueError(f"API URL not set for {self.provider}")
        
        # Add API key as URL parameter
        url_with_key = f"{self.api_url}?key={self.api_key}"
        
        # Convert messages to Gemini format
        contents = []
        for message in messages:
            role = message["role"]
            # Map OpenAI roles to Gemini roles (system becomes user with a specific prefix)
            if role == "system":
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System instruction: {message['content']}"}]
                })
            else:
                contents.append({
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": message["content"]}]
                })
        
        # Prepare payload for Gemini
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        # Make API call
        start_time = time.time()
        response = requests.post(url_with_key, headers=self.headers, json=payload)
        response_time = time.time() - start_time
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            
            # Extract the response text from Gemini's format
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    analysis = "".join([part.get("text", "") for part in parts])
                    
                    self.logger.info(f"Gemini analysis completed in {response_time:.2f}s")
                    
                    return {
                        'analysis': analysis,
                        'timestamp': int(time.time() * 1000),
                        'provider': self.provider,
                        'model': self.model,
                    }
            
            # If we couldn't parse the response properly
            self.logger.error(f"Gemini API returned unexpected response structure: {json.dumps(result)[:500]}...")
            return {
                'error': "Unexpected API response structure",
                'raw_response': str(result)[:1000],
                'timestamp': int(time.time() * 1000),
            }
        else:
            self.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return {
                'error': f"API error: {response.status_code}",
                'timestamp': int(time.time() * 1000),
            }


def get_llm_connector(provider: str = 'openai', config_path: str = 'config/settings.yaml') -> LLMConnector:
    """
    Factory method to get the appropriate LLM connector.
    
    Args:
        provider (str): LLM provider name ('openai', 'deepseek', 'gemini', etc.)
        config_path (str): Path to configuration file
        
    Returns:
        LLMConnector: Configured LLM connector
    """
    if provider.lower() == 'openai':
        return OpenAIConnector(config_path)
    elif provider.lower() == 'deepseek':
        return DeepSeekConnector(config_path)
    elif provider.lower() == 'gemini':
        return GeminiConnector(config_path)
    else:
        # Default to OpenAI
        logger = logging.getLogger("LLMConnector")
        logger.warning(f"Unknown provider '{provider}'. Defaulting to OpenAI.")
        return OpenAIConnector(config_path) 