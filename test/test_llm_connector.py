"""
Test script for the LLM connector.
"""
import os
import sys
import json
import argparse
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta

# Add project root directory to Python path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_analysis.llm_connector import get_llm_connector
from utils.logger import setup_logger

# Configure basic logging for the test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMConnectorTest")

def create_sample_market_data() -> Dict[str, Any]:
    """
    Create sample market data for testing the LLM analysis.
    
    Returns:
        Dict[str, Any]: Sample market data
    """
    # Current time and one hour ago
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    
    # Format timestamps in milliseconds
    now_ts = int(now.timestamp() * 1000)
    hour_ago_ts = int(one_hour_ago.timestamp() * 1000)
    
    return {
        "liquidations": [
            {
                "symbol": "BTC/USDT",
                "side": "sell",
                "quantity": 0.5,
                "price": 48750.0,
                "timestamp": now_ts - 2400000
            },
            {
                "symbol": "ETH/USDT",
                "side": "buy",
                "quantity": 5.2,
                "price": 2340.0,
                "timestamp": now_ts - 1800000
            },
            {
                "symbol": "BTC/USDT",
                "side": "sell",
                "quantity": 1.2,
                "price": 48500.0,
                "timestamp": now_ts - 900000
            }
        ],
        "funding_rates": [
            {
                "symbol": "BTC/USDT",
                "rate": 0.0001,
                "timestamp": hour_ago_ts
            },
            {
                "symbol": "ETH/USDT",
                "rate": 0.0002,
                "timestamp": hour_ago_ts
            },
            {
                "symbol": "BTC/USDT",
                "rate": 0.00015,
                "timestamp": now_ts
            },
            {
                "symbol": "ETH/USDT",
                "rate": 0.00025,
                "timestamp": now_ts
            }
        ],
        "open_interest": [
            {
                "symbol": "BTC/USDT",
                "open_interest": 150000000,
                "timestamp": hour_ago_ts
            },
            {
                "symbol": "ETH/USDT",
                "open_interest": 85000000,
                "timestamp": hour_ago_ts
            },
            {
                "symbol": "BTC/USDT",
                "open_interest": 152000000,
                "timestamp": now_ts
            },
            {
                "symbol": "ETH/USDT",
                "open_interest": 86500000,
                "timestamp": now_ts
            }
        ],
        "price_data": {
            "BTC/USDT": {
                "start_price": 48500.0,
                "end_price": 49200.0,
                "high": 49300.0,
                "low": 48400.0,
                "volume": 12500.0
            },
            "ETH/USDT": {
                "start_price": 2320.0,
                "end_price": 2380.0,
                "high": 2390.0,
                "low": 2310.0,
                "volume": 45600.0
            }
        },
        "analysis_period": {
            "start": hour_ago_ts,
            "end": now_ts,
            "duration_hours": 1
        }
    }

async def test_llm_connector(config_path: str = '../config/settings.yaml', provider: str = 'deepseek') -> None:
    """
    Test the LLM connector with sample market data.
    
    Args:
        config_path (str): Path to configuration file
        provider (str): LLM provider to test
    """
    try:
        logger.info(f"Testing {provider} LLM connector...")
        
        # Initialize the LLM connector
        llm_connector = get_llm_connector(provider, config_path)
        logger.info(f"Initialized {provider} connector with model: {llm_connector.model}")
        
        # Create sample market data
        market_data = create_sample_market_data()
        logger.info(f"Created sample market data with {len(market_data['liquidations'])} liquidations, " +
                   f"{len(market_data['funding_rates'])} funding rates, and " +
                   f"{len(market_data['open_interest'])} open interest records")
        
        # Call the LLM for analysis
        logger.info(f"Sending data to {provider} for analysis...")
        result = llm_connector.analyze_market_data(market_data)
        
        # Check the result
        if 'error' in result:
            logger.error(f"LLM analysis failed: {result['error']}")
            if 'raw_response' in result:
                logger.error(f"Raw response: {result['raw_response']}")
            return
        
        # Successfully got analysis
        logger.info(f"Successfully received analysis from {provider} ({result['model']})")
        
        # Print a summary of the analysis
        analysis_text = result['analysis']
        logger.info(f"Analysis length: {len(analysis_text)} characters")
        
        # Print the first 500 characters of the analysis
        preview = analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text
        logger.info(f"Analysis preview:\n{preview}")
        
        # Save the full result to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_output_{provider}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Full analysis saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error testing {provider} LLM connector: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test the LLM connector")
    parser.add_argument("--config", type=str, default="../config/settings.yaml",
                        help="Path to configuration file")
    parser.add_argument("--provider", type=str, default="deepseek",
                        help="LLM provider to test (openai, deepseek, gemini)")
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_llm_connector(args.config, args.provider))

if __name__ == "__main__":
    main() 