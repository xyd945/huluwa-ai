"""
Script to test the ExchangeConnector functionality.
"""
import asyncio
import logging
import json
import websockets
import time
from typing import Dict, Any, List

from data_ingestion.exchange_connector import ExchangeConnector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_binance_connector():
    """Test basic functionality of the Binance connector."""
    connector = ExchangeConnector('binance')
    logger = logging.getLogger('test_binance')
    
    # Test fetching markets
    logger.info("Testing fetch_markets...")
    markets = connector.fetch_markets()
    logger.info(f"Found {len(markets)} markets on Binance")
    if markets:
        logger.info(f"First market: {markets[0]['symbol']}")
    
    # Test fetching ticker
    logger.info("\nTesting fetch_ticker...")
    ticker = connector.fetch_ticker("BTC/USDT")
    logger.info(f"BTC/USDT ticker: {ticker}")
    
    # Test fetching OHLCV data
    logger.info("\nTesting fetch_ohlcv...")
    ohlcv = connector.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=5)
    logger.info(f"Got {len(ohlcv)} OHLCV candles")
    for candle in ohlcv[:2]:  # Show first two candles
        timestamp, open_price, high, low, close, volume = candle
        logger.info(f"Candle: Time: {timestamp}, Open: {open_price}, Close: {close}")
    
    # Test fetching order book
    logger.info("\nTesting fetch_order_book...")
    order_book = connector.fetch_order_book("BTC/USDT", limit=5)
    if 'bids' in order_book and 'asks' in order_book:
        logger.info(f"Order book - Bids: {len(order_book['bids'])}, Asks: {len(order_book['asks'])}")
        if order_book['bids']:
            logger.info(f"Top bid: {order_book['bids'][0]}")
        if order_book['asks']:
            logger.info(f"Top ask: {order_book['asks'][0]}")
    
    # Test fetching funding rates
    logger.info("\nTesting fetch_funding_rates...")
    funding_rates = await connector.fetch_funding_rates()
    if funding_rates:
        logger.info(f"Found {len(funding_rates)} funding rates")
        # Show first funding rate as example
        if len(funding_rates) > 0:
            first_key = list(funding_rates.keys())[0]
            logger.info(f"Example: {first_key}: {funding_rates[first_key]}")
    else:
        logger.info("No funding rates found or not supported")

    # Test getting futures symbols (for open interest)
    logger.info("\nTesting futures symbols availability...")
    markets = connector.fetch_markets()
    futures_symbols = [market['symbol'] for market in markets if 
                      market.get('future', False) or 
                      'future' in market.get('type', '').lower() or
                      'swap' in market.get('type', '').lower()]
    logger.info(f"Found {len(futures_symbols)} futures/swap markets")
    if futures_symbols:
        logger.info(f"Examples: {futures_symbols[:5]}")

    # Test open interest endpoint directly
    if hasattr(connector.exchange, 'fetch_open_interest'):
        logger.info("\nTesting fetch_open_interest...")
        try:
            # For Binance futures, symbols need specific format
            if 'binance' in connector.exchange_id:
                # For Binance, try with the first few futures symbols
                for symbol in futures_symbols[:5]:
                    try:
                        open_interest = connector.exchange.fetch_open_interest(symbol)
                        logger.info(f"Open interest for {symbol}: {open_interest}")
                        break
                    except Exception as e:
                        logger.error(f"Error fetching open interest for {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Error testing open interest: {str(e)}")
    else:
        logger.info("fetch_open_interest method not available in this exchange")

    # Test symbol conversion methods
    logger.info("\nTesting symbol conversion methods...")
    
    def convert_to_ccxt_symbol(symbol_raw: str) -> str:
        """Convert exchange symbol format to CCXT format."""
        # Test conversion for Binance style symbols
        if "/" not in symbol_raw:
            if symbol_raw.endswith("USDT"):
                return f"{symbol_raw[:-4]}/USDT"
            elif symbol_raw.endswith("BTC"):
                return f"{symbol_raw[:-3]}/BTC"
            elif symbol_raw.endswith("ETH"):
                return f"{symbol_raw[:-3]}/ETH"
        return symbol_raw
    
    test_symbols = ["BTCUSDT", "ETHBTC", "ADAUSDT"]
    for symbol in test_symbols:
        ccxt_symbol = convert_to_ccxt_symbol(symbol)
        logger.info(f"Converted {symbol} to {ccxt_symbol}")
    
    # Test WebSocket connection
    logger.info("\nTesting WebSocket connection...")
    
    async def test_binance_websocket():
        endpoint = "wss://stream.binance.com:9443/ws"
        try:
            async with websockets.connect(endpoint) as websocket:
                # Subscribe to BTC/USDT trade channel
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": ["btcusdt@trade"],
                    "id": 1
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Subscription response: {response}")
                
                # Wait for one trade message
                message = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(message)
                logger.info(f"Trade message received: {data}")
                
                # Unsubscribe
                unsubscribe_msg = {
                    "method": "UNSUBSCRIBE",
                    "params": ["btcusdt@trade"],
                    "id": 2
                }
                await websocket.send(json.dumps(unsubscribe_msg))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Unsubscription response: {response}")
                
                return True
        except Exception as e:
            logger.error(f"WebSocket test failed: {str(e)}")
            return False
    
    websocket_success = await test_binance_websocket()
    logger.info(f"WebSocket connection test {'succeeded' if websocket_success else 'failed'}")

    # Test futures WebSocket for liquidations
    logger.info("\nTesting futures WebSocket for liquidations...")
    
    async def test_binance_futures_websocket():
        endpoint = "wss://fstream.binance.com/ws"
        try:
            async with websockets.connect(endpoint) as websocket:
                # Subscribe to all liquidation events
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": ["!forceOrder@arr"],
                    "id": 1
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Futures subscription response: {response}")
                
                # Since liquidations are rare, we won't wait for one
                # Just test if the connection works
                
                # Unsubscribe
                unsubscribe_msg = {
                    "method": "UNSUBSCRIBE",
                    "params": ["!forceOrder@arr"],
                    "id": 2
                }
                await websocket.send(json.dumps(unsubscribe_msg))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Futures unsubscription response: {response}")
                
                return True
        except Exception as e:
            logger.error(f"Futures WebSocket test failed: {str(e)}")
            return False
    
    futures_websocket_success = await test_binance_futures_websocket()
    logger.info(f"Futures WebSocket connection test {'succeeded' if futures_websocket_success else 'failed'}")

    # Test individual futures open interest with multiple symbols
    logger.info("\nTesting individual futures open interest...")
    
    if futures_symbols:
        successful_symbols = []
        failed_symbols = []
        
        for symbol in futures_symbols[:10]:  # Test first 10 futures symbols
            try:
                open_interest = await asyncio.to_thread(connector.exchange.fetch_open_interest, symbol)
                logger.info(f"Open interest for {symbol}: {open_interest.get('openInterestAmount', 'N/A')}")
                successful_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Failed to get open interest for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
        
        logger.info(f"Successfully fetched open interest for {len(successful_symbols)} symbols")
        logger.info(f"Failed to fetch open interest for {len(failed_symbols)} symbols")

async def test_open_interest():
    """Test open interest functionality specifically."""
    connector = ExchangeConnector('binance')
    logger = logging.getLogger('test_open_interest')
    
    logger.info("\n=== TESTING OPEN INTEREST ===")
    
    # Get all market data from exchange
    logger.info("Fetching all markets...")
    markets = await asyncio.to_thread(connector.exchange.fetch_markets)
    
    # Filter for different types of futures markets to test
    linear_futures = []
    inverse_futures = []
    all_futures = []
    
    for market in markets:
        symbol = market.get('symbol', '')
        contract_type = market.get('type', '').lower()
        is_future = market.get('future', False)
        is_linear = market.get('linear', False)
        
        # Log detailed information about the first few futures markets for debugging
        if is_future or 'future' in contract_type or 'swap' in contract_type:
            all_futures.append(market)
            if len(all_futures) <= 5:
                logger.info(f"Futures market details: {symbol} | Type: {contract_type} | Linear: {is_linear}")
                logger.info(f"Market info: {market}")
            
            if is_linear:
                linear_futures.append(market)
            else:
                inverse_futures.append(market)
    
    logger.info(f"Found {len(all_futures)} total futures markets")
    logger.info(f"Found {len(linear_futures)} linear futures markets")
    logger.info(f"Found {len(inverse_futures)} inverse futures markets")
    
    # Let's test with different symbol formats to see what works
    test_formats = [
        {"name": "Original CCXT format", "symbols": [m.get('symbol') for m in all_futures[:5]]},
        {"name": "Spot-like format", "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
        {"name": "Futures with colon", "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]},
        {"name": "Futures with perp", "symbols": ["BTC/USDT:USDT_PERP", "ETH/USDT:USDT_PERP"]}
    ]
    
    # Now test each format to see which one works
    for format_test in test_formats:
        format_name = format_test["name"]
        symbols = format_test["symbols"]
        
        logger.info(f"\nTesting open interest with {format_name}: {symbols}")
        
        for symbol in symbols:
            try:
                # Use asyncio.to_thread to avoid blocking
                open_interest = await asyncio.to_thread(connector.exchange.fetch_open_interest, symbol)
                success = 'openInterestAmount' in open_interest
                
                if success:
                    logger.info(f"✅ SUCCESS: Open interest for {symbol}: {open_interest.get('openInterestAmount')}")
                    logger.info(f"Full response: {open_interest}")
                else:
                    logger.warning(f"⚠️ PARTIAL SUCCESS: Got response but no openInterestAmount for {symbol}: {open_interest}")
            except Exception as e:
                logger.error(f"❌ FAILED: Error fetching open interest for {symbol}: {str(e)}")

async def test_liquidation_websocket():
    """Test liquidation WebSocket connection and message handling."""
    logger = logging.getLogger('test_liquidation')
    
    logger.info("\n=== TESTING LIQUIDATION STREAM ===")
    
    # Define a simple symbol conversion function to test
    def convert_to_ccxt_symbol(symbol_raw: str) -> str:
        """Convert exchange symbol format to CCXT format."""
        # Return if already in CCXT format
        if "/" in symbol_raw:
            return symbol_raw
            
        # Handle Binance futures symbols
        if symbol_raw.endswith("USDT"):
            return f"{symbol_raw[:-4]}/USDT"
        elif symbol_raw.endswith("BUSD"):
            return f"{symbol_raw[:-4]}/BUSD"
        elif symbol_raw.endswith("BTC"):
            return f"{symbol_raw[:-3]}/BTC"
        elif symbol_raw.endswith("ETH"):
            return f"{symbol_raw[:-3]}/ETH"
        elif symbol_raw.endswith("USD"):
            # Might be a USD-settled futures contract
            return f"{symbol_raw[:-3]}/USD" 
        
        # If we can't parse the symbol, log a warning and return as-is
        logger.warning(f"Could not convert symbol to CCXT format: {symbol_raw}")
        return symbol_raw
    
    # Test the symbol conversion function with various formats
    test_symbols = ["BTCUSDT", "ETHBTC", "ADAUSDT", "BTCBUSD", "SOLUSDT"]
    logger.info("Testing symbol conversion function:")
    for symbol in test_symbols:
        ccxt_symbol = convert_to_ccxt_symbol(symbol)
        logger.info(f"Converted {symbol} to {ccxt_symbol}")
    
    # Connect to Binance futures WebSocket for liquidations
    logger.info("\nConnecting to Binance futures liquidation stream...")
    endpoint = "wss://fstream.binance.com/ws"
    
    try:
        async with websockets.connect(endpoint) as websocket:
            # Subscribe to all liquidation events
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": ["!forceOrder@arr"],  # This subscribes to all liquidation events
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            
            # Wait for subscription confirmation
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            logger.info(f"Subscription response: {response}")
            
            # Process messages (we'll wait for 60 seconds to try to catch some liquidations)
            logger.info("Waiting for liquidation events (60 seconds timeout)...")
            end_time = time.time() + 60
            liquidation_count = 0
            
            while time.time() < end_time:
                try:
                    # Set a short timeout so we can check if time is up
                    message = await asyncio.wait_for(websocket.recv(), timeout=2)
                    data = json.loads(message)
                    
                    # Check if it's a liquidation event
                    if isinstance(data, dict):
                        if "e" in data and data["e"] == "forceOrder":
                            # This is a single liquidation event
                            liquidation_count += 1
                            logger.info(f"Received liquidation event: {data}")
                            
                            # Try to process it with our symbol conversion
                            if "o" in data:
                                order_data = data["o"]
                                symbol_raw = order_data.get("s", "")
                                symbol = convert_to_ccxt_symbol(symbol_raw)
                                
                                logger.info(f"Processed liquidation: Exchange={symbol_raw}, CCXT={symbol}")
                        elif "data" in data and isinstance(data["data"], dict) and data["data"].get("e") == "forceOrder":
                            # This might be a wrapped liquidation event
                            liquidation_count += 1
                            logger.info(f"Received wrapped liquidation event: {data}")
                    
                except asyncio.TimeoutError:
                    # Just a timeout on recv, continue the loop
                    continue
            
            logger.info(f"Received {liquidation_count} liquidation events in 60 seconds")
            
            # Unsubscribe before closing
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE",
                "params": ["!forceOrder@arr"],
                "id": 2
            }
            await websocket.send(json.dumps(unsubscribe_msg))
            
            # Wait for unsubscription confirmation
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            logger.info(f"Unsubscription response: {response}")
            
            return True
            
    except Exception as e:
        logger.error(f"Liquidation WebSocket test failed: {str(e)}")
        return False

async def main():
    """Run all tests."""
    await test_binance_connector()
    await test_open_interest()
    await test_liquidation_websocket()

if __name__ == "__main__":
    asyncio.run(main()) 