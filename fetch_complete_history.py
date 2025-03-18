import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_history_fetch.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize Binance client
client = Client(
    api_key=os.getenv('BINANCE_API_KEY'),
    api_secret=os.getenv('BINANCE_API_SECRET'),
    # testnet=True
)

def get_symbol_start_time(symbol):
    """Get the earliest available timestamp for a symbol"""
    try:
        # Get the first kline data point
        klines = client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1HOUR,
            "18 Aug, 2017"  # Binance started in 2017
        )
        if klines:
            return pd.to_datetime(klines[0][0], unit='ms')
        return None
    except Exception as e:
        logging.error(f"Error getting start time for {symbol}: {e}")
        return None

def fetch_historical_data(symbol, start_time, end_time=None):
    """
    Fetch historical klines data in chunks to handle large time ranges
    """
    if end_time is None:
        end_time = datetime.now()
    
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        try:
            # Calculate the end time for this chunk (1000 hours at a time)
            chunk_end = min(current_start + timedelta(hours=1000), end_time)
            
            # Fetch data for this chunk
            klines = client.get_historical_klines(
                symbol,
                Client.KLINE_INTERVAL_1HOUR,
                start_str=current_start.strftime("%d %b, %Y %H:%M:%S"),
                end_str=chunk_end.strftime("%d %b, %Y %H:%M:%S")
            )
            
            if klines:
                all_klines.extend(klines)
                logging.info(f"Fetched {len(klines)} records from {current_start} to {chunk_end}")
            
            # Move to the next chunk
            current_start = chunk_end
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Error fetching data chunk: {e}")
            time.sleep(1)  # Wait longer on error
            continue
    
    return all_klines

def process_klines_data(klines):
    """Process raw klines data into a DataFrame"""
    if not klines:
        return None
        
    # Create DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Remove duplicates if any
    df = df.drop_duplicates(subset=['timestamp'])
    
    return df

def save_data(df, symbol):
    """Save data to CSV file"""
    if df is not None and not df.empty:
        # Create directory if it doesn't exist
        os.makedirs('binance_history', exist_ok=True)
        
        # Save to CSV
        filename = f"binance_history/{symbol}_1h_complete_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        logging.info(f"Saved complete historical data to {filename}")
        
        # Print summary
        print("\nData Summary:")
        print(f"Symbol: {symbol}")
        print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total Records: {len(df)}")
        print(f"File Size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
        return True
    return False

def main():
    # Get symbol from user input
    symbol = input("Enter the trading pair (e.g., BTCUSDT): ").strip().upper()
    
    # Get the earliest available timestamp
    start_time = get_symbol_start_time(symbol)
    if not start_time:
        logging.error(f"Could not determine start time for {symbol}")
        return
    
    logging.info(f"Starting data collection for {symbol} from {start_time}")
    
    # Fetch all historical data
    klines = fetch_historical_data(symbol, start_time)
    
    if klines:
        # Process the data
        df = process_klines_data(klines)
        
        # Save the data
        if save_data(df, symbol):
            logging.info("Data collection completed successfully")
        else:
            logging.error("Failed to save data")
    else:
        logging.error("No data was collected")

if __name__ == "__main__":
    main() 