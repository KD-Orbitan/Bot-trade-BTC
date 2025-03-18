import pandas as pd
import pandas_ta as ta

# Đọc dữ liệu từ file CSV
file_path = "binance_history/BTCUSDT_1h_complete_20250318.csv"
df = pd.read_csv(file_path)

# Chuyển timestamp sang datetime, sắp xếp và đặt làm index
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by="timestamp")
df.set_index("timestamp", inplace=True)

# Chuyển các cột Open, High, Low, Close, Volume sang dạng float
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

### TÍNH TOÁN CÁC CHỈ BÁO KỸ THUẬT ###

# VWAP (Volume Weighted Average Price)
df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

# EMA (21, 50)
df["EMA_21"] = ta.ema(df["close"], length=21)
df["EMA_50"] = ta.ema(df["close"], length=50)

# MACD (12, 26, 9)
macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
df["MACD"] = macd["MACD_12_26_9"]
df["MACD_Signal"] = macd["MACDs_12_26_9"]

# RSI (14)
df["RSI_14"] = ta.rsi(df["close"], length=14)

# Stochastic RSI (14, 3, 3)
stoch_rsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
df["Stoch_RSI"] = stoch_rsi["STOCHRSIk_14_14_3_3"]

# Bollinger Bands (20, 2)
bbands = ta.bbands(df["close"], length=20, std=2)
df["BB_Upper"] = bbands["BBU_20_2.0"]
df["BB_Middle"] = bbands["BBM_20_2.0"]
df["BB_Lower"] = bbands["BBL_20_2.0"]

# ATR (14)
df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)

# OBV
df["OBV"] = ta.obv(df["close"], df["volume"])

# MFI (14)
mfi_data = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
print(mfi_data.head())  # Xem dữ liệu trả về
df["MFI"] = mfi_data.astype(float)



### LƯU FILE ĐÃ XỬ LÝ ###
output_file = "binance_history/BTCUSDT_1h_processed.csv"
selected_columns = ["open", "high", "low", "close", "volume",
                    "VWAP", "EMA_21", "EMA_50", "MACD", "MACD_Signal",
                    "RSI_14", "Stoch_RSI", "BB_Upper", "BB_Middle", "BB_Lower",
                    "ATR", "OBV", "MFI"]

df[selected_columns].to_csv(output_file, index=True)

print(f"Dữ liệu đã được xử lý và lưu vào {output_file}")
