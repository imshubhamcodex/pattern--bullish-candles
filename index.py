import numpy as np
import yfinance as yf
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from datetime import datetime, time as datetime_time
from pytz import timezone



# Fetch data
def fetch_data(ticker="^NSEI", time_interval="1m"):
    # end_date = datetime.today().date()
    # start_date = end_date - timedelta(days=6)
    # data = yf.download(ticker, start=start_date, end=end_date, interval=time_interval)
    
    # return data
    
    return read_csv()

def read_csv():
    data = pd.read_csv('./data_stocks_n/NIFTY50-5.csv')
    data = data[['date', 'open', 'low', 'high', 'close']]
    data.rename(columns={'date': 'Datetime'}, inplace=True)
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S%z')  # Adjust the format as needed
    data.set_index('Datetime', inplace=True)
    df = pd.DataFrame(data)
    df.index = df.index.tz_localize(None)
    df = df.dropna()
    
    return df


def total_signal(df, current_candle):
    current_pos = df.index.get_loc(current_candle)
    
    # Buy condition
    c1 = df['high'].iloc[current_pos] >= df['high'].iloc[current_pos-1]
    c2 = df['high'].iloc[current_pos-1] >= df['low'].iloc[current_pos]
    c3 = df['low'].iloc[current_pos] > df['high'].iloc[current_pos-2]
    c4 = df['high'].iloc[current_pos-2] >= df['low'].iloc[current_pos-1]
    c5 = df['low'].iloc[current_pos-1] >= df['high'].iloc[current_pos-3]
    c6 = df['high'].iloc[current_pos-3] >= df['low'].iloc[current_pos-2]
    c7 = df['low'].iloc[current_pos-2] >= df['low'].iloc[current_pos-3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 1

    # # Symmetrical conditions for short (sell condition)
    # c1 = df['low'].iloc[current_pos] <= df['low'].iloc[current_pos-1]
    # c2 = df['low'].iloc[current_pos-1] <= df['high'].iloc[current_pos]
    # c3 = df['high'].iloc[current_pos] <= df['low'].iloc[current_pos-2]
    # c4 = df['low'].iloc[current_pos-2] <= df['high'].iloc[current_pos-1]
    # c5 = df['high'].iloc[current_pos-1] <= df['low'].iloc[current_pos-3]
    # c6 = df['low'].iloc[current_pos-3] <= df['high'].iloc[current_pos-2]
    # c7 = df['high'].iloc[current_pos-2] <= df['high'].iloc[current_pos-3]

    # if c1 and c2 and c3 and c4 and c5 and c6 and c7:
    #     return -1

    return 0


def count_signals(df):
    signal_counts = df['TotalSignal'].value_counts()
    print(signal_counts)


# Backtest function modified to track drawdown
def backtest_data(df, tp, sl):
    initial_balance_pts = 0  # Starting points
    balance = initial_balance_pts
    position = None
    entry_price = None

    total_trades = 0
    wins = 0
    losses = 0
    balance_history = [balance]  # Store balance for drawdown calculation

    i = 0
    while i < len(df):
        signal = df['TotalSignal'].iloc[i]
        price = df['close'].iloc[i]

        # Enter long position
        if signal == 1 and position is None:
            position = 'long'
            entry_price = price
            total_trades += 1
            for j in range(i + 1, len(df)):
                if df['high'].iloc[j] >= entry_price + tp:
                    balance += tp
                    wins += 1
                    position = None
                    entry_price = None
                    # i = j
                    balance_history.append(balance)
                    break
                elif df['low'].iloc[j] <= entry_price - sl:
                    balance -= sl
                    losses += 1
                    position = None
                    entry_price = None
                    # i = j
                    balance_history.append(balance)
                    break

        # Enter short position
        elif signal == -1 and position is None:
            position = 'short'
            entry_price = price
            total_trades += 1
            for j in range(i + 1, len(df)):
                if df['low'].iloc[j] <= entry_price - tp:
                    balance += tp
                    wins += 1
                    position = None
                    entry_price = None
                    # i = j
                    balance_history.append(balance)
                    break
                elif df['high'].iloc[j] >= entry_price + sl:
                    balance -= sl
                    losses += 1
                    position = None
                    entry_price = None
                    # i = j
                    balance_history.append(balance)
                    break

        # Move to the next candle if no trade was executed
        i += 1

    # Calculate win ratio
    win_ratio = wins*100 / total_trades if total_trades > 0 else 0

    # Calculate drawdown
    max_balance = np.maximum.accumulate(balance_history)
    drawdown = (max_balance - balance_history)
    max_drawdown = np.max(drawdown)

    return win_ratio, max_drawdown, balance/100, total_trades

# Function to generate heatmaps for both win ratio and drawdown
def generate_heatmaps(df, tp_range, sl_range):
    win_results = np.zeros((len(tp_range), len(sl_range)))
    drawdown_results = np.zeros((len(tp_range), len(sl_range)))
    balance_results = np.zeros((len(tp_range), len(sl_range)))
    total_trades_result = np.zeros((len(tp_range), len(sl_range)))
    
    for i, tp in enumerate(tp_range):
        for j, sl in enumerate(sl_range):
            win_ratio, max_drawdown, balance, total_trades = backtest_data(df, tp, sl)
            win_results[i, j] = win_ratio
            drawdown_results[i, j] = max_drawdown
            balance_results[i, j] = balance
            total_trades_result[i, j] = total_trades
            

    # # Plot Win Ratio Heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(win_results, xticklabels=sl_range, yticklabels=tp_range, annot=True, fmt=".2f", cmap="coolwarm")
    # plt.xlabel("Stop-Loss (SL)")
    # plt.ylabel("Take-Profit (TP)")
    # plt.title("Heat Map of Win Ratio for Varying TP and SL")
    # plt.show()

    # Plot Drawdown Heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(drawdown_results, xticklabels=sl_range, yticklabels=tp_range, annot=True, fmt=".2f", cmap="coolwarm")
    # plt.xlabel("Stop-Loss (SL)")
    # plt.ylabel("Take-Profit (TP)")
    # plt.title("Heat Map of Maximum Drawdown for Varying TP and SL")
    # plt.show()
    
    
    #  # Plot Balance Heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(balance_results, xticklabels=sl_range, yticklabels=tp_range, annot=True, fmt=".2f", cmap="coolwarm")
    # plt.xlabel("Stop-Loss (SL)")
    # plt.ylabel("Take-Profit (TP)")
    # plt.title("Heat Map of Point gained for Varying TP and SL")
    # plt.show()
    
    # # Plot Total Trade Heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(total_trades_result, xticklabels=sl_range, yticklabels=tp_range, annot=True, fmt=".2f", cmap="coolwarm")
    # plt.xlabel("Stop-Loss (SL)")
    # plt.ylabel("Take-Profit (TP)")
    # plt.title("Heat Map of Total Trade for Varying TP and SL")
    # plt.show()
    
    
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot Win Ratio Heatmap
    sns.heatmap(win_results, xticklabels=sl_range, yticklabels=tp_range, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_xlabel("Stop-Loss (SL)")
    axes[0].set_ylabel("Take-Profit (TP)")
    axes[0].set_title("Heat Map of Win Ratio for Varying TP and SL, Total Trade: " + str(total_trades_result[0,0]))

    # Plot Point Gained Heatmap
    sns.heatmap(balance_results, xticklabels=sl_range, yticklabels=tp_range, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_xlabel("Stop-Loss (SL)")
    axes[1].set_ylabel("Take-Profit (TP)")
    axes[1].set_title("Heat Map of Point gained for Varying TP and SL, Total Trade: " + str(total_trades_result[0,0]))

    plt.tight_layout()
    plt.show()

def main():
    df = fetch_data()
    df.index = pd.to_datetime(df.index)

    # Calculate signals
    total_signals = [total_signal(df, current_candle) for current_candle in df.index]
    df['TotalSignal'] = total_signals

    # Define TP and SL ranges
    tp_range = range(200, 270, 20)  # From 40 to 250 with step 30
    sl_range = range(130, 270, 20)  # From 40 to 250 with step 30

    generate_heatmaps(df, tp_range, sl_range)

main()
