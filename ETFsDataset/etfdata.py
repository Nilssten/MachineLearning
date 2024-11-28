import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("2021-12-11-BC6610D5-07C5-4F7E-9586-896D143D9302.csv") #dataframe
print(df.columns)

df['price_date'] = pd.to_datetime(df['price_date'], format='%Y-%m-%d') # Convert the column to datetime format

# Filter data for the specified date range
start_date = '2021-10-01'
end_date = '2021-11-01'
filtered_df = df[(df['price_date'] >= start_date) & (df['price_date'] <= end_date)]

# Calculate volatility (standard deviation of close price) for each fund symbol and sort in descending order
volatility_df = filtered_df.groupby('fund_symbol')['close'].std().reset_index(name='volatility')
volatility_df = volatility_df.sort_values(by='volatility', ascending=False)

top_10_symbols = volatility_df.head(10)['fund_symbol'].tolist() # Filter top 10 most volatile fund symbols
top_10_df = filtered_df[filtered_df['fund_symbol'].isin(top_10_symbols)]

# Apply MinMax scaling
scaler = MinMaxScaler()
columns_to_scale = ['open', 'high', 'low', 'close', 'adj_close']
scaled_data = scaler.fit_transform(top_10_df[columns_to_scale])

# Use .loc to assign the scaled data back to the DataFrame
top_10_df.loc[:, columns_to_scale] = scaled_data

top_10_df.to_csv("filteredetf.csv", index=False)

# Plot overlaid line graph for the top  10 volatile symbols
plt.figure(figsize=(12,  8))
for symbol in top_10_symbols:
    symbol_data = filtered_df[filtered_df['fund_symbol'] == symbol]
    plt.plot(symbol_data['price_date'], symbol_data['close'], label=symbol)

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Top 10 Most Volatile Symbols Close Price Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Create subplots with  2 columns and  5 rows
fig, axs = plt.subplots(5,  2, figsize=(8,  12))
axs = axs.ravel()

colors = plt.cm.tab10.colors # Define colors

# Plot close prices for each symbol in a separate subplot with a different color
for i, (symbol, color) in enumerate(zip(top_10_symbols, colors)):
    symbol_data = filtered_df[filtered_df['fund_symbol'] == symbol]
    axs[i].plot(symbol_data['price_date'], symbol_data['close'], color=color)
    axs[i].set_title(symbol)
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('Close Price')

plt.tight_layout()
plt.show()