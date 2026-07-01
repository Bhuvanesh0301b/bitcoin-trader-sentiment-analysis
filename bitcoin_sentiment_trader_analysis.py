import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== Advanced Bitcoin Trader Sentiment Analysis ===\n")

# Load Data
fg = pd.read_csv('fear_greed_index.csv')
trader = pd.read_csv('historical_data.csv')

print(f"Data Loaded: {fg.shape[0]} sentiment records | {trader.shape[0]} trades\n")

# Preprocessing
fg['date'] = pd.to_datetime(fg['date'])
trader['Timestamp'] = pd.to_datetime(trader['Timestamp IST'], errors='coerce')
trader = trader.dropna(subset=['Timestamp'])
trader['trade_date'] = trader['Timestamp'].dt.date
fg['trade_date'] = fg['date'].dt.date

trader['Closed PnL'] = pd.to_numeric(trader['Closed PnL'], errors='coerce')
trader['Size USD'] = pd.to_numeric(trader['Size USD'], errors='coerce')

# Merge
merged = trader.merge(fg[['trade_date', 'classification', 'value']], on='trade_date', how='left')

# ==================== ADVANCED ANALYSIS ====================

print("=== Performance by Market Sentiment ===")
sentiment_perf = merged.groupby('classification').agg(
    Total_PnL=('Closed PnL', 'sum'),
    Num_Trades=('Closed PnL', 'count'),
    Avg_PnL=('Closed PnL', 'mean'),
    Win_Rate=('Closed PnL', lambda x: (x > 0).mean() * 100),
    Total_Volume=('Size USD', 'sum')
).round(2)
print(sentiment_perf)

# Top Traders
print("\n=== Top 5 Traders by PnL ===")
top_traders = merged.groupby('Account')['Closed PnL'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(5)
print(top_traders)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.barplot(x=sentiment_perf.index, y='Total_PnL', data=sentiment_perf.reset_index(), ax=axes[0,0])
axes[0,0].set_title('Total PnL by Sentiment')
axes[0,0].tick_params(axis='x', rotation=45)

sns.barplot(x=sentiment_perf.index, y='Win_Rate', data=sentiment_perf.reset_index(), ax=axes[0,1])
axes[0,1].set_title('Win Rate (%) by Sentiment')

sns.barplot(x=sentiment_perf.index, y='Avg_PnL', data=sentiment_perf.reset_index(), ax=axes[1,0])
axes[1,0].set_title('Average PnL per Trade by Sentiment')

sns.barplot(x=sentiment_perf.index, y='Num_Trades', data=sentiment_perf.reset_index(), ax=axes[1,1])
axes[1,1].set_title('Number of Trades by Sentiment')

plt.tight_layout()
plt.savefig('sentiment_analysis.png', dpi=300)
print("\nAdvanced visualizations saved as 'sentiment_analysis.png'")

# Save results
sentiment_perf.to_csv('sentiment_performance_summary.csv')
print("\n✅ Project completed with advanced insights!")
print("Files generated: sentiment_analysis.png + sentiment_performance_summary.csv")