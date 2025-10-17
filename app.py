from agent import StockAgent
import csv
import os
from datetime import datetime

if __name__ == "__main__":
    ticker = input("Enter Stock Ticker (e.g. AAPL, TSLA, RELIANCE.NS): ").strip().upper()
    if "." not in ticker:
        ticker = ticker + ".NS"

    agent = StockAgent(ticker, lookback=60)

    result = agent.predict()

    print("\n=== Prediction Result ===")
    print(f"Ticker: {result['ticker']}")
    print(f"Last Close: {result['last_close']:.2f}")
    print(f"Predicted Next Close: {result['predicted_close']:.2f}")
    print(f"Trend: {result['trend']}")

    csv_file = "predictions.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Last_Close", "Predicted_Close", "Trend"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["ticker"],
            result["last_close"],
            result["predicted_close"],
            result["trend"]
        ])

    print("\nSaved to predictions.csv")
