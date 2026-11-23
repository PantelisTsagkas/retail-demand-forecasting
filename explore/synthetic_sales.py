import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# make the experiment reproducible
np.random.seed(42)

# generate 180 days of sale
dates = pd.date_range(start="2026-01-01", periods=180, freq="D")

# create a weekly sales pattern
weekly_pattern = np.array([0.85, 0.90, 0.95, 1.00, 1.10, 1.50, 1.20])

# generate sales
sales = []

for i, date in enumerate(dates):
    day_factor = weekly_pattern[date.dayofweek]

    trend = 0.05 * i

    noise = np.random.normal(0, 8)

    daily_sales = 100 * day_factor + trend + noise
    sales.append(max(0, round(daily_sales)))


# create Dataframe
df = pd.DataFrame({"date": dates, "sales": sales})

plt.figure(figsize=(14, 5))
plt.plot(df["date"], df["sales"])

plt.title("Daily Supermarket Sales")
plt.xlabel("Date")
plt.ylabel("Units Sold")

plt.xticks(rotation=45)
plt.grid()

plt.show()
