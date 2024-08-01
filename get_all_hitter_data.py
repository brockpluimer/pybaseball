import os
import time
import pandas as pd
from pybaseball import batting_stats
from tqdm import tqdm

def save_hitter_data(year):
    csv_filename = f'hitter_data/hitter_data_{year}.csv'
    if os.path.exists(csv_filename):
        print(f"Data for year {year} already exists, skipping download.")
        return
    try:
        data = batting_stats(year, qual=0)  # Set qual=0 to get all players
        if data.empty:
            print(f"No data available for year {year}, moving on.")
            return
        data.to_csv(csv_filename, index=False)
        print(f"Data for year {year} saved to {csv_filename}")
    except Exception as e:
        print(f"Error fetching data for year {year}: {e}")

# Fetch data for X number of years
start_year = 2024 # 1870 is the first year with available data
end_year = 2024    # Update this to the current year or the last year you want to fetch

for year in tqdm(range(start_year, end_year + 1), desc="Fetching data"):
    save_hitter_data(year)
    # Add delay to respect rate limits
    time.sleep(5)

print("Data fetching complete.")