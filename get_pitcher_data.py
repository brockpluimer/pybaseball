import os
import time
import pandas as pd
from pybaseball import pitching_stats
from tqdm import tqdm

# Function to save pitcher data for a specific year
def save_pitcher_data(year):
    csv_filename = f'pitcher_data/pitcher_data_{year}.csv'
    if os.path.exists(csv_filename):
        print(f"Data for year {year} already exists, skipping download.")
        return

    try:
        data = pitching_stats(year)
        if data.empty:
            print(f"No data available for year {year}, moving on.")
            return
        data.to_csv(csv_filename, index=False)
        print(f"Data for year {year} saved to {csv_filename}")
    except Exception as e:
        print(f"Error fetching data for year {year}: {e}")

# Fetch data for all years
start_year = 1871 #first year with pitcher data available
end_year = 2024 #the year of our lord Shohei Ohtani

for year in tqdm(range(start_year, end_year + 1), desc="Fetching data"):
    save_pitcher_data(year)
    # Add delay to respect rate limits
    time.sleep(5)