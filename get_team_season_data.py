import os
import pandas as pd
from pybaseball import standings
from tqdm import tqdm
import time
import random

def ensure_dir(directory):
    """
    Ensure that the specified directory exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_and_save_standings_for_decade(start_year, end_year):
    """
    Fetches final standings for each team from start_year to end_year and saves it to a single Excel file.
    Each year is saved as a separate sheet in the Excel file.
    """
    # Ensure the directory for saving results exists
    ensure_dir("season_standings")
    
    # Create an Excel writer object for the current decade
    output_file = f"season_standings/{start_year}_{end_year}_final_standings.xlsx"
    
    with pd.ExcelWriter(output_file) as writer:
        for year in tqdm(range(start_year, end_year + 1), desc=f"Fetching standings for {start_year}-{end_year}"):
            try:
                # Fetch the standings for the year
                divisions = standings(year)
                
                # Combine all divisions into a single DataFrame for the year
                final_standings = pd.concat(divisions)
                
                # Add the year as a column to distinguish data in the combined DataFrame
                final_standings['Year'] = year
                
                # Write the DataFrame to a sheet named after the year
                final_standings.to_excel(writer, sheet_name=str(year), index=False)
                
                print(f"Saved standings for {year}")
            except Exception as e:
                print(f"Error fetching standings for {year}: {str(e)}")
            
            # Implement random delay to mimic human browsing behavior
            time.sleep(random.uniform(3, 7))

def main():
    start_year = 1876  # Adjust this to your desired start year
    end_year = 2024  # Adjust this to your desired end year (or current year)
    chunk_size = 10   # Define the number of years in each chunk

    for decade_start in range(start_year, end_year + 1, chunk_size):
        decade_end = min(decade_start + chunk_size - 1, end_year)
        
        print(f"Fetching final season standings for {decade_start}-{decade_end}...")
        fetch_and_save_standings_for_decade(decade_start, decade_end)
        
        print(f"Data fetching for {decade_start}-{decade_end} complete!")
        
        # Wait for an hour before fetching the next decade
        if decade_end < end_year:
            print("Waiting for a half hour before fetching the next decade...")
            time.sleep(1800)  # Sleep for half hour

if __name__ == "__main__":
    main()