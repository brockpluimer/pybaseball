import os
import pandas as pd
from pybaseball import standings, schedule_and_record
from tqdm import tqdm
import time
import random

def ensure_dir(directory):
    """
    Ensure that the specified directory exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_and_save_game_data_for_year(year):
    """
    Fetches game-by-game data for each team in a specific year and saves it to Excel files.
    Each team has its own Excel file with a sheet for the given year.
    """
    # Ensure the directory for saving results exists
    ensure_dir("team_game_results")

    # Get all unique team names from the standings for the given year
    teams = set()
    try:
        divisions = standings(year)
        for division in divisions:
            teams.update(division['Tm'].tolist())
    except Exception as e:
        print(f"Error fetching standings for {year}: {str(e)}")
        return

    # Fetch and save game data for each team
    for team in tqdm(teams, desc=f"Fetching game data for {year}"):
        # Sanitize the team name to create a valid filename
        filename = f"team_game_results/{team.replace('/', '_').replace('.', '')}.xlsx"
        
        # Open an Excel writer for the team
        with pd.ExcelWriter(filename, mode='a') as writer:
            try:
                # Fetch the game-by-game schedule for the team
                df = schedule_and_record(year, team)
                
                # Check if the DataFrame is not empty
                if not df.empty:
                    # Write the data to the Excel file with a sheet for each year
                    df.to_excel(writer, sheet_name=str(year), index=False)
                    print(f"Saved game data for {team} in {year}")
                else:
                    print(f"No game data for {team} in {year}")
            except Exception as e:
                print(f"Error fetching data for {team} in {year}: {str(e)}")
            
            # Random sleep to mimic human behavior and avoid rate limiting
            time.sleep(random.uniform(3, 5))

def main():
    start_year = 1871  # Adjust this to your desired start year
    end_year = 2024    # Adjust this to your desired end year (or current year)
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching game-by-game results for {year}...")
        fetch_and_save_game_data_for_year(year)
        
        # Wait for 15 minutes before fetching the next year's data
        if year < end_year:
            print("Waiting for 15 minutes before fetching the next year's data...")
            time.sleep(900)  # Sleep for 15 minutes (900 seconds)
    
    print("Data fetching complete!")

if __name__ == "__main__":
    main()