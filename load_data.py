import os
import pandas as pd
import streamlit as st
from typing import List, Union, Optional, Tuple

@st.cache_data
def load_and_filter_data(data_type, player_names_or_ids=None):
    data_dir = 'hitter_data' if data_type == "Hitter" else 'pitcher_data'
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            year = filename.split('_')[-1].split('.')[0]
            file_path = os.path.join(data_dir, filename)
            data = pd.read_csv(file_path)
            data['year'] = int(year)
            data['player_type'] = data_type.lower()
            all_data.append(data)
    
    full_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    if player_names_or_ids:
        filtered_data = full_data[
            full_data['Name'].isin(player_names_or_ids) | 
            full_data['IDfg'].astype(str).isin(player_names_or_ids)
        ]
        return filtered_data if not filtered_data.empty else pd.DataFrame()
    else:
        return full_data

def load_and_prepare_data(data_type):
    data_df = load_and_filter_data(data_type)
    
    # Replace NaN values in IDfg with a placeholder (e.g., -1)
    data_df['IDfg'] = data_df['IDfg'].fillna(-1)
    
    # Convert IDfg to integer
    data_df['IDfg'] = data_df['IDfg'].astype(int)
    
    # Calculate first and last year for each player
    player_years = data_df.groupby('IDfg').agg({
        'Name': 'first',
        'year': ['min', 'max']
    }).reset_index()
    player_years.columns = ['IDfg', 'Name', 'FirstYear', 'LastYear']
    
    # Create unique labels for each player
    player_years['Label'] = player_years.apply(lambda row: f"{row['Name']} ({row['FirstYear']}-{row['LastYear']})", axis=1)
    
    return data_df, player_years