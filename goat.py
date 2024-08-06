import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from typing import Dict, List
from load_data import load_and_prepare_data

def how_is_he_the_goat():
    st.subheader("How is he the GOAT?")

    st.markdown("""
    This tool attempts to find the optimal weights for a selected set of baseball statistics that would make a chosen player the Greatest of All Time (GOAT). Here's how it works:
    
    1. You select a player and a time frame.
    2. The tool considers a predefined set of key stats for batters or pitchers.
    3. It then calculates weights for these stats that would rank your chosen player as the best among all players.
    4. If successful, it shows the weights and the resulting player rankings.
    5. If unsuccessful, it informs you that it's impossible to make that player the GOAT with the given data.

    Note: Some statistics are inversely weighted in the calculation. For batters, this includes stats like Strikeouts (SO) and Ground into Double Play (GDP), where lower values are better. For pitchers, this includes ERA, Walks (BB), and Hits Allowed (H), among others. The tool automatically adjusts for these "negative" stats in its calculations.

    This is a mathematical exercise and doesn't necessarily reflect real-world value. It's designed to explore what aspects of a player's performance would need to be emphasized to consider them the greatest.
    """)

    player_type = st.radio("Select player type:", ("Hitters", "Pitchers"), key="goat_player_type_radio")
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    data_df, player_years = load_and_prepare_data(data_type)

    # Year range selection
    min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
    year_range = st.slider("Select year range:", min_year, max_year, (min_year, max_year), key="goat_year_range_slider")
    data_df = data_df[(data_df['year'] >= year_range[0]) & (data_df['year'] <= year_range[1])]

    # Player selection with defaults
    default_player = "Shohei Ohtani" if data_type == "Hitter" else "Clayton Kershaw"
    
    # Check if the default player is in the dataset, if not use the first player in the list
    if default_player not in player_years['Name'].unique():
        default_player = player_years['Name'].iloc[0]
    
    # Create a dictionary mapping player labels to their IDfg
    player_label_to_id = dict(zip(player_years['Label'], player_years['IDfg']))
    
    selected_player_label = st.selectbox(
        "Select a player:",
        player_years['Label'],
        index=player_years['Label'].tolist().index(player_years[player_years['Name'] == default_player]['Label'].iloc[0]),
        key="goat_player_selectbox"
    )
    
    selected_player_id = player_label_to_id[selected_player_label]

    # Define stats to use based on player type
    if data_type == "Hitter":
        stats_to_use = ['G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'AVG', 'GB', 'FB', 'LD', 'IFFB', 'Pitches', 'Balls', 'Strikes', 'IFH', 'BB%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%', 'Fld', 'Spd']
        # Stats where higher is worse
        negative_stats = ['GDP', 'CS', 'SO', 'K%']
    else:  # Pitcher
        stats_to_use = ['W', 'L', 'ERA', 'G', 'GS', 'CG', 'ShO', 'SV', 'BS', 'IP', 'TBF', 'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'SO', 'GB', 'FB', 'LD', 'IFFB', 'Balls', 'Strikes', 'Pitches', 'RS', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 'FIP']
        # Stats where higher is worse
        negative_stats = ['L', 'ERA', 'R', 'ER', 'HR', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'BB/9', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'FIP']

    # Filter stats that are actually in the dataframe
    stats_to_use = [stat for stat in stats_to_use if stat in data_df.columns]

    if st.button("Find GOAT Weights"):
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data_df[stats_to_use]), 
                                   columns=stats_to_use, index=data_df.index)

        # Invert negative stats
        for stat in negative_stats:
            if stat in scaled_data.columns:
                scaled_data[stat] = 1 - scaled_data[stat]

        # Group by player (using IDfg) and take the mean of their stats
        player_stats = scaled_data.groupby(data_df['IDfg']).mean()

        # Handle missing data for the selected player
        selected_player_stats = player_stats.loc[selected_player_id]
        available_stats = selected_player_stats.dropna().index
        player_stats = player_stats[available_stats]

        # Function to optimize
        def objective(weights):
            war = (player_stats * weights).sum(axis=1)
            player_war = war[selected_player_id]
            return np.sum(np.maximum(0, war - player_war))  # Sum of how much others exceed the player

        # Constraints: sum of absolute weights = 1
        def constraint(weights):
            return np.sum(np.abs(weights)) - 1

        # Initial guess
        initial_weights = np.ones(len(available_stats)) / len(available_stats)

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                          constraints={'type': 'eq', 'fun': constraint},
                          options={'ftol': 1e-10, 'maxiter': 1000})

        if result.success and result.fun < 1e-5:  # Check if optimization succeeded and player is at the top
            optimal_weights = result.x
            
            # Calculate WAR with optimal weights
            war = (player_stats * optimal_weights).sum(axis=1)
            player_rankings = war.sort_values(ascending=False)
            
            st.success(f"Optimal weights found to make {selected_player_label} the GOAT!")
            
            # Display weights in ranked order
            st.subheader("Optimal Weights (Ranked by Importance):")
            weight_df = pd.DataFrame({'Stat': available_stats, 'Weight': optimal_weights})
            weight_df['Abs_Weight'] = np.abs(weight_df['Weight'])
            weight_df = weight_df.sort_values('Abs_Weight', ascending=False)
            for _, row in weight_df.iterrows():
                if row['Abs_Weight'] > 1e-4:  # Only show weights that are not essentially zero
                    st.write(f"{row['Stat']}: {row['Weight']:.4f}")
            
            # Display player ranking
            st.subheader("Top 10 Players with these weights:")
            top_10_players = player_rankings.head(10)
            top_10_labels = player_years.set_index('IDfg').loc[top_10_players.index, 'Label']
            st.table(pd.DataFrame({'Player': top_10_labels, 'Score': top_10_players}))
            
            # Show selected player's rank
            player_rank = player_rankings.index.get_loc(selected_player_id) + 1
            st.write(f"{selected_player_label}'s rank with these weights: {player_rank}")

            # Generate explanation
            top_positive_stats = weight_df[weight_df['Weight'] > 0].nlargest(3, 'Abs_Weight')['Stat'].tolist()
            top_negative_stats = weight_df[weight_df['Weight'] < 0].nlargest(2, 'Abs_Weight')['Stat'].tolist()
            
            explanation = f"{selected_player_label} is considered the GOAT in this analysis primarily due to "
            explanation += f"their exceptional performance in {', '.join(top_positive_stats[:-1])}, and {top_positive_stats[-1]}. "
            if top_negative_stats:
                explanation += f"The model also values their ability to minimize {' and '.join(top_negative_stats)}. "
            explanation += "This combination of strengths sets them apart in this particular weighting scheme."
            
            st.subheader("Explanation:")
            st.write(explanation)
            
            # Display stats used
            st.subheader("Stats Used in Calculation:")
            st.write(", ".join(available_stats))
            
        else:
            st.error(f"Sadly, it is impossible for {selected_player_label} to be the GOAT with the given stats and time frame.")
            st.write("Try adjusting the year range or selecting a different player.")