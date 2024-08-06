import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
from load_data import load_and_filter_data

def custom_war_generator():
    st.subheader("Custom WAR Generator")

    # Explanation of WAR
    st.markdown("""
    ### What is WAR?
    WAR (Wins Above Replacement) is a comprehensive statistic that attempts to summarize a player's total contributions to their team in one statistic. It's calculated differently for position players and pitchers. As a general rule, a player who earns 0-1 WAR in a year is considered replacement level; 1-2 is a utility player, 2-3 is an average starter, 3-4 is an above average starter, 4-5 is an All-Star, 5-6 is a Superstar, and 6+ is MVP-level

    #### bWAR (Baseball-Reference WAR):
    - For position players: Combines batting, baserunning, and fielding value adjusted for position and league.
    - For pitchers: Based on runs allowed with adjustments for team defense, ballpark, and quality of opposition.

    #### fWAR (FanGraphs WAR):
    - For position players: Similar to bWAR but uses different defensive metrics and offensive weights.
    - For pitchers: Based on FIP (Fielding Independent Pitching) rather than runs allowed.

    Both versions aim to measure a player's value in terms of wins above what a replacement-level player would provide.

    This tool allows you to create your own version of WAR by adjusting the weights of various statistics. The point is, what do YOU think makes a player valuable? Note that the maximum WAR a player can earn in this model is 15, and the lowest is -5.
    
    ### Understanding the Weight Range (-1 to 1)
    
    The weight range from -1 to 1 determines how much each statistic contributes to the custom WAR calculation:
    
    - A weight of 1 means the stat has the maximum positive impact on the WAR calculation.
    - A weight of -1 means the stat has the maximum negative impact on the WAR calculation.
    - A weight of 0 means the stat has no impact on the WAR calculation.
    - Values between 0 and 1 (or 0 and -1) represent varying degrees of positive (or negative) impact.
    
    For example, if you set the weight for home runs (HR) to 0.5 and the weight for strikeouts (SO) to -0.3, 
    it means that home runs will have a moderate positive impact on a player's WAR, while strikeouts will have 
    a slightly smaller negative impact. The relative magnitudes of these weights determine how much each stat 
    influences the final WAR value.
    """)

    player_type = st.radio("Select player type:", ("Hitters", "Pitchers"))
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    data_df = load_and_filter_data(data_type)

    calculation_type = st.radio("Select calculation type:", ("Season-by-Season", "Career Average"))

    use_min_filter = st.checkbox("Set minimum playing time filter?")
    min_games = min_pa = min_ip = None
    if use_min_filter:
        col1, col2 = st.columns(2)
        with col1:
            min_games = st.number_input("Minimum games per season:", min_value=1, value=20)
        with col2:
            if data_type == "Hitter":
                min_pa = st.number_input("Minimum PA per season:", min_value=1, value=200)
            else:  # Pitcher
                min_ip = st.number_input("Minimum IP per season:", min_value=1, value=50)

    if data_type == "Hitter":
        default_stats = ['AVG', 'OBP', 'SLG', 'HR', 'RBI', 'SB', 'BB%', 'K%', 'ISO', 'wRC+', 'Def']
    else:  # Pitcher
        default_stats = ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'FIP', 'IP', 'W', 'SV']

    additional_stats = st.multiselect("Select additional stats to include:", 
                                      [col for col in data_df.columns if col not in default_stats + ['Name', 'Team', 'year', 'Age', 'IDfg']])
    
    all_stats = default_stats + additional_stats

    stat_toggles = {}
    for stat in all_stats:
        if stat in data_df.columns:
            stat_toggles[stat] = st.checkbox(f"Include {stat}", value=True)

    stats_to_use = [stat for stat in all_stats if stat_toggles.get(stat, False)]

    weights = {}
    for stat in stats_to_use:
        weight = st.number_input(f"Weight for {stat}", min_value=-1.0, max_value=1.0, value=0.1, step=0.0001, format="%.4f")
        weights[stat] = weight

    if st.button("Generate Custom WAR"):
        if not stats_to_use:
            st.warning("Please select at least one stat to include in the WAR calculation.")
            return

        if use_min_filter:
            if data_type == "Hitter":
                data_df = data_df[(data_df['G'] >= min_games) & (data_df['PA'] >= min_pa)]
            else:  # Pitcher
                data_df = data_df[(data_df['G'] >= min_games) & (data_df['IP'] >= min_ip)]

        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data_df[stats_to_use]), 
                                   columns=stats_to_use, index=data_df.index)

        custom_war = pd.Series(0, index=data_df.index)
        for stat, weight in weights.items():
            custom_war += scaled_data[stat] * weight

        custom_war = (custom_war - custom_war.min()) / (custom_war.max() - custom_war.min()) * 20 - 5

        data_df['Custom WAR'] = custom_war

        if calculation_type == "Career Average":
            career_war = data_df.groupby('Name').agg({
                'Custom WAR': 'mean',
                'year': ['min', 'max', 'count']
            }).reset_index()
            career_war.columns = ['Name', 'Custom WAR', 'First Year', 'Last Year', 'Years Played']
            data_df = career_war
        else:
            data_df['Years Played'] = 1
            data_df['First Year'] = data_df['year']
            data_df['Last Year'] = data_df['year']

        st.session_state.custom_war_data = data_df
        st.session_state.calculation_type = calculation_type

        st.success("Custom WAR calculated successfully. Use the date range slider below to view results.")

    if 'custom_war_data' in st.session_state:
        data_df = st.session_state.custom_war_data
        calculation_type = st.session_state.calculation_type

        if calculation_type == "Season-by-Season":
            min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
            top_players_start_year, top_players_end_year = st.slider(
                "Select the year range for top players:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            filtered_df = data_df[(data_df['year'] >= top_players_start_year) & (data_df['year'] <= top_players_end_year)]
        else:
            filtered_df = data_df

        st.subheader(f"Top Players by Custom WAR ({calculation_type})")
        top_players = filtered_df.sort_values('Custom WAR', ascending=False).head(20)
        st.table(top_players[['Name', 'Custom WAR', 'Years Played', 'First Year', 'Last Year']].round(2))

        fig = px.histogram(filtered_df, x='Custom WAR', nbins=50, 
                           title=f"Distribution of Custom WAR ({calculation_type})")
        st.plotly_chart(fig)

        if 'WAR' in filtered_df.columns:
            fig = px.scatter(filtered_df, x='WAR', y='Custom WAR', hover_data=['Name', 'year'],
                             title=f"Custom WAR vs Traditional WAR ({calculation_type})")
            st.plotly_chart(fig)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Custom WAR Data",
            data=csv,
            file_name=f"custom_war_data_{calculation_type.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )