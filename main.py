#base imports
import os
import pandas as pd
import numpy as np
import random
import streamlit as st
#plotting and stats
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
#baseball functions
from load_data import load_and_filter_data, load_and_prepare_data
from colors import load_team_colors, get_team_color
from player_view import display_player_stats, individual_player_view, compare_players_view
from race import process_data_for_race, create_race_plot, race_chart_view
from histogram import plot_league_wide_stat, create_hover_text, league_wide_stats_view
from similarity import calculate_similarity_scores, player_similarity_view
from make_war import custom_war_generator
from milestone_tracker import milestone_tracker
from goat import how_is_he_the_goat
from bangbang import generate_astros_cheating_fact, display_astros_cheating_fact


def main():
    st.title("Brock's Baseball Stats Explorer")
    
    st.header("It's Time For Dodger Baseball! (or your team who sucks)")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the mode",
                                ["Individual Player", "Compare Players", "Historical Histogram", "Milestone Tracker",
                                 "Career Stat Race", "Player Similarity", "Custom WAR Generator",
                                 "How is he the GOAT?"])
    
    # Reset session state when changing modes
    if 'current_mode' not in st.session_state or st.session_state.current_mode != app_mode:
        st.session_state.player_data = None
        st.session_state.current_mode = app_mode
    
    if app_mode == "Individual Player":
        individual_player_view()
    elif app_mode == "Compare Players":
        compare_players_view()
    elif app_mode == "Historical Histogram":
        league_wide_stats_view()
    elif app_mode == "Career Stat Race":
        race_chart_view()
    elif app_mode == "Player Similarity":
        player_similarity_view()
    elif app_mode == "Custom WAR Generator":
        custom_war_generator()
    elif app_mode == "Milestone Tracker":
        milestone_tracker()
    else:  # How is he the GOAT?
        how_is_he_the_goat()
    
    # Display the Astros cheating fact at the bottom of each page
    display_astros_cheating_fact()

if __name__ == "__main__":
    main()