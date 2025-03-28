import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

def generate_advanced_features(player_history, upcoming_match):
    """Generate advanced features based on research paper methodology"""
    
    # Create player feature dict
    player_features = {}
    
    # Basic features
    for player_name in player_history['player_name'].unique():
        player_data = player_history[player_history['player_name'] == player_name]
        
        if len(player_data) == 0:
            continue
            
        # Base features
        features = {
            'player_name': player_name,
            'team': player_data['team'].iloc[0],
            'role': player_data['role'].iloc[0],
            'matches_played': len(player_data),
            'avg_fantasy_points': player_data['fantasy_points'].mean(),
        }
        
        # Feature 1: Recent form - Exponentially weighted recent performance
        if len(player_data) >= 3:
            # Sort by recency
            recent_data = player_data.sort_values('match_date', ascending=False)
            
            # Apply exponential weights (more recent games matter more)
            weights = np.exp(-0.5 * np.arange(len(recent_data)))
            weights = weights / weights.sum()  # Normalize
            
            # Calculate weighted average
            features['weighted_recent_form'] = np.sum(recent_data['fantasy_points'] * weights[:len(recent_data)])
        else:
            features['weighted_recent_form'] = features['avg_fantasy_points']
            
        # Feature 2: Performance consistency (coefficient of variation)
        if len(player_data) >= 3:
            features['consistency'] = player_data['fantasy_points'].std() / (player_data['fantasy_points'].mean() + 1e-8)
        else:
            features['consistency'] = 0.5  # Average consistency if not enough data
        
        # Feature 3: Venue-specific performance
        venue_data = player_data[player_data['venue'] == upcoming_match['venue']]
        if len(venue_data) >= 2:
            features['venue_performance'] = venue_data['fantasy_points'].mean()
        else:
            features['venue_performance'] = features['avg_fantasy_points']
        
        # Feature 4: Opposition-specific performance
        opposition = upcoming_match['away_team'] if player_data['team'].iloc[0] == upcoming_match['home_team'] else upcoming_match['home_team']
        opposition_data = player_data[player_data['opponent'] == opposition]
        if len(opposition_data) >= 2:
            features['opposition_performance'] = opposition_data['fantasy_points'].mean()
        else:
            features['opposition_performance'] = features['avg_fantasy_points']
        
        # Feature 5: Role-specific performance metrics
        role = player_data['role'].iloc[0]
        if role == 'BAT':
            # Batting metrics
            features['batting_impact'] = features['avg_fantasy_points'] * (1 - features['consistency'])
        elif role == 'BOWL':
            # Bowling metrics
            features['bowling_impact'] = features['avg_fantasy_points'] * (1 - features['consistency'])
        elif role == 'AR':
            # All-rounder combined impact
            features['all_round_impact'] = features['avg_fantasy_points'] * 1.1  # Bonus for versatility
        else:  # WK
            # Wicket-keeper impact
            features['wk_impact'] = features['avg_fantasy_points'] * 0.95
        
        # Feature 6: Form trend (increasing/decreasing)
        if len(player_data) >= 5:
            recent_5 = player_data.sort_values('match_date', ascending=False).head(5)
            x = np.arange(len(recent_5))
            y = recent_5['fantasy_points'].values
            
            # Simple linear regression to detect trend
            slope = np.polyfit(x, y, 1)[0]
            features['form_trend'] = slope
        else:
            features['form_trend'] = 0
            
        # Feature 7: Match importance factor (playoff/final matches get higher importance)
        # Here we're simulating a playoff importance based on match date
        match_date = pd.to_datetime(upcoming_match['date'])
        
        # Assume playoffs are in May
        if match_date.month == 5:
            features['match_importance'] = 1.2
        else:
            features['match_importance'] = 1.0
            
        # Store features
        player_features[player_name] = features
        
    return pd.DataFrame(list(player_features.values()))

def apply_pitch_and_weather_adjustments(players_df, upcoming_match):
    """Apply pitch condition and weather adjustments based on research"""
    
    venue = upcoming_match.get('venue', '')
    match_date = upcoming_match.get('date', '')
    
    # Pitch characteristics (based on research/historical data)
    pitch_characteristics = {
        'Wankhede Stadium': {
            'batting_friendly': 0.8,  # 0-1 scale (1 = very batting friendly)
            'spin_friendly': 0.3,     # 0-1 scale (1 = very spin friendly)
            'pace_friendly': 0.7,     # 0-1 scale (1 = very pace friendly)
        },
        'M. A. Chidambaram Stadium': {
            'batting_friendly': 0.4,
            'spin_friendly': 0.9,
            'pace_friendly': 0.2,
        },
        'Eden Gardens': {
            'batting_friendly': 0.6,
            'spin_friendly': 0.5,
            'pace_friendly': 0.6,
        },
        # Add more venues as needed
    }
    
    # Default values if venue not found
    pitch_info = pitch_characteristics.get(venue, {
        'batting_friendly': 0.5,
        'spin_friendly': 0.5,
        'pace_friendly': 0.5,
    })
    
    # Weather information (would be fetched from API in production)
    # Simulating weather data here
    weather_info = {
        'is_rainy': False,
        'humidity': 0.5,  # 0-1 scale
        'wind_speed': 0.3,  # 0-1 scale
    }
    
    # Apply adjustments to each player
    for idx, player in players_df.iterrows():
        role = player['role']
        base_points = players_df.at[idx, 'predicted_points']
        
        # Factor 1: Pitch Adjustments
        pitch_adjustment = 1.0
        if role == 'BAT':
            # Batsmen perform better on batting-friendly pitches
            pitch_adjustment *= (1 + (pitch_info['batting_friendly'] - 0.5) * 0.3)
        elif role == 'BOWL':
            # Bowlers perform better on bowling-friendly (less batting-friendly) pitches
            pitch_adjustment *= (1 + (0.7 - pitch_info['batting_friendly']) * 0.3)
            
            # Check for spin/pace bowler (would need more detailed player data)
            # Simulating by alternating players
            is_spin_bowler = (idx % 2 == 0)
            if is_spin_bowler:
                pitch_adjustment *= (1 + (pitch_info['spin_friendly'] - 0.5) * 0.2)
            else:
                pitch_adjustment *= (1 + (pitch_info['pace_friendly'] - 0.5) * 0.2)
                
        elif role == 'AR':
            # All-rounders get partial adjustments for both batting and bowling
            bat_adj = (1 + (pitch_info['batting_friendly'] - 0.5) * 0.15)
            bowl_adj = (1 + (0.7 - pitch_info['batting_friendly']) * 0.15)
            pitch_adjustment *= (bat_adj + bowl_adj) / 2
            
        # Factor 2: Weather Adjustments
        weather_adjustment = 1.0
        if weather_info['is_rainy']:
            # Rain typically helps bowlers, hurts batsmen
            if role in ['BOWL', 'AR']:
                weather_adjustment *= 1.1
            elif role == 'BAT':
                weather_adjustment *= 0.9
        
        if weather_info['humidity'] > 0.7:
            # High humidity helps swing bowlers
            if role == 'BOWL':
                # Assume every other bowler is a swing bowler (simplified)
                if idx % 2 == 0:
                    weather_adjustment *= 1.05
        
        if weather_info['wind_speed'] > 0.6:
            # High wind can affect batting precision
            if role == 'BAT':
                weather_adjustment *= 0.95
        
        # Apply combined adjustments
        final_adjustment = pitch_adjustment * weather_adjustment
        players_df.at[idx, 'predicted_points'] = base_points * final_adjustment
    
    return players_df

def enhance_player_predictions(players_df, upcoming_match):
    """
    Enhanced player predictions using research paper methodology
    """
    print(f"Enhancing predictions using research methodology for {upcoming_match['home_team']} vs {upcoming_match['away_team']}")
    
    # Create directory for storing historical data if it doesn't exist
    os.makedirs('data/historical', exist_ok=True)
    
    # Map team abbreviations to full names
    team_mapping = {
        'CHE': 'Chennai Super Kings',
        'DC': 'Delhi Capitals',
        'GT': 'Gujarat Titans',
        'KKR': 'Kolkata Knight Riders',
        'LSG': 'Lucknow Super Giants',
        'MI': 'Mumbai Indians',
        'PBKS': 'Punjab Kings',
        'RCB': 'Royal Challengers Bangalore',
        'RR': 'Rajasthan Royals',
        'SRH': 'Sunrisers Hyderabad'
    }
    
    # Get team abbreviations for home/away teams
    home_team = upcoming_match['home_team']
    home_abbr = next((abbr for abbr, name in team_mapping.items() if name == home_team), home_team)
    
    # Step 1: Add columns for recent form if they don't exist
    if 'recent_form' not in players_df.columns:
        players_df['recent_form'] = 0.0
    if 'last_5_matches' not in players_df.columns:
        players_df['last_5_matches'] = ""  # Store as string instead of list
    if 'ground_performance' not in players_df.columns:
        players_df['ground_performance'] = 0.0
    
    # Step 2: Load or create historical performance data
    historical_data_path = 'data/historical/player_history.csv'
    if os.path.exists(historical_data_path):
        historical_df = pd.read_csv(historical_data_path)
        print(f"Loaded historical data for {len(historical_df['player_name'].unique())} players")
    else:
        # Create sample historical data based on player credits
        # In a real system, you'd scrape this from websites or use an API
        historical_df = create_sample_historical_data(players_df)
        historical_df.to_csv(historical_data_path, index=False)
        print(f"Created sample historical data for {len(historical_df['player_name'].unique())} players")
    
    # Step 3: Calculate recent form metrics for each player
    for idx, player in players_df.iterrows():
        player_name = player['player_name']
        
        # Get player's match history
        player_history = historical_df[historical_df['player_name'] == player_name]
        
        if len(player_history) > 0:
            # Sort by date (most recent first)
            player_history = player_history.sort_values('match_date', ascending=False)
            
            # Recent form: Last 5 matches average
            last_5 = player_history.head(5)
            if len(last_5) > 0:
                players_df.at[idx, 'recent_form'] = last_5['fantasy_points'].mean()
                # FIX: Store as JSON string instead of list
                players_df.at[idx, 'last_5_matches'] = json.dumps(last_5['fantasy_points'].tolist()[:5])
            
            # Ground performance: Average at the upcoming venue
            venue = upcoming_match['venue']
            venue_history = player_history[player_history['venue'] == venue]
            if len(venue_history) > 0:
                players_df.at[idx, 'ground_performance'] = venue_history['fantasy_points'].mean()
            else:
                # No data for this ground, use overall average
                players_df.at[idx, 'ground_performance'] = player_history['fantasy_points'].mean()
        else:
            # No historical data, estimate based on player credits
            # Higher credit players tend to be more consistent
            base_points = player['cost'] * 5
            players_df.at[idx, 'recent_form'] = base_points * (0.8 + 0.4 * np.random.random())
            players_df.at[idx, 'ground_performance'] = base_points * (0.7 + 0.6 * np.random.random())
            players_df.at[idx, 'last_5_matches'] = json.dumps([])  # Empty list as JSON string
    
    # NEW: Generate advanced features from historical data
    advanced_features = generate_advanced_features(historical_df, upcoming_match)
    
    # NEW: Merge advanced features with players_df
    if advanced_features is not None and len(advanced_features) > 0:
        players_df = players_df.merge(
            advanced_features[['player_name', 'weighted_recent_form', 'consistency', 
                             'venue_performance', 'opposition_performance', 'form_trend', 
                             'match_importance']],
            on='player_name', how='left'
        )
    
    # Fill missing advanced features
    for col in ['weighted_recent_form', 'consistency', 'venue_performance', 
               'opposition_performance', 'form_trend', 'match_importance']:
        if col in players_df.columns:
            players_df[col] = players_df[col].fillna(0)
    
    # Rest of your function continues...
    # (remaining code for ML prediction, final formula, etc.)
    
    # Create feature matrix for ML prediction
    if 'weighted_recent_form' in players_df.columns:
        # Define features for ML prediction
        feature_cols = ['cost', 'weighted_recent_form', 'consistency', 
                      'venue_performance', 'opposition_performance', 
                      'form_trend', 'match_importance']
        
        # Use ML model for prediction if available
        try:
            from ml_engine import predict_player_performance
            # Convert categorical features to numeric
            players_df_ml = players_df.copy()
            
            # Generate one-hot encoding for role
            role_dummies = pd.get_dummies(players_df_ml['role'], prefix='role')
            players_df_ml = pd.concat([players_df_ml, role_dummies], axis=1)
            
            # Generate one-hot encoding for team
            team_dummies = pd.get_dummies(players_df_ml['team'], prefix='team')
            players_df_ml = pd.concat([players_df_ml, team_dummies], axis=1)
            
            # Add encoded columns to feature list
            for col in players_df_ml.columns:
                if col.startswith('role_') or col.startswith('team_'):
                    feature_cols.append(col)
            
            # Use ML model for prediction
            ml_predictions = predict_player_performance(players_df_ml, feature_cols)
            
            if ml_predictions is not None:
                players_df['ml_predicted_points'] = ml_predictions
                print("Successfully applied ML predictions")
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            players_df['ml_predicted_points'] = players_df['cost'] * 5 * 0.9  # Fallback
    
    # Final prediction formula based on research paper
    # This weighting combines ML predictions with domain knowledge
    for idx, player in players_df.iterrows():
        # Base formula using various factors (research paper approach)
        if 'ml_predicted_points' in players_df.columns:
            # Use ML predictions with expert knowledge adjustments
            players_df.at[idx, 'predicted_points'] = (
                0.5 * player['ml_predicted_points'] +
                0.2 * player.get('weighted_recent_form', player['cost'] * 5) +
                0.15 * player.get('venue_performance', player['cost'] * 5) +
                0.1 * player.get('opposition_performance', player['cost'] * 5) +
                0.05 * player.get('form_trend', 0) * 10  # Scale trend impact
            )
        else:
            # Use weighted formula when ML is not available
            players_df.at[idx, 'predicted_points'] = (
                0.4 * player.get('weighted_recent_form', player['cost'] * 5) +
                0.3 * player.get('venue_performance', player['cost'] * 5) +
                0.2 * player.get('opposition_performance', player['cost'] * 5) +
                0.1 * player.get('form_trend', 0) * 10  # Scale trend impact
            )
        
        # Apply match importance factor
        if 'match_importance' in players_df.columns:
            players_df.at[idx, 'predicted_points'] *= player.get('match_importance', 1.0)
    
    # Apply pitch and weather adjustments
    players_df = apply_pitch_and_weather_adjustments(players_df, upcoming_match)
    
    print(f"Enhanced predictions for {len(players_df)} players using research methodology")
    return players_df

def create_sample_historical_data(players_df):
    """Create sample historical match data for players based on their quality (credits)"""
    historical_data = []
    
    # All IPL venues
    venues = [
        'Wankhede Stadium', 'M. A. Chidambaram Stadium', 'Eden Gardens',
        'Arun Jaitley Stadium', 'Narendra Modi Stadium', 'Rajiv Gandhi International Stadium',
        'Punjab Cricket Association Stadium', 'M. Chinnaswamy Stadium',
        'Sawai Mansingh Stadium', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium'
    ]
    
    # Teams
    teams = ['MI', 'CSK', 'RCB', 'KKR', 'DC', 'PBKS', 'RR', 'SRH', 'GT', 'LSG']
    
    # Generate 2 years of IPL matches (2023, 2024)
    start_date = datetime(2023, 3, 31)
    end_date = datetime(2024, 5, 28)
    
    # Generate match dates
    match_dates = []
    current_date = start_date
    while current_date <= end_date:
        # IPL matches typically happen almost daily during the season
        # Add some matches to the list (March-May each year)
        if (3 <= current_date.month <= 5):
            match_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    # For each player, generate historical performances
    for _, player in players_df.iterrows():
        player_name = player['player_name']
        role = player['role']
        team = player['team']
        credit = player['cost']
        
        # Players with higher credits play more matches and score more consistently
        match_count = int(10 + credit * 2)  # Higher credit players play more matches
        base_points = credit * 5  # Base points related to player quality
        
        # Generate match history
        for _ in range(min(match_count, len(match_dates))):
            # Random match from the schedule
            match_idx = np.random.randint(0, len(match_dates))
            match_date = match_dates[match_idx]
            
            # Random opponent
            opponent = teams[np.random.randint(0, len(teams))]
            while opponent == team:  # Make sure opponent is different
                opponent = teams[np.random.randint(0, len(teams))]
            
            # Random venue
            venue = venues[np.random.randint(0, len(venues))]
            
            # Generate fantasy points based on player quality and role
            # Add variations for realism
            
            # Base prediction
            consistency = 0.7 + (credit / 20)  # Higher credits are more consistent
            base_prediction = base_points * (consistency - 0.5 + np.random.random())
            
            # Role-specific variance
            if role == 'BAT':
                # Batsmen: sometimes get ducks, sometimes score big
                variance = np.random.choice([0.2, 0.8, 1.0, 1.2, 1.5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            elif role == 'BOWL':
                # Bowlers: somewhat consistent
                variance = np.random.choice([0.5, 0.8, 1.0, 1.2, 1.4], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            elif role == 'AR':
                # All-rounders: high variance
                variance = np.random.choice([0.3, 0.7, 1.0, 1.3, 1.7], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            else:  # WK
                # Wicket-keepers: consistent but lower ceiling
                variance = np.random.choice([0.6, 0.8, 1.0, 1.1, 1.3], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            
            # Apply variance to base prediction
            fantasy_points = base_prediction * variance
            
            # Add to historical data
            historical_data.append({
                'player_name': player_name,
                'team': team,
                'role': role,
                'match_date': match_date,
                'opponent': opponent,
                'venue': venue,
                'fantasy_points': fantasy_points
            })
    
    return pd.DataFrame(historical_data)

def optimize_team_synergy(selected_players, player_stats, budget_remaining):
    """Optimize final team based on player correlations and synergy (research paper methodology)"""
    
    # If no player stats or less than 2 players selected, return original selection
    if len(selected_players) < 2 or player_stats is None:
        return selected_players  # Return the original selection if conditions are not met
    
    print("Optimizing team synergy based on player correlations...")
    
    # Create a list of player names already selected
    selected_player_names = [p['player_name'] for p in selected_players]
    
    # Identify potential replacement candidates
    replacement_candidates = player_stats[~player_stats['player_name'].isin(selected_player_names)].copy()
    
    # Calculate average points per credit for selected team
    selected_value = sum(p.get('predicted_points', 0) for p in selected_players) / \
                    sum(p.get('cost', 0) for p in selected_players)
    
    # Calculate correlation of each potential replacement with already selected players
    team_synergy_score = {}
    
    # For each selected player position, try potential replacements
    for i, current_player in enumerate(selected_players):
        current_role = current_player['role']
        current_team = current_player['team']
        current_points = current_player.get('predicted_points', 0)
        current_cost = current_player.get('cost', 0)
        
        # Get potential replacements for this role
        potential_replacements = replacement_candidates[
            (replacement_candidates['role'] == current_role) & 
            (replacement_candidates['cost'] <= current_cost + budget_remaining)
        ]
        
        for _, replacement in potential_replacements.iterrows():
            # Skip if already suggested as replacement
            if replacement['player_name'] in team_synergy_score:
                continue
                
            replacement_dict = replacement.to_dict()
            
            # Calculate synergy improvement
            # 1. Check if replacement is from same team as another selected player
            team_synergy = sum(1 for p in selected_players if p['team'] == replacement['team'] and p != current_player)
            
            # 2. Value improvement
            value_change = (replacement.get('predicted_points', 0) / replacement.get('cost', 100)) - (current_points / current_cost)
            
            # 3. Calculate opposition advantage (batsmen vs weak bowling teams, etc.)
            opposition_advantage = 0
            match_team1 = selected_players[0].get('team', '')
            match_team2 = selected_players[-1].get('team', '')
            if match_team1 != match_team2:  # Ensure we have proper team information
                if replacement['role'] == 'BAT' and replacement['team'] == match_team1:
                    # Bonus for batsmen playing against certain teams known for weak bowling
                    opposition_advantage = 0.1
                elif replacement['role'] == 'BOWL' and replacement['team'] == match_team2:
                    # Bonus for bowlers playing against certain teams known for weak batting
                    opposition_advantage = 0.1
            
            # Combined synergy score - weights based on research paper
            synergy_score = (
                0.4 * value_change +
                0.3 * team_synergy +
                0.3 * opposition_advantage
            )
            
            # Store the synergy score
            team_synergy_score[(i, replacement['player_name'])] = {
                'score': synergy_score,
                'player': replacement_dict,
                'replace_idx': i,
                'cost': replacement['cost'],
                'points': replacement.get('predicted_points', 0)
            }
    
    # Sort by synergy score and make replacements
    replacements = sorted(team_synergy_score.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Implement top replacements that improve team
    replacement_indices = set()
    for (_, replacement_name), replacement_info in replacements:
        idx = replacement_info['replace_idx']
        
        # Skip if we already replaced this position
        if idx in replacement_indices:
            continue
            
        # Check if this replacement improves the team
        if replacement_info['score'] > 0:
            # Replace the player
            current_cost = selected_players[idx]['cost']
            budget_change = current_cost - replacement_info['cost']
            
            if budget_change >= 0:  # Only if it doesn't exceed budget
                selected_players[idx] = replacement_info['player']
                budget_remaining += budget_change
                replacement_indices.add(idx)
                print(f"Replaced {selected_players[idx]['player_name']} with {replacement_name} (Synergy improvement: {replacement_info['score']:.3f})")
        
        # Stop after 2-3 replacements
        if len(replacement_indices) >= 3:
            break
    
    return selected_players

def select_enhanced_fantasy_team(upcoming_match, budget=100.0, captain_bonus=2.0, vice_captain_bonus=1.5):
    """Fantasy team selection with enhanced predictions"""
    # First, import team_selection module
    from team_selection import select_fantasy_team
    
    # Load Dream11 player data
    players_df = pd.read_csv('data/player_data/dream11_players.csv', header=0)
    players_df = players_df.iloc[:, :4]
    players_df.columns = ['cost', 'role', 'player_name', 'team']
    
    # Remove empty rows and convert cost to numeric
    players_df = players_df[players_df['player_name'].notna() & (players_df['player_name'] != '')]
    players_df['cost'] = pd.to_numeric(players_df['cost'], errors='coerce')
    
    # Standardize role names
    role_mapping = {'WK': 'WK', 'BAT': 'BAT', 'ALL': 'AR', 'BOWL': 'BOWL'}
    players_df['role'] = players_df['role'].map(role_mapping)
    
    # Enhance predictions with form and ground performance
    enhanced_df = enhance_player_predictions(players_df, upcoming_match)
    
    # Save the enhanced predictions
    enhanced_df.to_csv('data/player_data/enhanced_dream11_players.csv', index=False)
    
    # Now use the regular team selection with enhanced predictions
    print("Using enhanced predictions for Dream11 team selection")
    return select_fantasy_team(upcoming_match, budget, captain_bonus, vice_captain_bonus)

if __name__ == "__main__":
    # Test with upcoming match
    test_match = {
        'match_id': 1,
        'home_team': 'Chennai Super Kings',
        'away_team': 'Royal Challengers Bangalore',
        'venue': 'M. A. Chidambaram Stadium',
        'date': '2025-04-01'
    }
    
    # Generate enhanced predictions and select team
    selected_team = select_enhanced_fantasy_team(test_match)
    print("Fantasy team selection complete with enhanced predictions!")
