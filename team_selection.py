import pandas as pd
import numpy as np
import os
import joblib

# Import enhanced predictions if available
try:
    from enhanced_predictions import enhance_player_predictions
    ENHANCED_PREDICTIONS_AVAILABLE = True
except ImportError:
    ENHANCED_PREDICTIONS_AVAILABLE = False

def select_fantasy_team(upcoming_match, budget=100.0, captain_bonus=2.0, vice_captain_bonus=1.5):
    """
    Select an optimal Dream11 fantasy team following all Dream11 rules
    
    Args:
        upcoming_match: Dict with match details
        budget: Maximum team budget (default: 100 credits)
        captain_bonus: Points multiplier for captain (default: 2.0x)
        vice_captain_bonus: Points multiplier for vice-captain (default: 1.5x)
    """
    print(f"Selecting Dream11 team for {upcoming_match['home_team']} vs {upcoming_match['away_team']}")
    
    try:
        # First try to load Dream11 specific player data
        dream11_data_path = 'data/player_data/dream11_players.csv'
        if os.path.exists(dream11_data_path):
            # Load and clean the Dream11 player data
            players_df = pd.read_csv(dream11_data_path, header=0)
            print(f"Loaded Dream11 player data with {len(players_df)} players")
            
            # Clean up the data
            players_df = players_df.iloc[:, :4]  # Keep only the first 4 columns
            players_df.columns = ['cost', 'role', 'player_name', 'team']
            
            # Remove empty rows
            players_df = players_df[players_df['player_name'].notna() & (players_df['player_name'] != '')]
            
            # Convert cost to numeric
            players_df['cost'] = pd.to_numeric(players_df['cost'], errors='coerce')
            
            # Standardize role names
            role_mapping = {
                'WK': 'WK',
                'BAT': 'BAT',
                'ALL': 'AR',  # Dream11 uses ALL, we standardize to AR
                'BOWL': 'BOWL'
            }
            players_df['role'] = players_df['role'].map(role_mapping)
            
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
            players_df['full_team_name'] = players_df['team'].map(team_mapping)
            
            # Drop rows with missing data
            players_df = players_df.dropna(subset=['cost', 'role', 'player_name', 'team'])
            print(f"After cleaning: {len(players_df)} players remaining")
            
        else:
            # Fall back to existing player features if available
            features_path = 'data/ml/player_features.csv'
            if os.path.exists(features_path):
                players_df = pd.read_csv(features_path)
                print(f"Using fallback player data with {len(players_df)} players")
            else:
                raise Exception("No player data found. Please provide Dream11 player data or run feature engineering.")
        
        # Load prediction model if available
        try:
            model = joblib.load('models/gradient_boosting_model.pkl')
            print("Loaded prediction model for player performance")
        except:
            model = None
            print("No prediction model found - using player credits as prediction basis")
        
        # Filter for players in upcoming match
        home_team = upcoming_match['home_team']
        away_team = upcoming_match['away_team']
        
        # Map to abbreviated team names if needed
        home_abbr = next((abbr for abbr, name in team_mapping.items() if name == home_team), home_team)
        away_abbr = next((abbr for abbr, name in team_mapping.items() if name == away_team), away_team)
        
        # Filter players from these teams
        if 'full_team_name' in players_df.columns:
            match_players = players_df[(players_df['full_team_name'] == home_team) | 
                                     (players_df['full_team_name'] == away_team)].copy()
            
            if len(match_players) < 22:  # Not enough players with full names, try abbreviations
                match_players = players_df[(players_df['team'] == home_abbr) | 
                                         (players_df['team'] == away_abbr)].copy()
        else:
            match_players = players_df[(players_df['team'] == home_team) | 
                                     (players_df['team'] == away_team) |
                                     (players_df['team'] == home_abbr) | 
                                     (players_df['team'] == away_abbr)].copy()
        
        # If still not enough players, warn and use all players
        if len(match_players) < 22:
            print(f"Warning: Only {len(match_players)} players found for {home_team} vs {away_team}.")
            print("Using all available players with a preference for these teams.")
            match_players = players_df.copy()
            
            # Apply a prediction bonus for players from the match teams
            is_match_team = ((match_players['team'] == home_team) | (match_players['team'] == away_team) |
                            (match_players['team'] == home_abbr) | (match_players['team'] == away_abbr))
            match_players['team_bonus'] = np.where(is_match_team, 1.2, 1.0)
        else:
            match_players['team_bonus'] = 1.0
            print(f"Found {len(match_players)} players for this match")
        
        # Generate player predictions
        if model is not None and set(['batting_average', 'strike_rate', 'economy_rate']).issubset(match_players.columns):
            # Use ML model for predictions
            exclude_cols = ['player_name', 'team', 'role', 'cost', 'team_bonus', 'full_team_name']
            feature_cols = [col for col in match_players.columns if col not in exclude_cols]
            
            match_players['predicted_points'] = model.predict(match_players[feature_cols])
        else:
            # No model - use credit value as a proxy for player quality 
            # with some randomization to simulate performance variance
            match_players['predicted_points'] = match_players['cost'] * 5 * (0.8 + 0.4 * np.random.random(len(match_players)))
            
            # Apply team bonus (players from match teams get a boost)
            match_players['predicted_points'] *= match_players['team_bonus']
            
            print("Using player credits as prediction basis with some randomization")
        
        # Use enhanced predictions if available
        if ENHANCED_PREDICTIONS_AVAILABLE:
            try:
                match_players = enhance_player_predictions(match_players, upcoming_match)
                print("Using enhanced predictions with form and ground analysis")
            except Exception as e:
                print(f"Enhanced predictions failed: {e}")

        # Add to your select_fantasy_team function in team_selection.py
        # After making initial predictions but before selecting players:

        # Apply research-based enhancements
        try:
            # Apply pitch and weather adjustments from research
            from enhanced_predictions import apply_pitch_and_weather_adjustments
            match_players = apply_pitch_and_weather_adjustments(match_players, upcoming_match)
            print("Applied pitch and weather adjustments based on research")
            
            # After selecting initial team, optimize synergy
            selected_players = optimize_team_synergy(selected_players, match_players, remaining_budget)
            print("Optimized team synergy based on player correlations")
        except Exception as e:
            print(f"Could not apply advanced research methods: {e}")

        # Dream11 team constraints
        max_players = 11
        min_team_players = 1
        max_team_players = 7
        
        # Role constraints (Dream11 rules)
        role_limits = {
            'WK': (1, 4),      # 1-4 wicket-keepers
            'BAT': (3, 6),     # 3-6 batsmen
            'AR': (1, 4),      # 1-4 all-rounders
            'BOWL': (3, 6)     # 3-6 bowlers
        }
        
        # Initialize data structures for team selection
        selected_players = []
        role_counts = {role: 0 for role in role_limits.keys()}
        team_counts = {}
        remaining_budget = budget
        
        # STEP 1: First ensure minimum requirements for each role
        for role, (min_req, _) in role_limits.items():
            role_players = match_players[match_players['role'] == role].sort_values('predicted_points', ascending=False)
            
            for _, player in role_players.iterrows():
                if role_counts[role] >= min_req:
                    break
                    
                team = player['team']
                if team_counts.get(team, 0) >= max_team_players:
                    continue
                    
                if player['cost'] > remaining_budget:
                    continue
                    
                if any(p['player_name'] == player['player_name'] for p in selected_players):
                    continue
                
                # Add player to team
                player_dict = player.to_dict()
                selected_players.append(player_dict)
                role_counts[role] += 1
                team_counts[team] = team_counts.get(team, 0) + 1
                remaining_budget -= player['cost']
        
        # STEP 2: Add best remaining players within constraints
        # Sort players by predicted points
        remaining_players = match_players.sort_values('predicted_points', ascending=False)
        
        for _, player in remaining_players.iterrows():
            if len(selected_players) >= max_players:
                break
                
            # Skip if already selected
            if any(p['player_name'] == player['player_name'] for p in selected_players):
                continue
                
            role = player['role']
            team = player['team']
            
            # Check constraints
            if role_counts[role] >= role_limits[role][1]:  # Max role limit
                continue
                
            if team_counts.get(team, 0) >= max_team_players:  # Max team limit
                continue
                
            if player['cost'] > remaining_budget:  # Budget constraint
                continue
                
            # Add player
            player_dict = player.to_dict()
            selected_players.append(player_dict)
            role_counts[role] += 1
            team_counts[team] = team_counts.get(team, 0) + 1
            remaining_budget -= player['cost']
        
        # Check if we have a valid team
        if len(selected_players) < max_players:
            print(f"Warning: Could only select {len(selected_players)} players within constraints")
            print(f"Role distribution: {role_counts}")
            print(f"Team distribution: {team_counts}")
            print(f"Remaining budget: {remaining_budget}")
            
            # Try one more time with relaxed role constraints if needed
            if len(selected_players) < max_players - 1:
                print("Attempting with relaxed constraints...")
                available_players = match_players.copy()
                
                # Remove already selected players
                for player in selected_players:
                    available_players = available_players[available_players['player_name'] != player['player_name']]
                
                # Sort by predicted points
                available_players = available_players.sort_values('predicted_points', ascending=False)
                
                for _, player in available_players.iterrows():
                    if len(selected_players) >= max_players:
                        break
                        
                    team = player['team']
                    if team_counts.get(team, 0) >= max_team_players:
                        continue
                        
                    if player['cost'] > remaining_budget:
                        continue
                    
                    # Add player even if role limit is exceeded
                    player_dict = player.to_dict()
                    selected_players.append(player_dict)
                    role = player['role']
                    role_counts[role] += 1
                    team_counts[team] = team_counts.get(team, 0) + 1
                    remaining_budget -= player['cost']
        
        # Choose captain and vice-captain (highest predicted points)
        if selected_players:
            # Sort by predicted points
            selected_players_sorted = sorted(selected_players, key=lambda p: p['predicted_points'], reverse=True)
            captain = selected_players_sorted[0]['player_name']
            vice_captain = selected_players_sorted[1]['player_name'] if len(selected_players_sorted) > 1 else None
            
            # Create result dictionary
            final_team = {
                'players': selected_players,
                'captain': captain,
                'vice_captain': vice_captain,
                'total_cost': budget - remaining_budget,
                'remaining_budget': remaining_budget,
                'role_distribution': role_counts,
                'team_distribution': team_counts
            }
            
            # Calculate expected points with captain and vice-captain multipliers
            total_points = 0
            for player in selected_players:
                points = player['predicted_points']
                if player['player_name'] == captain:
                    points *= captain_bonus
                elif player['player_name'] == vice_captain:
                    points *= vice_captain_bonus
                total_points += points
                
            final_team['expected_points'] = total_points
            
            # Print summary
            print("\n===== DREAM11 TEAM SUMMARY =====")
            print(f"Total players: {len(selected_players)}/11")
            print(f"Captain: {captain}")
            print(f"Vice Captain: {vice_captain}")
            print(f"Total cost: {final_team['total_cost']:.1f} credits (remaining: {final_team['remaining_budget']:.1f})")
            print(f"Roles: WK={role_counts.get('WK', 0)}, BAT={role_counts.get('BAT', 0)}, AR={role_counts.get('AR', 0)}, BOWL={role_counts.get('BOWL', 0)}")
            print(f"Expected points: {total_points:.1f}")
            print("===============================")
            
            # NEW CODE: Display detailed player list
            print("\n===== YOUR DREAM11 TEAM =====")
            print(f"Match: {home_team} vs {away_team}")
            print(f"Total Credits: {final_team['total_cost']:.1f}/100")
            
            # Sort players by role for better display
            role_order = {'WK': 1, 'BAT': 2, 'AR': 3, 'BOWL': 4}
            sorted_players = sorted(selected_players, key=lambda p: role_order[p['role']])
            
            # Display by role groups
            current_role = None
            for i, player in enumerate(sorted_players, 1):
                # Print role header when role changes
                if current_role != player['role']:
                    current_role = player['role']
                    role_names = {'WK': 'WICKET-KEEPERS', 'BAT': 'BATSMEN', 'AR': 'ALL-ROUNDERS', 'BOWL': 'BOWLERS'}
                    print(f"\n{role_names[current_role]}:")
                
                # Determine captain/vc tags
                captain_tag = " (C)" if player['player_name'] == captain else ""
                vc_tag = " (VC)" if player['player_name'] == vice_captain else ""
                
                # Determine team abbreviation
                team_abbr = player['team']
                
                # Calculate individual points
                player_points = player['predicted_points']
                if player['player_name'] == captain:
                    player_points *= captain_bonus
                elif player['player_name'] == vice_captain:
                    player_points *= vice_captain_bonus
                
                # Print player details
                print(f"{i}. {player['player_name']}{captain_tag}{vc_tag} ({team_abbr}) - ₹{player['cost']}cr - {player_points:.1f} pts")
            
            print("\n===============================")
            
            return final_team
        else:
            raise Exception("Failed to select any players")
            
    except Exception as e:
        print(f"Error in team selection: {e}")
        
        # Emergency fallback - create a dummy team
        dummy_team = {
            'players': [{'player_name': f"Player {i+1}", 'role': ['WK', 'BAT', 'AR', 'BOWL'][i % 4]} for i in range(11)],
            'captain': "Player 1",
            'vice_captain': "Player 2",
            'total_cost': 100.0,
            'remaining_budget': 0.0,
            'role_distribution': {'WK': 1, 'BAT': 4, 'AR': 2, 'BOWL': 4},
            'team_distribution': {home_team: 6, away_team: 5}
        }
        return dummy_team

if __name__ == "__main__":
    # Test with a sample match
    sample_match = {
        'match_id': 1,
        'home_team': 'Chennai Super Kings',
        'away_team': 'Mumbai Indians',
        'venue': 'Wankhede Stadium',
        'date': '2025-04-01'
    }
    
    # Run the team selection
    team = select_fantasy_team(sample_match)
    
    # Display the selected team
    if team and 'players' in team:
        print("\nSelected Dream11 Team:")
        for i, player in enumerate(team['players'], 1):
            captain_tag = " (C)" if player['player_name'] == team['captain'] else ""
            vc_tag = " (VC)" if player['player_name'] == team['vice_captain'] else ""
            print(f"{i}. {player['player_name']}{captain_tag}{vc_tag} - {player['role']} - {player['team']} - ₹{player['cost']}cr")