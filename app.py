from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
from team_selection import select_fantasy_team
from enhanced_predictions import select_enhanced_fantasy_team

app = Flask(__name__)

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)

@app.route('/')
def home():
    # Get list of upcoming matches
    try:
        upcoming_matches = pd.read_csv('data/upcoming_matches.csv')
        matches = upcoming_matches.to_dict('records')
    except Exception as e:
        # Sample matches if file doesn't exist
        matches = [
            {'match_id': 1, 'home_team': 'Chennai Super Kings', 'away_team': 'Mumbai Indians', 
             'venue': 'M. A. Chidambaram Stadium', 'date': '2025-04-01'},
            {'match_id': 2, 'home_team': 'Royal Challengers Bangalore', 'away_team': 'Kolkata Knight Riders', 
             'venue': 'M. Chinnaswamy Stadium', 'date': '2025-04-02'},
            {'match_id': 3, 'home_team': 'Delhi Capitals', 'away_team': 'Rajasthan Royals', 
             'venue': 'Arun Jaitley Stadium', 'date': '2025-04-03'}
        ]
    
    return render_template('index.html', matches=matches)

@app.route('/predict', methods=['POST'])
def predict():
    match_id = request.form.get('match_id')
    advanced = request.form.get('advanced') == 'true'
    
    try:
        # Get match details
        upcoming_matches = pd.read_csv('data/upcoming_matches.csv')
        match_details = upcoming_matches[upcoming_matches['match_id'] == int(match_id)].iloc[0].to_dict()
    except:
        # Fallback to hardcoded match if file doesn't exist
        match_details = {
            'match_id': match_id,
            'home_team': request.form.get('home_team', 'Chennai Super Kings'),
            'away_team': request.form.get('away_team', 'Mumbai Indians'),
            'venue': request.form.get('venue', 'M. A. Chidambaram Stadium'),
            'date': request.form.get('date', '2025-04-01')
        }
    
    # Select fantasy team with enhanced predictions
    if advanced:
        print("Using advanced prediction model")
        selected_team = select_enhanced_fantasy_team(match_details)
    else:
        print("Using standard prediction model")
        selected_team = select_fantasy_team(match_details)
    
    # Calculate role counts if not already present
    if 'role_counts' not in selected_team:
        role_counts = {}
        for player in selected_team['players']:
            role = player['role']
            role_counts[role] = role_counts.get(role, 0) + 1
        selected_team['role_counts'] = role_counts
    
    return render_template('prediction.html', team=selected_team, match=match_details)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    match_details = data.get('match')
    advanced = data.get('advanced', True)
    
    try:
        if advanced:
            selected_team = select_enhanced_fantasy_team(match_details)
        else:
            selected_team = select_fantasy_team(match_details)
        return jsonify({"success": True, "team": selected_team})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/customize', methods=['GET', 'POST'])
def customize_team():
    if request.method == 'POST':
        # Handle customization logic here
        return jsonify({"success": True})
    else:
        match_id = request.args.get('match_id')
        # Get match details
        # Similar to predict route
        return render_template('customize.html', match=match_details)

if __name__ == '__main__':
    app.run(debug=True)