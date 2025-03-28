Dream11 Advanced Prediction System
A sophisticated, data-driven fantasy cricket team prediction system powered by machine learning for Dream11 contests. This system uses real historical cricket data to make accurate predictions, optimize team selection, and employ advanced strategies used by top Dream11 players.

Features
Real Cricket Data: Uses comprehensive historical cricket performance data (~1600+ players)
ML-Powered Predictions: Employs trained machine learning models to predict player performances
Multiple Team Variations: Generates diverse team options with different strategic approaches
News Integration: Adjusts predictions based on latest player news and updates
Field Condition Analysis: Considers toss results, pitch reports, and weather conditions
Advanced C/VC Selection: Optimizes captain/vice-captain selection using multiple factors
Pre-Match Validation: Performs last-minute checks for playing XI and critical updates
Installation
Setup
1. Kaggle API Setup (for data acquisition)
2. Initialize the System
This will:

Download cricket performance data from Kaggle
Process the data for batsmen and bowlers
Train machine learning models
Create player name mappings
Usage
Then visit http://127.0.0.1:5000 in your browser to:

Select upcoming matches
Generate optimal fantasy teams
Create multiple team variations
Run pre-match checks
System Architecture
Core Components:
Data Integration: Processes and transforms cricket data for prediction
ML Models: Predicts player performance using Random Forest and Gradient Boosting
Team Selection: Selects optimal team within Dream11 constraints
Strategy System: Implements various team building strategies
Advanced Features:
News Integration: Scrapes and analyzes cricket news for player updates
Field Conditions: Analyzes toss results, weather, and pitch reports
Captain Optimization: Uses multi-factor analysis for C/VC selection
Pre-match Checker: Validates team selection before match start
File Structure
Advanced Strategies
Multiple Team Generation
Generates variations with different strategic focuses:

Balanced: Optimal mix of consistency and upside
High Ceiling: Focuses on players with explosive potential
Recent Form: Prioritizes players in excellent current form
Matchup Based: Considers head-to-head performance history
Contrarian: Differentiates from common picks for tournaments
News Integration
Scrapes cricket news from ESPNCricinfo and Cricbuzz
Identifies player mentions and impact on performance
Adjusts player projections based on news sentiment
Field Condition Analysis
Analyzes toss results and their impact on player performance
Considers pitch conditions (pace/spin friendly)
Factors in weather conditions (rain, humidity, wind)
Captain Optimization
Uses multi-factor analysis including form, matchup, and venue
Calculates ceiling and floor projections for players
Identifies optimal captain and vice-captain selections
Technologies Used
Python: Core programming language
Pandas/NumPy: Data manipulation
Scikit-learn: Machine learning models
Flask: Web framework
BeautifulSoup: Web scraping
Kaggle API: Data acquisition
Acknowledgments
Cricket player performance data from Kaggle
Dream11 format and scoring system
For Dream11 Gamethon Participants
This system gives you a significant advantage by implementing strategies used by top Dream11 players:

Create multiple team variations to diversify risk
Update selections based on latest news and field conditions
Optimize captain/vice-captain selections for maximum points
Run pre-match checks to validate your final team
