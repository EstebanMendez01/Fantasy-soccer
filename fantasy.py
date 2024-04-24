import pandas as pd
import numpy as np  # Add numpy for generating random data
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the preprocessed data
data = pd.read_csv('static/players_data.csv', index_col='player')  # Set 'player' column as index

# Prepare features and target
X = data.select_dtypes(include=['float64', 'int64'])  # Only select numeric columns
y_goals = data['goals']
y_assists = data['assists']

# Encode categorical variables
label_encoder = LabelEncoder()
X.index = label_encoder.fit_transform(X.index)

# Train the models
model_goals = RandomForestRegressor(n_estimators=100, random_state=42)
model_goals.fit(X, y_goals)

model_assists = RandomForestRegressor(n_estimators=100, random_state=42)
model_assists.fit(X, y_assists)

# Save the trained models
joblib.dump(model_goals, 'goals_prediction_model.pkl')
joblib.dump(model_assists, 'assists_prediction_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Get all teams
all_teams = sorted(data['team'].unique())

def select_best_players(team):
    # Filter the data for the selected team
    team_data = data[data['team'] == team]
    
    # Sort the players based on some criteria (e.g., goals or assists)
    sorted_players = team_data.sort_values(by=['goals', 'assists'], ascending=False)
    
    # Select the top 5 players and get their names
    best_players = sorted_players.index[:5].tolist()
    
    return best_players

def predict_player_performance(player_stats, model_goals, model_assists):
    # Ensure proper feature names
    player_stats.columns = X.columns

    # Make predictions
    player_stats_values = player_stats.values.reshape(1, -1)  # Reshape the data for prediction
    predicted_goals = int(model_goals.predict(player_stats_values)[0])
    predicted_assists = int(model_assists.predict(player_stats_values)[0])

    # Return all predicted stats
    return predicted_goals, predicted_assists


@app.route('/')
def home():
    return render_template('index2.html', teams=all_teams)

@app.route('/select_players', methods=['POST'])
def select_players():
    team1 = request.form['team1']
    team2 = request.form['team2']

    best_players_team1 = select_best_players(team1)
    best_players_team2 = select_best_players(team2)
    
    return render_template('index2.html', team1=team1, team2=team2, teams=all_teams, best_players_team1=best_players_team1, best_players_team2=best_players_team2)

@app.route('/predict_performance', methods=['POST'])
def predict_performance():
    team1 = request.form['team1']
    team2 = request.form['team2']

    best_players_team1 = select_best_players(team1)
    best_players_team2 = select_best_players(team2)

    # Load the trained models
    model_goals = joblib.load('goals_prediction_model.pkl')
    model_assists = joblib.load('assists_prediction_model.pkl')

    # Generate random player statistics for prediction
    random_player_stats = pd.DataFrame(np.random.randint(0, 10, size=(len(best_players_team1 + best_players_team2), len(X.columns))), columns=X.columns)

    # Get predicted stats for each player
    predicted_stats = {}
    for player, stats in zip(best_players_team1 + best_players_team2, random_player_stats.iterrows()):
        predicted_goals, predicted_assists = predict_player_performance(stats[1], model_goals, model_assists)
        predicted_stats[player] = (predicted_goals, predicted_assists)
    
    return render_template('index2.html', team1=team1, team2=team2, teams=all_teams,
                           best_players_team1=best_players_team1, best_players_team2=best_players_team2,
                           predicted_stats=predicted_stats)

if __name__ == '__main__':
    app.run(debug=True)
