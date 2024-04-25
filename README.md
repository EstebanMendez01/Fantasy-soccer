# Fantasy-soccer

This project aims to predict the performance of football players in terms of goals and assists using machine learning techniques. It provides a web interface where users can select teams and view predictions for the top players from each team.

## Requirements

- Python 3.x
- Flask
- Pandas
- NumPy
- scikit-learn

## Installation

1. Clone the repository:

```
git clone <repository-url>
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the Flask application:

```
python app.py
```

## Usage

1. Open the web interface in your browser (by default, it should be accessible at `http://localhost:5000`).
2. Select the teams for which you want to predict player performance.
3. Click on the "Predict Performance" button to view the predicted goals and assists for the top players from each selected team.

## Files

- `fantasy.py`: Flask application file containing routes and logic for the web interface.
- `static/players_data.csv`: CSV file containing preprocessed data of football player statistics.
- `index2.html`: HTML template file for rendering the web interface.
- `goals_prediction_model.pkl`: Pickled file containing the trained model for predicting goals.
- `assists_prediction_model.pkl`: Pickled file containing the trained model for predicting assists.
- `label_encoder.pkl`: Pickled file containing the label encoder used for encoding player names.
