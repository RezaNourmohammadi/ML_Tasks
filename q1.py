import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Sequence

def load_rain_data_from_csv(filename: str):
    df = pd.read_csv(filename)
    df['RainProbability'] = pd.to_numeric(df['RainProbability'], errors='coerce')
    df['Day'] = df['Day'].astype(int)
    df = df.dropna(subset=['RainProbability'])
    
    if len(df) != 365:
        raise ValueError("CSV file must contain 365 days of data.")
    
    return df['RainProbability'].values

def generate_rain_data_from_probabilities(rain_probabilities: np.ndarray, num_samples: int = 1000, days_in_year: int = 365):
    data = []
    
    for _ in range(num_samples):
        rain_binary = np.random.rand(days_in_year) < rain_probabilities
        rainy_days_count = np.sum(rain_binary)
        data.append(np.concatenate([rain_probabilities, [rainy_days_count]]))
    
    columns = [f'Day_{i+1}' for i in range(days_in_year)] + ['Rainy_Days']
    df = pd.DataFrame(data, columns=columns)
    return df

def train_random_forest_regressor(rain_data: pd.DataFrame):
    X = rain_data.drop(columns=['Rainy_Days'])
    y = rain_data['Rainy_Days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse:.4f}")
    
    return rf_model

def prob_rain_more_than_n_regression(p: Sequence[float], n: int, model: RandomForestRegressor) -> float:
    p_df = pd.DataFrame([p], columns=[f'Day_{i+1}' for i in range(len(p))])
    predicted_rainy_days = model.predict(p_df)[0]
    return 1.0 if predicted_rainy_days >= n else 0.0

# Example usage:

rain_probabilities = load_rain_data_from_csv('rain_probabilities.csv')
rain_data = generate_rain_data_from_probabilities(rain_probabilities)
rf_model = train_random_forest_regressor(rain_data)

n = 100
probability = prob_rain_more_than_n_regression(rain_probabilities, n, rf_model)
print(f"The probability of at least {n} rainy days: {probability}")
