# run_pipeline.py (outside src/, or inside notebooks/)

from src.load_data import load_wine_data
from src.preprocess import preprocess_data
from src.train_model import train_random_forest
from src.evaluate import evaluate_model

# Paths to datasets
red_csv_path = 'data/winequality-red.csv'
white_csv_path = 'data/winequality-white.csv'

# Load and preprocess
df = load_wine_data(red_csv_path, white_csv_path)
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train and evaluate
model = train_random_forest(X_train, y_train)
evaluate_model(model, X_test, y_test)
