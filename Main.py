# main.py

import os
from src.load_data import load_wine_data
from src.preprocess import preprocess_data
from src.train_model import train_random_forest
from src.evaluate import evaluate_model
from datetime import datetime
def main():
    # Paths
    red_path = 'data/winequality-red.csv'
    white_path = 'data/winequality-white.csv'
    model_path = 'models/random_forest_wine_model.pkl'
    report_path = 'docs/evaluation_report.txt'

    # Step 1: Load & preprocess
    print("Loading and preprocessing data...")
    df = load_wine_data(red_path, white_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Step 2: Train model
    print("Training model...")
    model = train_random_forest(X_train, y_train, model_path)

    # Step 3: Evaluate & log report
    print("Evaluating model...")
    report = evaluate_model(model, X_test, y_test)

    # Step 4: Save report
    os.makedirs('docs', exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Wine Quality Prediction Report - {datetime.now()}\n\n")
        f.write(report)

    print(f"Evaluation report saved to: {report_path}")

if __name__ == "__main__":
    main()
