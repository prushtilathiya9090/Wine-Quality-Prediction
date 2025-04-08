from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_random_forest(X_train, y_train, model_path='../models/random_forest_wine_model.pkl'):
    # ✅ Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # This is the key part to prevent FileNotFoundError

    # ✅ Train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ✅ Save the model
    joblib.dump(rf, model_path)
    print(f"✅ Model saved at: {model_path}")
    return rf
