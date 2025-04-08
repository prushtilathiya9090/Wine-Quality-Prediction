from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Convert wine type to numeric
    df['type'] = df['type'].map({'red': 0, 'white': 1})

    # Split features and target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
