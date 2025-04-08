import pandas as pd

def load_wine_data(red_path: str, white_path: str) -> pd.DataFrame:
    # Load red and white wine datasets
    red_df = pd.read_csv(red_path, sep=';')
    white_df = pd.read_csv(white_path, sep=';')

    # Add a 'type' column to distinguish
    red_df['type'] = 'red'
    white_df['type'] = 'white'

    # Combine datasets
    combined_df = pd.concat([red_df, white_df], ignore_index=True)
    return combined_df