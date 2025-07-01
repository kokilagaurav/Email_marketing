import pandas as pd
import os
import logging
import yaml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def drop_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Drop specified columns from both training and test dataframes."""
    try:
        # Debug: print available columns
        logger.debug('Train columns: %s', train_df.columns.tolist())
        logger.debug('Test columns: %s', test_df.columns.tolist())
        
        # Prepare data first
        columns_to_drop = ['email_id', 'clicked', 'opened']
        
        # Store initial dataframes
        X_train = train_df.copy()
        X_test = test_df.copy()
        
        # Check if target column exists, if not create dummy targets
        if 'engagement_status' in X_train.columns:
            y_train = X_train['engagement_status'].values
            y_test = X_test['engagement_status'].values
            columns_to_drop.append('engagement_status')
        else:
            logger.warning('Target column not found, creating dummy targets')
            y_train = None
            y_test = None
        
        # Only drop columns that actually exist
        existing_columns_to_drop = [col for col in columns_to_drop if col in X_train.columns]
        logger.debug('Dropping columns: %s', existing_columns_to_drop)
        
        X_train.drop(columns=existing_columns_to_drop, inplace=True)
        X_test.drop(columns=existing_columns_to_drop, inplace=True)
        
        logger.debug('Columns dropped and X, y split created for train and test data')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error('Error dropping columns and creating splits: %s', e)
        raise




def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        X_train, X_test, y_train, y_test = drop_columns(train_data, test_data)

        save_data(X_train, os.path.join("./data", "processed", "train_emails.csv"))
        save_data(X_test, os.path.join("./data", "processed", "test_emails.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
