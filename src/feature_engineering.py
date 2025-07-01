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

def load_preprocessed_data(train_processed,test_processed):
    """
    Load preprocessed data from CSV file.
    
    """
    try:
        logger.info(f"Loading preprocessed data from {train_processed,test_processed}")
        train_processed = pd.read_csv('data/interim/train_processed.csv')
        test_processed = pd.read_csv('data/interim/train_processed.csv')
        logger.debug('Data loaded properly')

        return train_processed, test_processed
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise


def drop_anomaly_rows(train_data, test_data):
    """
    Drop rows where engagement_status is 'Anomaly' from both train and test data
    
    """
    try:
        train_initial_rows = len(train_data)
        test_initial_rows = len(test_data)

        if 'engagement_status' in train_data.columns:
            train_data = train_data[train_data['engagement_status'] != 'Anomaly'].reset_index(drop=True)
            train_rows_dropped = train_initial_rows - len(train_data)
            logger.info(f"Dropped {train_rows_dropped} rows with Anomaly status from training data")

        if 'engagement_status' in test_data.columns:
            test_data = test_data[test_data['engagement_status'] != 'Anomaly'].reset_index(drop=True)
            test_rows_dropped = test_initial_rows - len(test_data)
            logger.info(f"Dropped {test_rows_dropped} rows with Anomaly status from test data")

        return train_data, test_data
    except Exception as e:
        logger.error(f"Error dropping anomaly rows: {str(e)}")
        raise


def drop_columns(train_data, test_data):
    """
    Drop unnecessary columns from both train and test DataFrames.
    
    """
    try:
        columns_to_drop = ['email_id', 'clicked', 'opened']
        
        train_copy = train_data.copy()
        test_copy = test_data.copy()
        
        train_copy.drop(columns=columns_to_drop, inplace=True)
        test_copy.drop(columns=columns_to_drop, inplace=True)
        
        logger.info(f"Dropped unnecessary columns {columns_to_drop} from both train and test data")
        return train_copy, test_copy
        
    except Exception as e:
        logger.error(f"Error dropping columns: {str(e)}")
        raise

def save_cleaned_data(train_data, test_data):
    """
    Save cleaned train and test data to CSV files in the data/cleaned folder.

    """
    try:
        # Create data/cleaned directory if it doesn't exist
        data_dir = os.path.join("./data", "cleaned")
        os.makedirs(data_dir, exist_ok=True)
        
        # Define output paths
        train_output_path = os.path.join(data_dir, 'train_cleaned.csv')
        test_output_path = os.path.join(data_dir, 'test_cleaned.csv')
        
        # Save train data
        logger.info(f"Saving cleaned training data to {train_output_path}")
        train_data.to_csv(train_output_path, index=False)
        logger.info(f"Successfully saved cleaned training data with shape {train_data.shape}")
        
        # Save test data
        logger.info(f"Saving cleaned test data to {test_output_path}")
        test_data.to_csv(test_output_path, index=False)
        logger.info(f"Successfully saved cleaned test data with shape {test_data.shape}")
        
    except Exception as e:
        logger.error(f"Error saving cleaned data: {str(e)}")
        raise

def main():
    try:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        train_data, test_data = load_preprocessed_data('data/interim/train_processed.csv', 
                                                      'data/interim/test_processed.csv')

        # Drop anomaly rows
        logger.info("Dropping anomaly rows...")
        train_data, test_data = drop_anomaly_rows(train_data, test_data)

        # Drop unnecessary columns
        logger.info("Dropping unnecessary columns...")
        train_data, test_data = drop_columns(train_data, test_data)

        # Save cleaned data
        logger.info("Saving cleaned data...")
        save_cleaned_data(train_data, test_data)

        logger.info("Feature engineering process completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
