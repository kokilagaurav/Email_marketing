import os
import logging
import pandas as pd 

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def add_engagement_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add engagement status column based on opened and clicked values."""
    try:
        def get_engagement_status(row):
            if row['clicked'] == 1 and row['opened'] == 1:
                return "Clicked and Opened"
            elif row['opened'] == 1 and row['clicked'] == 0:
                return "Opened but Not Clicked"
            elif row['opened'] == 0 and row['clicked'] == 0:
                return "Not Opened"
            else:
                return "Anomaly"

        df['engagement_status'] = df.apply(get_engagement_status, axis=1)
        logger.debug('Engagement status column added successfully')
        return df
    except Exception as e:
        logger.error('Failed to add engagement status: %s', e)
        raise




def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, add engagement status, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # add engagement status to both train and test data
        train_processed_data = add_engagement_status(train_data)
        test_processed_data = add_engagement_status(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
