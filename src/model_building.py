import pandas as pd
import os
import logging
import yaml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

def load_cleaned_data(train_cleaned,test_cleaned):
    """
    Load preprocessed data from CSV file.
    
    """
    try:
        logger.info(f"Loading cleaned data from {train_cleaned}, {test_cleaned}")
        train_cleaned = pd.read_csv('data/cleaned/train_cleaned.csv')
        test_cleaned = pd.read_csv('data/cleaned/test_cleaned.csv')
        logger.debug('Data loaded properly')

        return train_cleaned, test_cleaned
    except Exception as e:
        logger.error(f"Error loading cleaned data: {str(e)}")
        raise

def split_features_target(train_cleaned, test_cleaned):
    """
    Split data into features (X) and target (y) for both train and test sets.
    
    Args:
        train_cleaned (pd.DataFrame): Cleaned training data
        test_cleaned (pd.DataFrame): Cleaned test data
        
    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    try:
        logger.info("Splitting data into features and target")
        
        features = ['email_text', 'email_version', 'user_country', 
                    'user_past_purchases', 'hour', 'weekday']
        
        X_train = train_cleaned[features]
        y_train = train_cleaned['engagement_status']
        
        X_test = test_cleaned[features]
        y_test = test_cleaned['engagement_status']
        
        logger.debug('Data split successfully')
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error splitting features and target: {str(e)}")
        raise

def encode_target(y_train, y_test):
    """
    Encode target variables using LabelEncoder.

    """
    try:
        logger.info("Encoding target variables")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        logger.debug('Target variables encoded successfully')
        return y_train_encoded, y_test_encoded
        
    except Exception as e:
        logger.error(f"Error encoding target variables: {str(e)}")
        raise

def preprocess_features(X_train, X_test):
    """
    Preprocess features using OneHotEncoder for categorical variables and StandardScaler for numerical variables.   
    """
    try:
        logger.info("Preprocessing features")
        
        cat_cols = ['email_text', 'email_version', 'user_country', 'weekday']
        num_cols = ['user_past_purchases', 'hour']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_cols', OneHotEncoder(drop='first'), cat_cols),
                ('num_cols', StandardScaler(), num_cols)
            ]
        )
        
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        logger.debug('Features preprocessed successfully')
        return X_train_preprocessed, X_test_preprocessed
        
    except Exception as e:
        logger.error(f"Error preprocessing features: {str(e)}")
        raise

def create_model_ensemble():
    """
    Create an ensemble of base models using VotingClassifier.
    
    Returns:
        VotingClassifier: Ensemble model with soft voting
    """
    try:
        logger.info("Creating model ensemble")
        
        base_models = {
            'logistic': LogisticRegression(class_weight='balanced', 
                                         solver='saga', max_iter=2000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gradient_boost': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42, class_weight='balanced'),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }

        # Create the voting classifier
        voting_classifier = VotingClassifier(
            estimators=list(base_models.items()),
            voting='soft'
        )

        logger.debug('Model ensemble created successfully')
        return voting_classifier
        
    except Exception as e:
        logger.error(f"Error creating model ensemble: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting model building process")

        # Load cleaned data
        train_cleaned, test_cleaned = load_cleaned_data('data/cleaned/train_cleaned.csv', 
                                                        'data/cleaned/test_cleaned.csv')

        # Split features and target
        X_train, y_train, X_test, y_test = split_features_target(train_cleaned, test_cleaned)

        # Encode target variables
        y_train_encoded, y_test_encoded = encode_target(y_train, y_test)

        # Preprocess features
        X_train_preprocessed, X_test_preprocessed = preprocess_features(X_train, X_test)

        # Create ensemble model
        voting_classifier = create_model_ensemble()

        # Fit the ensemble model
        logger.info("Fitting the ensemble model")
        voting_classifier.fit(X_train_preprocessed, y_train_encoded)

        logger.info("Model building process completed successfully")
        # Optionally, you can return the fitted model
        return voting_classifier

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
