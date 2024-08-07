import argparse
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import sklearn
import dvc.api
import mlflow
import mlflow.sklearn

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

def load_data(file_path):
    """Load dataset from a local file."""
    data = pd.read_csv(file_path, sep=";")
    logging.info("Data loaded successfully from %s", file_path)
    return data

def preprocess_data(data):
    """Preprocess the dataset: handle missing values and scale features."""
    # Handle missing values
    if data.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        logging.info("Missing values handled")
    
    # Separate features and target variable
    X = data.drop("quality", axis=1)
    y = data["quality"]
    
    # Scaling features
    preprocessor = Pipeline(steps=[('scaler', StandardScaler())])
    X = preprocessor.fit_transform(X)
    logging.info("Data preprocessing completed")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info("Data split into training and test sets")
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train, y_train):
    """Train a baseline Random Forest model."""
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log the classification report and accuracy."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Model evaluation completed")
    logging.info("Classification Report:\n%s", report)
    logging.info("Accuracy: %f", accuracy)
    return report, accuracy

def perform_grid_search(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def main(data_path, model_dir, log_dir):
    # Setup logging
    setup_logging(log_dir)
    
    logging.info("SkLearn Version: %s", str(sklearn.__version__))

    # Load data from DVC
    data_url = dvc.api.get_url(path=data_path)
    data = load_data(data_url)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Set experiment name
    mlflow.set_experiment("Wine Quality Random Forest Experiment")

    best_accuracy = 0
    best_model = None

    # Train and log the baseline model
    with mlflow.start_run(run_name="Baseline Model"):
        baseline_model = train_baseline_model(X_train, y_train)
        report, accuracy = evaluate_model(baseline_model, X_test, y_test)
        
        mlflow.log_params({"n_estimators": 10, "max_depth": 3, "random_state": 42})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(baseline_model, "model")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = baseline_model

    # Perform hyperparameter tuning
    grid_search = perform_grid_search(X_train, y_train)

    # Log results of each experiment in GridSearchCV
    for i, params in enumerate(grid_search.cv_results_['params']):
        if i >= 10:
            break
        with mlflow.start_run(run_name=f"Experiment {i + 1}"):
            model = grid_search.best_estimator_
            report, accuracy = evaluate_model(model, X_test, y_test)
            
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

    # Save the best model using joblib
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.joblib')
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    logging.info("Best model saved to %s with accuracy %f", model_path, best_accuracy)
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model on the Wine Quality dataset.")
    parser.add_argument('--data-path', type=str, default='data/winequality-red.csv', help='Path to the dataset')
    parser.add_argument('--model-dir', type=str, default='model', help='Directory to save the trained model')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save the log files')
    
    args = parser.parse_args()
    main(args.data_path, args.model_dir, args.log_dir)