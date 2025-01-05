import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Function to train the machine learning model
def create_model(data):
    # Separate the features (X) and target variable (y)
    X = data.drop(['id', 'diagnosis'], axis=1)
    y = data['diagnosis']

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize the features for better performance of models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)  # Retain 10 principal components
    X_pca = pca.fit_transform(X_scaled)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Initialize and train Logistic Regression model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = lr.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return lr, scaler, imputer, pca

# Function to load and preprocess the data
def get_clean_data():
    # Load the dataset
    data = pd.read_csv("data/data.csv")
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32'], axis=1, errors='ignore')
    # Map the target variable: Malignant ('M') as 1, Benign ('B') as 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    # Load the cleaned dataset
    data = get_clean_data()

    # Train the model and get the trained scaler, imputer, and PCA
    lr_model, scaler, imputer, pca = create_model(data)

    # Ensure the directory 'model/' exists
    os.makedirs('model', exist_ok=True)

    # Save the trained model to a file for later use
    with open('model/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    # Save the scaler to a file for consistent scaling during predictions
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save the imputer to handle missing values during predictions
    with open('model/imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    # Save the PCA transformer for consistent dimensionality reduction
    with open('model/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

if __name__ == '__main__':
    main()
