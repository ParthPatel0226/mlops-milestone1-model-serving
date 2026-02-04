"""
Train a simple classification model for demonstration.
This creates a logistic regression model for the Iris dataset.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def train_model():
    """Train and save a simple ML model."""
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Save model
    print("Saving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    
    # Test deterministic loading
    print("\nTesting deterministic loading...")
    model_reloaded = joblib.load('model.pkl')
    test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred1 = model.predict(test_input)
    pred2 = model_reloaded.predict(test_input)
    
    assert np.array_equal(pred1, pred2), "Model loading is not deterministic!"
    print(f"✓ Deterministic test passed! Prediction: {pred1[0]}")
    
    print("\n✓ Model training complete!")
    print(f"  - Model saved as: model.pkl")
    print(f"  - Accuracy: {accuracy:.2%}")
    print(f"  - Input shape: 4 features")
    print(f"  - Output: 3 classes (Iris species: 0=setosa, 1=versicolor, 2=virginica)")

if __name__ == "__main__":
    train_model()