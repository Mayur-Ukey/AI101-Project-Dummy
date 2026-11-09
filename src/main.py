"""
main.py (Execution lead)
Runs everything in order: prepare data -> train model -> evaluate -> predict.
Run from the repo folder with: python src/main.py
"""

from data_prep import load_and_split
from train_model import train
from evaluate import evaluate
# Import the predict function from predict.py
from predict import predict 

print("STEP 1/4: preparing the data")
X_train, X_test, y_train, y_test = load_and_split()

print("\nSTEP 2/4: training the model")
model = train(X_train, y_train)

print("\nSTEP 3/4: evaluating the model")
evaluate(model, X_test, y_test)

print("\nSTEP 4/4: testing a prediction")
# Grab the first match from the test set as a sample
sample_match = X_test.iloc[[0]] 
prediction = predict(model, sample_match)
print(f"Sample Prediction Output: {prediction}")

print("\nDone. Full pipeline executed successfully!")
