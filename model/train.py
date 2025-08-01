import boto3
import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load example dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model to file
model_path = "model.joblib"
joblib.dump(model, model_path)

# Upload model to S3
s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET_NAME"]  # Get bucket name from GitHub secret during CI/CD
s3.upload_file(model_path, bucket, f"models/{model_path}")

print("Model trained and uploaded to S3.")

