import boto3
import joblib
import os
import pandas as pd
import tarfile
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

# Compress model to .tar.gz for SageMaker
archive_path = "model.tar.gz"
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(model_path)

# Upload tar.gz to S3
s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET_NAME"]  # Use GitHub Secret at runtime
s3.upload_file(archive_path, bucket, "model/model.tar.gz")

print(f"Model trained and uploaded to s3://{bucket}/model/model.tar.gz")

