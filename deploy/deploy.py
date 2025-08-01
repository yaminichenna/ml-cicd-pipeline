import boto3
import os
import time

# AWS clients
region = os.environ.get("AWS_REGION", "us-east-1")
sagemaker = boto3.client("sagemaker", region_name=region)

# Dynamic model and endpoint names
timestamp = int(time.time())
model_name = f"ml-model-{timestamp}"
endpoint_config_name = f"ml-config-{timestamp}"
endpoint_name = f"ml-endpoint-{timestamp}"  # 🆕 unique endpoint name

# S3 path to model
bucket = os.environ["S3_BUCKET_NAME"]
model_data_url = f"s3://{bucket}/models/model.tar.gz"

# SageMaker container image
container_image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"

# IAM role
role = os.environ["SAGEMAKER_ROLE_ARN"]

print(f" Creating model: {model_name}")
try:
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": container_image,
            "ModelDataUrl": model_data_url,
        },
        ExecutionRoleArn=role,
    )

    print(f" Creating endpoint config: {endpoint_config_name}")
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.t2.medium",
            }
        ],
    )

    print(f" Creating endpoint: {endpoint_name}")
    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    print(" Deployment started. Check AWS Console → SageMaker → Endpoints")
    print(f" Endpoint Name: {endpoint_name}")

except Exception as e:
    print(" Deployment failed:", e)

