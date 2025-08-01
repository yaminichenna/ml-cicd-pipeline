import boto3
import os
import time
from botocore.exceptions import ClientError

# Environment variables from GitHub Secrets
region = os.environ["AWS_REGION"]
bucket = os.environ["S3_BUCKET_NAME"]
role = os.environ["SAGEMAKER_ROLE_ARN"]

# Unique resource names
timestamp = str(int(time.time()))
model_name = f"ml-model-{timestamp}"
endpoint_config_name = f"{model_name}-config"
endpoint_name = "ml-endpoint"

# Initialize SageMaker client
sagemaker = boto3.client("sagemaker", region_name=region)

# Container definition (scikit-learn prebuilt container)
container = {
    "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
    "ModelDataUrl": f"s3://{bucket}/model/model.tar.gz"
}

try:
    # 1. Create SageMaker model
    print(f"Creating model: {model_name}")
    sagemaker.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer=container
    )

    # 2. Create endpoint config
    print(f" Creating endpoint config: {endpoint_config_name}")
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.t2.medium",
                "InitialInstanceCount": 1
            }
        ]
    )

    # 3. Deploy endpoint
    print(f" Deploying endpoint: {endpoint_name}")
    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    print(f" SageMaker endpoint '{endpoint_name}' deployment started.")

except ClientError as e:
    print(f" Deployment failed: {e}")
