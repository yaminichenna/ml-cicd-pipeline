import boto3
import os
import time

sm_client = boto3.client("sagemaker", region_name=os.environ["AWS_REGION"])

model_name = f"ml-model-{int(time.time())}"
model_data_url = f"s3://{os.environ['S3_BUCKET_NAME']}/models/model.tar.gz"

print(f"Creating model: {model_name}")

try:
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "ModelDataUrl": model_data_url,
        },
        ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
    )

    print("Creating endpoint config...")
    endpoint_config_name = model_name + "-config"
    sm_client.create_endpoint_config(
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

    print("Creating endpoint...")
    sm_client.create_endpoint(
        EndpointName="ml-endpoint",
        EndpointConfigName=endpoint_config_name
    )

    print("Deployment triggered. Monitor status in SageMaker Console → Endpoints.")
except Exception as e:
    print("Deployment failed:", e)
