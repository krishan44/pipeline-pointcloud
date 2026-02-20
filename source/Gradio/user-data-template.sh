#!/bin/bash
# User Data script for running Gaussian Splatting pipeline on EC2
# This script is passed to EC2 instances at launch time

set -e

# Log everything to CloudWatch and local file
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== Starting Gaussian Splatting Pipeline ==="
echo "Timestamp: $(date)"

# Get instance metadata
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)
REGION=$(ec2-metadata --availability-zone | cut -d " " -f 2 | sed 's/[a-z]$//')

echo "Instance ID: $INSTANCE_ID"
echo "Region: $REGION"

# Environment variables will be injected by Lambda/Step Functions
# UUID={{UUID}}
# S3_INPUT={{S3_INPUT}}
# S3_OUTPUT={{S3_OUTPUT}}
# ECR_IMAGE_URI={{ECR_IMAGE_URI}}
# and all other required environment variables...

# Authenticate Docker to ECR
echo "Authenticating to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin {{ECR_IMAGE_URI_BASE}}

# Pull the container image
echo "Pulling Docker image: {{ECR_IMAGE_URI}}"
docker pull {{ECR_IMAGE_URI}}

# Create directory for pipeline work
mkdir -p /opt/gaussian-splatting
cd /opt/gaussian-splatting

# Run the container with all environment variables
echo "Starting Gaussian Splatting container..."
docker run --gpus all \
  --rm \
  -e UUID="{{UUID}}" \
  -e S3_INPUT="{{S3_INPUT}}" \
  -e S3_OUTPUT="{{S3_OUTPUT}}" \
  -e FILENAME="{{FILENAME}}" \
  -e INPUT_PREFIX="{{INPUT_PREFIX}}" \
  -e OUTPUT_PREFIX="{{OUTPUT_PREFIX}}" \
  -e S3_BUCKET_NAME="{{S3_BUCKET_NAME}}" \
  -e MAX_NUM_IMAGES="{{MAX_NUM_IMAGES}}" \
  -e FILTER_BLURRY_IMAGES="{{FILTER_BLURRY_IMAGES}}" \
  -e RUN_SFM="{{RUN_SFM}}" \
  -e SFM_SOFTWARE_NAME="{{SFM_SOFTWARE_NAME}}" \
  -e USE_POSE_PRIOR_COLMAP_MODEL_FILES="{{USE_POSE_PRIOR_COLMAP_MODEL_FILES}}" \
  -e USE_POSE_PRIOR_TRANSFORM_JSON="{{USE_POSE_PRIOR_TRANSFORM_JSON}}" \
  -e SOURCE_COORDINATE_NAME="{{SOURCE_COORDINATE_NAME}}" \
  -e POSE_IS_WORLD_TO_CAM="{{POSE_IS_WORLD_TO_CAM}}" \
  -e ENABLE_ENHANCED_FEATURE_EXTRACTION="{{ENABLE_ENHANCED_FEATURE_EXTRACTION}}" \
  -e MATCHING_METHOD="{{MATCHING_METHOD}}" \
  -e RUN_TRAIN="{{RUN_TRAIN}}" \
  -e MODEL="{{MODEL}}" \
  -e MAX_STEPS="{{MAX_STEPS}}" \
  -e ENABLE_MULTI_GPU="{{ENABLE_MULTI_GPU}}" \
  -e ROTATE_SPLAT="{{ROTATE_SPLAT}}" \
  -e SPHERICAL_CAMERA="{{SPHERICAL_CAMERA}}" \
  -e SPHERICAL_CUBE_FACES_TO_REMOVE="{{SPHERICAL_CUBE_FACES_TO_REMOVE}}" \
  -e OPTIMIZE_SEQUENTIAL_SPHERICAL_FRAME_ORDER="{{OPTIMIZE_SEQUENTIAL_SPHERICAL_FRAME_ORDER}}" \
  -e REMOVE_BACKGROUND="{{REMOVE_BACKGROUND}}" \
  -e BACKGROUND_REMOVAL_MODEL="{{BACKGROUND_REMOVAL_MODEL}}" \
  -e MASK_THRESHOLD="{{MASK_THRESHOLD}}" \
  -e REMOVE_HUMAN_SUBJECT="{{REMOVE_HUMAN_SUBJECT}}" \
  -e LOG_VERBOSITY="{{LOG_VERBOSITY}}" \
  -e NUM_THREADS="{{NUM_THREADS}}" \
  -e NUM_GPUS="{{NUM_GPUS}}" \
  -e AWS_REGION="$REGION" \
  {{ECR_IMAGE_URI}}

DOCKER_EXIT_CODE=$?

echo "Container finished with exit code: $DOCKER_EXIT_CODE"

# Cleanup
echo "Cleaning up Docker images..."
docker system prune -af

# Self-terminate the instance
echo "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION

echo "=== Pipeline Complete ==="
