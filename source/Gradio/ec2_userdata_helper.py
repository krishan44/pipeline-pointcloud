# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Helper module for generating EC2 user data script"""

import base64
import os
import multiprocessing


def generate_user_data(env_vars: dict, ecr_image_uri: str) -> str:
    """
    Generate user data script for EC2 instance
    
    Args:
        env_vars: Dictionary of environment variables
        ecr_image_uri: ECR image URI
        
    Returns:
        Base64 encoded user data script
    """
    
    # Extract ECR base URI (without tag)
    ecr_base_uri = ecr_image_uri.split(':')[0] if ':' in ecr_image_uri else ecr_image_uri
    
    # Read the template
    template_path = os.path.join(os.path.dirname(__file__), 'user-data-template.sh')
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    user_data_script = template.replace('{{UUID}}', env_vars.get('UUID', ''))
    user_data_script = user_data_script.replace('{{S3_INPUT}}', env_vars.get('S3_INPUT', ''))
    user_data_script = user_data_script.replace('{{S3_OUTPUT}}', env_vars.get('S3_OUTPUT', ''))
    user_data_script = user_data_script.replace('{{ECR_IMAGE_URI}}', ecr_image_uri)
    user_data_script = user_data_script.replace('{{ECR_IMAGE_URI_BASE}}', ecr_base_uri)
    user_data_script = user_data_script.replace('{{FILENAME}}', env_vars.get('FILENAME', ''))
    user_data_script = user_data_script.replace('{{INPUT_PREFIX}}', env_vars.get('INPUT_PREFIX', ''))
    user_data_script = user_data_script.replace('{{OUTPUT_PREFIX}}', env_vars.get('OUTPUT_PREFIX', ''))
    user_data_script = user_data_script.replace('{{S3_BUCKET_NAME}}', env_vars.get('S3_BUCKET_NAME', ''))
    user_data_script = user_data_script.replace('{{MAX_NUM_IMAGES}}', env_vars.get('MAX_NUM_IMAGES', ''))
    user_data_script = user_data_script.replace('{{FILTER_BLURRY_IMAGES}}', env_vars.get('FILTER_BLURRY_IMAGES', ''))
    user_data_script = user_data_script.replace('{{RUN_SFM}}', env_vars.get('RUN_SFM', ''))
    user_data_script = user_data_script.replace('{{SFM_SOFTWARE_NAME}}', env_vars.get('SFM_SOFTWARE_NAME', ''))
    user_data_script = user_data_script.replace('{{USE_POSE_PRIOR_COLMAP_MODEL_FILES}}', env_vars.get('USE_POSE_PRIOR_COLMAP_MODEL_FILES', ''))
    user_data_script = user_data_script.replace('{{USE_POSE_PRIOR_TRANSFORM_JSON}}', env_vars.get('USE_POSE_PRIOR_TRANSFORM_JSON', ''))
    user_data_script = user_data_script.replace('{{SOURCE_COORDINATE_NAME}}', env_vars.get('SOURCE_COORD_NAME', ''))
    user_data_script = user_data_script.replace('{{POSE_IS_WORLD_TO_CAM}}', env_vars.get('POSE_IS_WORLD_TO_CAM', ''))
    user_data_script = user_data_script.replace('{{ENABLE_ENHANCED_FEATURE_EXTRACTION}}', env_vars.get('ENABLE_ENHANCED_FEATURE_EXTRACTION', ''))
    user_data_script = user_data_script.replace('{{MATCHING_METHOD}}', env_vars.get('MATCHING_METHOD', ''))
    user_data_script = user_data_script.replace('{{RUN_TRAIN}}', env_vars.get('RUN_TRAIN', ''))
    user_data_script = user_data_script.replace('{{MODEL}}', env_vars.get('MODEL', ''))
    user_data_script = user_data_script.replace('{{MAX_STEPS}}', env_vars.get('MAX_STEPS', ''))
    user_data_script = user_data_script.replace('{{ENABLE_MULTI_GPU}}', env_vars.get('ENABLE_MULTI_GPU', ''))
    user_data_script = user_data_script.replace('{{ROTATE_SPLAT}}', env_vars.get('ROTATE_SPLAT', ''))
    user_data_script = user_data_script.replace('{{SPHERICAL_CAMERA}}', env_vars.get('SPHERICAL_CAMERA', ''))
    user_data_script = user_data_script.replace('{{SPHERICAL_CUBE_FACES_TO_REMOVE}}', env_vars.get('SPHERICAL_CUBE_FACES_TO_REMOVE', ''))
    user_data_script = user_data_script.replace('{{OPTIMIZE_SEQUENTIAL_SPHERICAL_FRAME_ORDER}}', env_vars.get('OPTIMIZE_SEQUENTIAL_SPHERICAL_FRAME_ORDER', ''))
    user_data_script = user_data_script.replace('{{REMOVE_BACKGROUND}}', env_vars.get('REMOVE_BACKGROUND', ''))
    user_data_script = user_data_script.replace('{{BACKGROUND_REMOVAL_MODEL}}', env_vars.get('BACKGROUND_REMOVAL_MODEL', ''))
    user_data_script = user_data_script.replace('{{MASK_THRESHOLD}}', env_vars.get('MASK_THRESHOLD', ''))
    user_data_script = user_data_script.replace('{{REMOVE_HUMAN_SUBJECT}}', env_vars.get('REMOVE_HUMAN_SUBJECT', ''))
    user_data_script = user_data_script.replace('{{LOG_VERBOSITY}}', env_vars.get('LOG_VERBOSITY', ''))
    user_data_script = user_data_script.replace('{{NUM_THREADS}}', str(multiprocessing.cpu_count()))
    user_data_script = user_data_script.replace('{{NUM_GPUS}}', '1')
    
    # Base64 encode
    encoded = base64.b64encode(user_data_script.encode('utf-8')).decode('utf-8')
    
    return encoded


def map_instance_type(sagemaker_instance_type: str) -> str:
    """
    Map SageMaker instance types to EC2 GPU instance types
    
    Args:
        sagemaker_instance_type: SageMaker instance type (e.g., ml.p3.2xlarge)
        
    Returns:
        EC2 instance type (e.g., p3.2xlarge)
    """
    # Remove 'ml.' prefix if present
    if sagemaker_instance_type.startswith('ml.'):
        ec2_type = sagemaker_instance_type[3:]
    else:
        ec2_type = sagemaker_instance_type
    
    # Map to cost-effective alternatives
    instance_map = {
        'p3.2xlarge': 'g4dn.xlarge',     # Much cheaper, good for inference
        'p3.8xlarge': 'g4dn.4xlarge',
        'p3.16xlarge': 'g5.4xlarge',
        'p3dn.24xlarge': 'g5.8xlarge',
        'p2.xlarge': 'g4dn.xlarge',
        'p2.8xlarge': 'g4dn.4xlarge',
        'p2.16xlarge': 'g4dn.8xlarge',
    }
    
    # Return mapped type or default to g4dn.xlarge
    return instance_map.get(ec2_type, 'g4dn.xlarge')
