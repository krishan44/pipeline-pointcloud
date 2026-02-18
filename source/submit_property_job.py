#!/usr/bin/env python3
"""
Quick Job Submission Script for 360 Property Image 3D Reconstruction

This script simplifies job submission for property scanning with 360 images.
It pre-configures common settings and handles all the complexity.

Usage:
    python submit_property_job.py --bucket gs-job-bucket-xxx --images my_property_images --mode "360"
"""

import os
import uuid
import json
import boto3
import argparse
from datetime import datetime

def submit_property_job(
    bucket_name: str,
    media_folder: str,
    mode: str = "standard",  # "standard" or "360" for equirectangular
    remove_background: bool = False,
    quality: str = "medium",  # "fast", "medium", "high"
    property_name: str = None
) -> str:
    """
    Submit a 3D reconstruction job for property images
    
    Args:
        bucket_name: S3 bucket name (from CDK/Terraform output)
        media_folder: Local folder containing images or path to video file
        mode: "standard" for regular perspective images, "360" for equirectangular
        remove_background: Whether to remove background from images
        quality: Processing quality - "fast", "medium", "high"
        property_name: Friendly name for logging
    
    Returns:
        Job UUID for tracking
    """
    
    s3 = boto3.client('s3')
    
    # Validate inputs
    if not os.path.exists(media_folder) and not media_folder.endswith(('.mp4', '.mov')):
        raise FileNotFoundError(f"Media folder/file not found: {media_folder}")
    
    if bucket_name == "":
        raise ValueError("Bucket name cannot be empty")
    
    # Generate unique job ID
    job_uuid = str(uuid.uuid4())
    media_name = os.path.basename(media_folder).split('.')[0]
    
    property_name = property_name or media_name
    
    print(f"\n{'='*60}")
    print(f"🏘️  Submitting 360 Property Reconstruction Job")
    print(f"{'='*60}")
    print(f"Property:     {property_name}")
    print(f"Mode:         {mode} ({'Equirectangular 360°' if mode == '360' else 'Standard perspective'})")
    print(f"Quality:      {quality}")
    print(f"Background:   {'Removed' if remove_background else 'Kept'}")
    print(f"Job ID:       {job_uuid}")
    print(f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure quality settings
    quality_settings = {
        "fast": {
            "maxSteps": "5000",
            "matchingMethod": "sequential",
            "instance_type": "ml.g5.4xlarge",
            "maxNumImages": "150"
        },
        "medium": {
            "maxSteps": "15000",
            "matchingMethod": "spatial",
            "instance_type": "ml.g6e.4xlarge",
            "maxNumImages": "300"
        },
        "high": {
            "maxSteps": "30000",
            "matchingMethod": "exhaustive",
            "instance_type": "ml.g5.12xlarge",
            "maxNumImages": "500"
        }
    }
    
    settings = quality_settings.get(quality, quality_settings["medium"])
    
    # Build job configuration
    job_config = {
        "uuid": job_uuid,
        "property_name": property_name,
        "submit_timestamp": datetime.now().isoformat(),
        "instanceType": settings["instance_type"],
        "logVerbosity": "info",
        "s3": {
            "bucketName": bucket_name,
            "inputPrefix": "media-input",
            "inputKey": media_name,
            "outputPrefix": "workflow-output"
        },
        "videoProcessing": {
            "maxNumImages": settings["maxNumImages"],
        },
        "imageProcessing": {
            "filterBlurryImages": "true"
        },
        "sfm": {
            "enable": "true",
            "softwareName": "glomap",
            "enableEnhancedFeatureExtraction": "false",
            "matchingMethod": settings["matchingMethod"],
            "posePriors": {
                "usePosePriorColmapModelFiles": "false",
                "usePosePriorTransformJson": {
                    "enable": "false",
                    "sourceCoordinateName": "arkit",
                    "poseIsWorldToCam": "true",
                },
            }
        },
        "training": {
            "enable": "true",
            "maxSteps": settings["maxSteps"],
            "model": "splatfacto",
            "enableMultiGpu": "false" if quality != "high" else "true",
            "rotateSplat": "true"
        },
        # 360 spherical camera settings
        "sphericalCamera": {
            "enable": "true" if mode == "360" else "false",
            "cubeFacesToRemove": "['down', 'up']",  # Exclude floor and ceiling
            "optimizeSequentialFrameOrder": "true"
        },
        "segmentation": {
            "removeBackground": "true" if remove_background else "false",
            "backgroundRemovalModel": "sam2",  # High-quality segmentation
            "maskThreshold": "0.6",
            "removeHumanSubject": "false"
        }
    }
    
    # Create local JSON file
    local_filename = f"./workflow-submissions/{job_uuid}.json"
    os.makedirs("./workflow-submissions", exist_ok=True)
    
    try:
        with open(local_filename, "w", encoding="utf-8") as f:
            json.dump(job_config, f, indent=2)
        print(f"✓ Created local job configuration")
    except Exception as e:
        print(f"✗ Error creating local config file: {e}")
        raise e
    
    # Upload to S3
    try:
        s3.upload_file(
            Filename=local_filename,
            Bucket=bucket_name,
            Key=f"workflow-input/{job_uuid}.json",
            ExtraArgs={
                "CacheControl": "no-cache",
                "Metadata": {
                    "property": property_name,
                    "mode": mode,
                    "quality": quality
                }
            }
        )
        print(f"✓ Uploaded job configuration to S3")
    except Exception as e:
        print(f"✗ Error uploading to S3: {e}")
        raise e
    
    print(f"\n{'='*60}")
    print(f"✅ JOB SUBMITTED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\n📍 Tracking Information:")
    print(f"   Job ID:        {job_uuid}")
    print(f"   Output Path:   s3://{bucket_name}/workflow-output/{job_uuid}/")
    print(f"\n📊 Monitoring:")
    print(f"   1. AWS Console → Step Functions → Executions")
    print(f"   2. Search for: {job_uuid}")
    print(f"   3. Check email for completion notification")
    print(f"\n⏱️  Estimated Processing Time:")
    time_estimates = {
        "fast": "20-30 minutes",
        "medium": "45-90 minutes",
        "high": "2-4 hours"
    }
    print(f"   {time_estimates.get(quality, '1-2 hours')}")
    print(f"\n💾 Output Files:")
    print(f"   • point_cloud.ply       - 3D Point Cloud")
    print(f"   • gaussian_splat.spz    - Optimized Splat Model")
    print(f"   • gaussian_splat.ply    - Splat as PLY")
    print(f"   • metadata.json         - Processing details")
    print(f"\n📥 Download results: aws s3 sync s3://{bucket_name}/workflow-output/{job_uuid}/ ./results/")
    print(f"\n{'='*60}\n")
    
    return job_uuid


def main():
    parser = argparse.ArgumentParser(
        description="Submit 360 property image reconstruction job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard property images
  python submit_property_job.py --bucket gs-job-bucket-xxx --images ./my_property

  # 360 equirectangular images with background removal
  python submit_property_job.py --bucket gs-job-bucket-xxx --images ./360_images --mode 360 --remove-bg

  # Video input with high quality
  python submit_property_job.py --bucket gs-job-bucket-xxx --images walkthrough.mp4 --quality high

  # Fast processing with custom property name
  python submit_property_job.py --bucket gs-job-bucket-xxx --images ./images --quality fast --name "123 Main St"
        """
    )
    
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket name from CDK/Terraform deployment"
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to image folder or video file"
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "360"],
        default="standard",
        help="Image mode: standard (perspective) or 360 (equirectangular)"
    )
    parser.add_argument(
        "--remove-bg",
        action="store_true",
        help="Enable background removal"
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "medium", "high"],
        default="medium",
        help="Processing quality (affects cost and time)"
    )
    parser.add_argument(
        "--name",
        help="Friendly property name for tracking"
    )
    
    args = parser.parse_args()
    
    try:
        job_uuid = submit_property_job(
            bucket_name=args.bucket,
            media_folder=args.images,
            mode=args.mode,
            remove_background=args.remove_bg,
            quality=args.quality,
            property_name=args.name
        )
        print(f"\n💡 TIP: Keep the job ID '{job_uuid}' for tracking!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
