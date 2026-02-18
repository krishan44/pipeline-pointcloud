# Complete Step-by-Step Deployment Guide for 360 Property 3D Reconstruction

## STEP 1: Prerequisites Setup

### 1.1 Install Required Software

On your local machine, install:

**For CDK Deployment:**
```bash
# Install Node.js (required for CDK)
# Windows: Download from https://nodejs.org/ (LTS version)

# Install Python 3.9+
# Windows: Download from https://www.python.org/ (3.9 or higher)

# Install AWS CLI v2
# Windows: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

# Install Docker Desktop
# Windows: https://www.docker.com/products/docker-desktop
```

**For Terraform Deployment:**
```bash
# Install Terraform
# Windows: Download from https://www.terraform.io/downloads

# Install AWS CLI v2 (same as above)
# Install Docker Desktop (same as above)
```

### 1.2 Configure AWS Credentials

```bash
# Open Command Prompt/PowerShell and configure AWS credentials
aws configure

# You'll be prompted for:
# AWS Access Key ID: [your-access-key]
# AWS Secret Access Key: [your-secret-key]
# Default region name: us-east-1 (or your preferred region)
# Default output format: json
```

### 1.3 Verify AWS Account Permissions

Ensure your AWS IAM user has permissions for:
- S3 (bucket creation, uploads)
- Lambda
- SageMaker
- Step Functions
- DynamoDB
- ECR
- CloudFormation (for CDK)
- IAM (role creation)

---

## STEP 2: Deploy Infrastructure

### Option A: CDK Deployment (Recommended for AWS-native approach)

#### Step 2A.1: Configure CDK Settings

1. Navigate to the deployment folder:
```bash
cd deployment/cdk
```

2. Edit `config.json`:
```json
{
    "comment": "NOTE: BEFORE DEPLOYING STACK, UPDATE THE ACCOUNT ID AND REGION BELOW",
    "accountId": "YOUR-AWS-ACCOUNT-ID",      # ← Update with your 12-digit AWS account ID
    "region": "us-east-1",                    # ← Use your preferred region
    "constructNamePrefix": "3dgs",
    "s3TriggerKey": "workflow-input",
    "adminEmail": "your-email@example.com",   # ← Update with your email for notifications
    "maintainS3ObjectsOnStackDeletion": "true"
}
```

To find your AWS Account ID:
```bash
aws sts get-caller-identity
# Look for "Account" field in the output
```

#### Step 2A.2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install CDK CLI globally (if not already installed)
npm install -g aws-cdk
```

#### Step 2A.3: Bootstrap CDK Environment

```bash
# Only needed first time per region per account
cdk bootstrap aws://YOUR-ACCOUNT-ID/us-east-1
```

#### Step 2A.4: Synthesize and Deploy

```bash
# Validate the CDK stack
cdk synth

# Review what will be created (optional)
cdk diff

# Deploy the infrastructure
cdk deploy

# When prompted "Do you wish to proceed?", type: y
# Deployment takes 5-15 minutes
```

**Note down the outputs** - You'll need the S3 bucket name. They appear at the end of deployment like:
```
Outputs:
3dgsInfraStack.S3JobBucketName = gs-job-bucket-xxx
3dgsInfraStack.ECRRepositoryURI = xxx.dkr.ecr.us-east-1.amazonaws.com/3dgs-repo
```

---

## STEP 3: Build and Push Docker Container to ECR

### 3.1 Authenticate Docker with ECR

```bash
# Get the login command (replace region and account with yours)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR-ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com

# If successful, you'll see: "Login Succeeded"
```

### 3.2 Build the Docker Image

```bash
# Navigate to container directory
cd source/container

# Build the Docker image (this takes 10-20 minutes)
docker build -t 3dgs-splat:latest .

# Verify the image was created
docker images | grep 3dgs-splat
```

### 3.3 Tag and Push to ECR

```bash
# Get your ECR repository URI from CDK/Terraform outputs (e.g., xxx.dkr.ecr.us-east-1.amazonaws.com/3dgs-repo)
set ECR_REPO_URI=YOUR-ECR-REPOSITORY-URI

# Tag the image for ECR
docker tag 3dgs-splat:latest %ECR_REPO_URI%:latest

# Push to ECR
docker push %ECR_REPO_URI%:latest

# Verify in ECR console
# AWS Console → ECR → Repositories → 3dgs-repo (should show "latest" tag)
```

---

## STEP 4: Upload 360 Property Images to S3

### 4.1 Prepare Your 360 Images

1. **Collect 360 property images** (equirectangular or standard perspective images)
   - Format: JPG, PNG
   - Recommended: 300-500 images for best results
   - For equirectangular images: set `sphericalCamera.enable: "true"` in job config
   - For standard perspective: `sphericalCamera.enable: "false"`

2. **Organize images** in a local folder:
```
my_property_images/
├── image_001.jpg
├── image_002.jpg
├── image_003.jpg
└── ... (up to 300+ images)
```

### 4.2 Create S3 Folder Structure

```bash
# Get bucket name from CDK/Terraform output
set BUCKET_NAME=gs-job-bucket-xxx

# Create input folder in S3 (using AWS CLI)
aws s3 cp NUL s3://%BUCKET_NAME%/media-input/

# Verify folder created
aws s3 ls s3://%BUCKET_NAME%/media-input/
```

### 4.3 Upload Images to S3

**Option 1: Using AWS CLI (fastest for many files)**
```bash
# Upload all images to S3
aws s3 sync my_property_images/ s3://%BUCKET_NAME%/media-input/

# Verify upload
aws s3 ls s3://%BUCKET_NAME%/media-input/ --recursive
```

**Option 2: Using AWS Console**
1. Open AWS S3 Console
2. Navigate to your bucket
3. Go to `media-input/` folder
4. Click "Upload"
5. Select all images and upload

### 4.4 Alternative: Upload Video instead

If you have a video of the property walkthrough:
```bash
# Upload video file
aws s3 cp property_walkthrough.mp4 s3://%BUCKET_NAME%/media-input/
```

---

## STEP 5: Configure the Job JSON

### 5.1 Edit the generate_splat.py Script

Open [generate_splat.py](generate_splat.py) (already open in your editor):

```python
# Line 24-25: Update these values
s3_bucket_name = "gs-job-bucket-xxx"           # ← Your bucket name from CDK/Terraform
media_filename = "property_images"             # ← Folder name or video filename
instance_type = "ml.g6e.4xlarge"              # For GPU compute
```

### 5.2: Configure Pipeline Settings

In the same file, update the `file_contents` dictionary for 360 property scanning:

```python
file_contents = {
    "uuid": str(unique_uuid),
    "instanceType": "ml.g6e.4xlarge",  # GPU instance for faster processing
    "logVerbosity": "info",            # Logs: "debug", "info", or "error"
    
    "s3": {
        "bucketName": "gs-job-bucket-xxx",     # Your bucket
        "inputPrefix": "media-input",           # Where images are uploaded
        "inputKey": "property_images",          # Your folder/video name
        "outputPrefix": "workflow-output"       # Where results are saved
    },
    
    "videoProcessing": {
        "maxNumImages": "300",  # Max frames from video (or max images to use)
    },
    
    "imageProcessing": {
        "filterBlurryImages": "true"  # Remove blurry images automatically
    },
    
    "sfm": {  # Structure from Motion settings
        "enable": "true",
        "softwareName": "glomap",           # "colmap" or "glomap" (glomap is faster)
        "enableEnhancedFeatureExtraction": "false",
        "matchingMethod": "sequential",     # Feature matching: "sequential", "spatial", "vocab", "exhaustive"
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
        "maxSteps": "15000",           # Higher = better quality but slower
        "model": "splatfacto",         # Gaussian splat model
        "enableMultiGpu": "false",
        "rotateSplat": "true"          # Rotate for Gradio viewer
    },
    
    "sphericalCamera": {  # FOR 360 EQUIRECTANGULAR IMAGES
        "enable": "false",             # Set to "true" if using 360 images
        "cubeFacesToRemove": "['down', 'up']",  # Remove unwanted views
        "optimizeSequentialFrameOrder": "true"
    },
    
    "segmentation": {  # Background removal
        "removeBackground": "false",        # Set "true" to remove background
        "backgroundRemovalModel": "u2net",  # "u2net" or "sam2" (sam2 is better)
        "maskThreshold": "0.6",
        "removeHumanSubject": "false"
    }
}
```

### 5.3: For 360 Equirectangular Images

If your images are 360 equirectangular format:

```python
"sphericalCamera": {
    "enable": "true",                          # ← Enable spherical processing
    "cubeFacesToRemove": "['down', 'up']",    # Remove overhead/floor if needed
    "optimizeSequentialFrameOrder": "true"
}
```

---

## STEP 6: Submit the Job

### 6.1: Run the generate_splat.py Script

```bash
# Navigate to source directory
cd source

# Run the script to create and submit job
python generate_splat.py

# Success output:
# Successfully uploaded output metadata file: 
#     <uuid>.json to s3://gs-job-bucket-xxx/workflow-input
```

This will:
1. Generate a unique UUID for your job
2. Create a JSON metadata file
3. Upload the configuration to S3 `workflow-input/` folder
4. **Automatically trigger the entire pipeline**

### 6.2: Or Submit Job Manually

If you prefer manual submission:

```bash
# Create local JSON file manually
# Copy the file_contents from generate_splat.py into a file: my_job.json

# Upload to S3
aws s3 cp my_job.json s3://gs-job-bucket-xxx/workflow-input/

# This triggers the workflow
```

---

## STEP 7: Monitor the Pipeline Processing

### 7.1: View Pipeline Status in AWS Console

**Method 1: Step Functions UI (Real-time monitoring)**
1. Go to AWS Console → Step Functions → State Machines
2. Click the state machine (named something like `3dgs-workflow-sm`)
3. Find your execution (search by UUID)
4. Watch real-time progress through each step:
   - ✓ Validate Input
   - ✓ Download Media
   - ✓ Video to Images (if video input)
   - ✓ Filter Blurry Images
   - ✓ Structure from Motion (SfM) - Generates point cloud
   - ✓ Train Gaussian Splats
   - ✓ Post-Process & Export
   - ✓ Mark Complete

**Method 2: CloudWatch Logs**
1. AWS Console → CloudWatch → Log Groups
2. Find logs for your SageMaker training job
3. Watch real-time processing logs

### 7.2: Via AWS CLI

```bash
# List recent executions
aws stepfunctions list-executions --state-machine-arn arn:aws:states:region:account:stateMachine:3dgs-workflow-sm

# Get specific execution details
aws stepfunctions describe-execution --execution-arn <your-execution-arn>

# View execution history
aws stepfunctions get-execution-history --execution-arn <your-execution-arn>
```

### 7.3: Check for Completion Notification

- **Processing time**: 30 seconds to 2 hours (depending on image count and instance type)
- **Email notification**: Check your email (provided in config.json) for completion notification
- SNS will send success/failure message with output S3 location

---

## STEP 8: Download and View Results

### 8.1: View Output Files

**Once completed, check S3 output folder:**

```bash
# List output files
aws s3 ls s3://gs-job-bucket-xxx/workflow-output/ --recursive

# Structure:
# workflow-output/
# ├── <uuid>/
# │   ├── point_cloud.ply          ← 3D Point Cloud (Polygon file format)
# │   ├── gaussian_splat.spz       ← Gaussian Splat (optimized 3D model)
# │   ├── gaussian_splat.ply       ← Splat as PLY format
# │   └── metadata.json            ← Processing metadata
```

### 8.2: Download Results Locally

```bash
# Download all outputs for your job
aws s3 sync s3://gs-job-bucket-xxx/workflow-output/<uuid>/ ./my_property_results/

# View results locally
cd my_property_results
```

### 8.3: View 3D Point Cloud

**Point Cloud View Options:**

1. **CloudCompare** (Free)
   - Download: https://www.cloudcompare.org/
   - Open: `point_cloud.ply` file
   - Excellent for point cloud analysis

2. **Potree Viewer** (Web-based)
   - https://potree.org/
   - Drag & drop your PLY file
   - Interactive 3D viewing

3. **MeshLab** (Free)
   - Download: https://www.meshlab.net/
   - Open PLY files with full analysis tools

4. **Gradio Web Interface** (Built-in)
   - The solution includes a Gradio viewer for Gaussian splat visualization
   - Check [Gradio/generate_splat_gradio.py](../source/Gradio/generate_splat_gradio.py)

---

## STEP 9: Understanding What Each Pipeline Component Does

### Sequential Pipeline Flow:

1. **Video to Images** (if video input)
   - Extracts frames from MP4/MOV → Individual images
   - Limiting to `maxNumImages` (e.g., 300)

2. **Filter Blurry Images**
   - Analyzes sharpness of each image
   - Removes blurry/low-quality images automatically

3. **Background Removal** (optional)
   - Uses deep learning to isolate property from background
   - Two models:
     - `u2net`: General object removal
     - `sam2`: High-quality point/mask-based removal

4. **Structure from Motion (SfM)** ← **Creates Point Cloud**
   - Detects feature points in images (corners, edges, textures)
   - Matches features across multiple images
   - Estimates camera position/orientation for each image
   - Triangulates 3D points where features match across images
   - Outputs: Camera poses + sparse 3D point cloud
   - **This is your raw 3D point cloud**

5. **Gaussian Splat Training** (optional, for better visualization)
   - Takes camera poses + point cloud
   - Optimizes 3D Gaussian primitives for each point
   - Refines colors and transparency
   - Creates high-quality, memory-efficient 3D representation
   - Outputs: Splat model (lighter than point cloud, renders faster)

6. **Export & Post-Processing**
   - Converts to standard formats: PLY, SPZ
   - Generates metadata
   - Uploads to S3

---

## STEP 10: Configuration Reference for 360 Properties

### Recommended Settings for Property Scanning

```json
{
    "For Interior Property Scans":
    {
        "sfm.softwareName": "glomap",
        "sfm.matchingMethod": "spatial",
        "imageProcessing.filterBlurryImages": "true",
        "segmentation.removeBackground": "false",
        "training.maxSteps": "20000",
        "sphericalCamera.enable": "false"
    },
    
    "For 360 Equirectangular Images":
    {
        "sphericalCamera.enable": "true",
        "sphericalCamera.cubeFacesToRemove": "['down', 'up']",
        "training.maxSteps": "15000",
        "videoProcessing.maxNumImages": "300"
    },
    
    "For Fast Processing (Lower Quality)":
    {
        "training.maxSteps": "5000",
        "sfm.enableEnhancedFeatureExtraction": "false",
        "segmentation.removeBackground": "false"
    },
    
    "For Best Quality (Slower)":
    {
        "training.maxSteps": "30000",
        "sfm.enableEnhancedFeatureExtraction": "true",
        "training.model": "splatfacto-big",
        "sfm.matchingMethod": "exhaustive",
        "instance_type": "ml.g5.12xlarge"
    }
}
```

---

## Troubleshooting

### Common Issues:

| Issue | Cause | Solution |
|-------|-------|----------|
| Docker build fails | Missing dependencies | Ensure Docker Desktop is running |
| ECR authentication fails | Incorrect region | Check AWS region in configure step |
| Images not found in S3 | Wrong folder name in config | Verify `inputKey` matches S3 folder name |
| Job remains "RUNNING" for >2 hours | Too many images or insufficient GPU | Check CloudWatch logs, consider ml.g5.12xlarge instance |
| Point cloud is empty or partial | Poor image overlap or resolution | Use images with 60-80% overlap, higher resolution |
| No email notification | SNS topic issue | Verify email in config.json was confirmed in SNS |

---

## Cost Estimation

- **SageMaker Training** (main cost): ~$3-8 per job (depending on instance type and processing time)
- **S3 Storage**: ~$0.02-0.05 per job (input + output)
- **Data Transfer**: ~$0.01 per job
- **Total per job**: ~$3-8 (~$278/month for 100 jobs)

See README.md for detailed cost breakdown.

---

## Next Steps

1. ✅ View your generated point cloud in CloudCompare or Potree
2. 📊 Experiment with different image sets and configurations
3. 🎨 Use the Gaussian splat in game engines (Unreal, Unity) or web viewers
4. 🔄 Automate job submission for batch property scanning
5. 🌐 Integrate with web application for real-time 3D property visualization

For detailed AWS deployment guide, see the Implementation Guide linked in README.md.
