# Quick Start Checklist for 360 Property 3D Reconstruction

## ✅ PRE-DEPLOYMENT (Do Once)

### Prerequisites
- [ ] AWS Account with elevated permissions
- [ ] AWS CLI v2 installed
- [ ] Docker Desktop installed
- [ ] Python 3.9+ installed
- [ ] Node.js LTS installed (for CDK)
- [ ] Text editor for configuration files

### AWS Credentials Setup
- [ ] Run `aws configure` with your credentials
- [ ] Verify: `aws sts get-caller-identity` (should show your account)
- [ ] Save your **AWS Account ID** (12 digits)

---

## 📋 DEPLOYMENT (Choose One Path)

### 🛠️ Path A: AWS CDK Deployment (Recommended)

**Step 1: Configuration**
```
cd deployment/cdk
nano config.json
  - accountId: [Paste your 12-digit AWS Account ID]
  - region: us-east-1 (or your default region)
  - adminEmail: your-email@example.com
```
- [ ] AccountId filled in
- [ ] Region set correctly
- [ ] Admin email entered

**Step 2: Install & Deploy**
```bash
pip install -r requirements.txt
npm install -g aws-cdk
cdk bootstrap aws://YOUR-ACCOUNT-ID/us-east-1
cdk deploy
```
- [ ] Dependencies installed
- [ ] CDK bootstrapped
- [ ] Stack deployed (takes 5-15 minutes)
- [ ] **Save outputs** (especially S3 bucket name and ECR URI)

---

### 🌍 Path B: Terraform Deployment

**Step 1: Configuration**
```bash
cd deployment/terraform
nano terraform.tfvars
  - account_id = "YOUR-AWS-ACCOUNT-ID"
  - region = "us-east-1"
  - admin_email = "your-email@example.com"
```
- [ ] Account ID filled in
- [ ] Region set
- [ ] Email set

**Step 2: Deploy**
```bash
terraform init
terraform plan
terraform apply
terraform output > outputs.txt
```
- [ ] Terraform initialized
- [ ] Plan reviewed
- [ ] Infrastructure deployed
- [ ] **Save outputs** (S3 bucket, ECR URI)

---

## 🐳 DOCKER CONTAINER (Do Once)

**Step 1: Build Container**
```bash
cd source/container
docker build -t 3dgs-splat:latest .
```
- [ ] Build succeeded (10-20 minutes)
- [ ] No build errors

**Step 2: Push to ECR**
```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <YOUR-ACCOUNT-ID>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag 3dgs-splat:latest <ECR-REPO-URI>:latest
docker push <ECR-REPO-URI>:latest
```
- [ ] ECR authentication successful
- [ ] Image tagged
- [ ] Image pushed (visible in AWS ECR console)

---

## 🏘️ FOR EACH PROPERTY SCAN

### Step 1: Prepare Images
- [ ] Images collected (300-500 recommended for 360 scans)
- [ ] Format: JPG or PNG
- [ ] Organized in local folder: `my_property_images/`
- [ ] If 360: Equirectangular format confirmed

### Step 2: Upload to S3
```bash
# Upload folder of images
aws s3 sync my_property_images/ s3://gs-job-bucket-xxx/media-input/

# OR upload single video
aws s3 cp walkthrough.mp4 s3://gs-job-bucket-xxx/media-input/
```
- [ ] Images uploaded to S3
- [ ] Verified in AWS S3 console

### Step 3: Create & Submit Job Config

**Option A: Using the new submit script (EASIEST)**
```bash
cd source

# For standard perspective images
python submit_property_job.py \
  --bucket gs-job-bucket-xxx \
  --images my_property_images \
  --quality medium \
  --name "123 Main Street"

# For 360 equirectangular images
python submit_property_job.py \
  --bucket gs-job-bucket-xxx \
  --images my_property_images \
  --mode 360 \
  --quality high \
  --name "360 Tour - Downtown"
```
- [ ] Script executed successfully
- [ ] Job submitted
- [ ] **Save Job ID** (shows in output)

**Option B: Using original generate_splat.py**
```bash
nano source/generate_splat.py
# Edit:
#  - s3_bucket_name = "gs-job-bucket-xxx"
#  - media_filename = "my_property_images"
#  - Adjust settings in file_contents dict
# Then run:
python source/generate_splat.py
```
- [ ] Configuration updated
- [ ] Script executed
- [ ] Success message shown

### Step 4: Monitor Progress
```bash
# AWS Console Method
# 1. Go to AWS Console
# 2. Step Functions → State Machines
# 3. Find your execution by UUID
# 4. Watch real-time progress
```
- [ ] Execution found in Step Functions
- [ ] Status: RUNNING vs SUCCEEDED vs FAILED
- [ ] Check CloudWatch logs for detailed status

**Processing Time Estimates:**
- `--quality fast`: 20-30 minutes (~$3-4)
- `--quality medium`: 45-90 minutes (~$5-7)
- `--quality high`: 2-4 hours (~$10-15)

### Step 5: Receive Notification
- [ ] Check email for completion notification (from SNS)
- [ ] Notification shows output S3 path

### Step 6: Download Results
```bash
# Download all outputs
aws s3 sync s3://gs-job-bucket-xxx/workflow-output/<JOB-UUID>/ ./results/

# Or download specific file
aws s3 cp s3://gs-job-bucket-xxx/workflow-output/<JOB-UUID>/point_cloud.ply ./
```
- [ ] Files downloaded locally
- [ ] Files present: `point_cloud.ply`, `gaussian_splat.spz`, metadata

### Step 7: View Point Cloud
- [ ] CloudCompare installed: https://www.cloudcompare.org/
- [ ] Open `point_cloud.ply` in CloudCompare
- [ ] Visualize 3D reconstruction

---

## 📊 TROUBLESHOOTING

| Issue | Fix |
|-------|-----|
| `aws configure` fails | Download AWS CLI v2 from official AWS website |
| `cdk deploy` fails | Verify CDK bootstrap ran: `cdk bootstrap aws://ID/region` |
| Docker build fails | Ensure Docker Desktop is running |
| ECR push rejected | Run: `aws ecr get-login-password ...` command again |
| Images not found in S3 | Check folder name in `inputKey` matches S3 folder |
| Job stuck > 2 hours | Check CloudWatch logs; may need higher instance type |
| No email notification | Verify email address in config; check SNS subscriptions in AWS Console |
| Point cloud empty | Ensure image overlap ~60-80%; try different property or higher resolution |

---

## 🎯 QUICK COMMANDS REFERENCE

```bash
# Get your AWS Account ID
aws sts get-caller-identity

# List S3 buckets
aws s3 ls

# Upload images to S3
aws s3 sync ./my_images s3://bucket-name/media-input/

# List executions
aws stepfunctions list-executions --state-machine-arn <ARN>

# Download results
aws s3 sync s3://bucket-name/workflow-output/job-uuid/ ./results/

# Check job status
aws stepfunctions describe-execution --execution-arn <ARN>

# View CloudWatch logs
aws logs tail /aws/sagemaker/ --follow
```

---

## 💡 TIPS FOR BEST RESULTS

### For 360 Property Scans:
1. **Image Quality**: Use high-resolution images (3000x3000+ pixels)
2. **Image Overlap**: Ensure 60-80% overlap between consecutive images
3. **Lighting**: Consistent lighting throughout property (avoid extreme shadows)
4. **Coverage**: Capture from multiple heights and angles
5. **Remove Obstacles**: Remove people, moving objects before scanning

### Cost Optimization:
- Start with `--quality fast` for testing
- Use `--quality high` for final/client deliverables
- Batch multiple properties in one job when possible
- Set CloudWatch alarms to monitor costs

### Quality Optimization:
1. Use 360 mode for equirectangular images
2. Enable background removal if outdoor context not needed
3. Increase `maxSteps` from 15000 to 30000+ for more detail
4. Use multi-GPU instance (`ml.g5.12xlarge`) for complex properties

---

## 📞 SUPPORT RESOURCES

- GitHub Issues: Report bugs at the repository
- AWS Documentation: https://docs.aws.amazon.com/
- Open Source Libraries:
  - COLMAP: https://colmap.github.io/
  - Glomap: https://github.com/colmap/glomap
  - NerfStudio: https://docs.nerf.studio/
  - gsplat: https://github.com/nerfstudio-project/gsplat

---

## ✨ NEXT STEPS AFTER FIRST JOB

1. **Experiment** with different quality settings
2. **Batch Process** multiple properties
3. **Integrate** with web application
4. **Share** 3D files with clients via web viewer
5. **Scale** to production with AWS infrastructure

---

**Quick Reference:**
- **Simple job**: `python submit_property_job.py --bucket gs-bucket --images ./photos --name "123 Main St"`
- **Monitor**: AWS Console → Step Functions
- **Download**: `aws s3 sync s3://bucket/workflow-output/JOB-ID/ ./results/`
- **View**: Open `.ply` file in CloudCompare

🎉 **You're ready to generate 3D point clouds from 360 property images!**
