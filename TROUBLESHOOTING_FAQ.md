# FAQ & Troubleshooting Guide

## Frequently Asked Questions

### General Questions

**Q: What file formats do you support?**
A: 
- Image input: JPG, PNG (3000x3000+ pixels recommended)
- Video input: MP4, MOV (will be converted to images)
- Output formats: PLY (point cloud), SPZ (Gaussian splat), JSON (metadata)

**Q: How many images do I need?**
A: 
- Minimum: 50-100 images for basic 3D reconstruction
- Recommended: 300-500 images for property scans
- Maximum: 1000+ (but increases processing time/cost)
- For 360 properties: 300-500 equirectangular images ideal

**Q: What is the difference between point cloud and Gaussian splat output?**
A:
- **Point Cloud (PLY)**: Raw 3D points from SfM - sparse, lightweight, shows raw geometry
- **Gaussian Splat (SPZ)**: Optimized representation - higher quality, faster rendering, AI-enhanced
- You get both! Use point cloud for analysis, splat for visualization

**Q: Can I use this for outdoor properties?**
A: Yes! The pipeline handles both interior and exterior scans. For outdoor scenes with sky:
- Set `sphericalCamera.cubeFacesToRemove: ['back']` to exclude background
- Or enable background removal to isolate the property

**Q: How long does processing take?**
A:
- Fast mode: 20-30 minutes
- Medium mode: 45-90 minutes
- High mode: 2-4 hours
- Depends on: image count, resolution, instance type chosen

---

### Technical Questions

**Q: What AWS instance types are supported?**
A: 
- `ml.g5.4xlarge` - Standard (4 GPUs, 48GB RAM) - Default
- `ml.g5.8xlarge` - High performance (8 GPUs, 96GB RAM)
- `ml.g5.12xlarge` - Multi-GPU (8 GPUs, 192GB RAM with shared memory)
- `ml.g6e.4xlarge` - Latest generation (1 GPU, 16 vCPU) - Recommended for cost
- Higher instances = faster processing but higher cost

**Q: What's the difference between COLMAP and Glomap for SfM?**
A:
- **COLMAP**: Slower, incremental, very accurate
- **Glomap**: Faster, global structure-from-motion, good for large scene
- For property scans: Glomap recommended (faster)

**Q: What does "feature matching" do in SfM?**
A: Finds identical visual features (corners, textures) across images:
- `sequential`: Fast, good for sequential photos
- `spatial`: Balanced, considers image proximity
- `vocab`: Vocabulary-based matching
- `exhaustive`: Slow but most thorough
- For property: `spatial` or `sequential` recommended

**Q: Why is my point cloud sparse or incomplete?**
A: Possible causes:
1. Low image overlap (<50%) - retake with more overlap
2. Blurry images - filter enabled will remove them
3. Insufficient texture/features in scenes (white walls, blank surfaces)
4. Low image resolution - use higher resolution images
5. Too few images - try 500+ images

**Q: What's the difference between different background removal models?**
A:
- `u2net`: General purpose, decent for objects and buildings
- `u2net-human`: Optimized for human removal
- `sam2`: High-quality, mask-based, slower but best results
- For property scans: sam2 recommended if removing background

---

### Cost & Billing Questions

**Q: How much will this cost per job?**
A: Rough estimates:
- Fast mode (5k steps, ml.g5.4xlarge): $3-4
- Medium mode (15k steps, ml.g6e.4xlarge): $5-7
- High mode (30k steps, ml.g5.12xlarge): $10-15
- Plus: S3 storage (~$0.02-0.05), data transfer (~$0.01)
- **Total: $3-15 per job typically**

**Q: How do I estimate total monthly cost?**
A: 
```
Monthly cost = (Jobs per month) × (Cost per job)
Example: 100 jobs × $6 average = $600/month
This is ONLY for SageMaker training. Infrastructure costs ~$50-100/month
```

**Q: Can I reduce costs?**
A: Yes:
1. Use `--quality fast` mode
2. Reduce image count with `maxNumImages: 200`
3. Use smaller instance types (ml.g6e.4xlarge vs ml.g5.12xlarge)
4. Skip unnecessary processing (e.g., high feature extraction if not needed)
5. Set up AWS Budgets to monitor and alert

---

### Configuration Questions

**Q: How do I configure for 360 equirectangular images?**
A: Set in job config:
```json
"sphericalCamera": {
    "enable": "true",
    "cubeFacesToRemove": "['down', 'up']",
    "optimizeSequentialFrameOrder": "true"
}
```
The pipeline converts 360 images to cube faces for processing.

**Q: How do I enable background removal?**
A: In job config:
```json
"segmentation": {
    "removeBackground": "true",
    "backgroundRemovalModel": "sam2",    # Best quality
    "maskThreshold": "0.6",
    "removeHumanSubject": "false"        # Set true to remove people too
}
```

**Q: What's the difference between pose prior options?**
A: Pose priors provide camera position hints (optional):
- `usePosePriorColmapModelFiles`: Pre-computed camera poses from external tool
- `usePosePriorTransformJson`: JSON file with camera transformations
- For most cases: leave both `false` (let SfM compute poses)

**Q: How do I increase output quality?**
A: Multiple options:
1. Increase training steps: `"maxSteps": "30000"` (default 15000)
2. Better model: `"model": "splatfacto-big"` vs `"splatfacto"`
3. Enhanced feature extraction: `"enableEnhancedFeatureExtraction": "true"`
4. Use exhaustive matching: `"matchingMethod": "exhaustive"`
5. Higher resolution images: 3000x3000+, not 800x600
6. More images: 500+ instead of 300

---

## Troubleshooting

### Deployment Issues

#### Problem: "CommandNotFound: cdk"
**Cause**: CDK CLI not installed
**Solution**:
```bash
npm install -g aws-cdk
# Verify: cdk --version
```

#### Problem: "Could not connect to AWS"
**Cause**: AWS credentials not configured
**Solution**:
```bash
aws configure
# Enter your AWS Access Key ID and Secret Access Key
# Verify: aws sts get-caller-identity
```

#### Problem: "User: arn:aws:iam::... is not authorized"
**Cause**: IAM permissions insufficient
**Solution**:
- Ensure your IAM user has permissions for: S3, Lambda, SageMaker, DynamoDB, ECR, IAM, CloudFormation
- Contact AWS administrator to add permissions
- Or use admin credentials temporarily for deployment

#### Problem: "cdk deploy" hangs or times out
**Cause**: Large Docker image or network issues
**Solution**:
```bash
# Cancel (Ctrl+C) and try again
# Or check progress in CloudFormation console
# AWS Console → CloudFormation → Stacks
```

#### Problem: CDK fails with "Insufficient SageMaker Training Job Quota"
**Cause**: AWS account quota too low
**Solution**:
1. AWS Console → Service Quotas
2. Search for "SageMaker Training Job Usage"
3. Request quota increase to 100+
4. Takes ~15 minutes to be approved

---

### Docker Issues

#### Problem: "Docker daemon is not running"
**Cause**: Docker Desktop not started
**Solution**:
- Windows: Open Docker Desktop application
- It auto-starts if configured to do so

#### Problem: "failed to solve with frontend dockerfile.v0"
**Cause**: Docker build context or Dockerfile error
**Solution**:
```bash
# Make sure you're in source/container directory
cd source/container

# Clean and retry
docker system prune -a
docker build -t 3dgs-splat:latest .
```

#### Problem: "denied: User: arn:aws:iam::... is not authorized"
**Cause**: ECR authentication failed
**Solution**:
```bash
# Re-authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR-ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com

# Then retry push
docker push ECR-REPO-URI:latest
```

---

### Job Submission Issues

#### Problem: "NoSuchBucket" or bucket not found
**Cause**: Wrong bucket name or bucket doesn't exist
**Solution**:
```bash
# List your buckets
aws s3 ls

# Get bucket name from CDK/Terraform outputs
# Use the exact name including dashes and numbers
```

#### Problem: "Key (media file) not found in bucket"
**Cause**: Images haven't been uploaded or wrong folder name
**Solution**:
```bash
# Verify images uploaded
aws s3 ls s3://your-bucket/media-input/ --recursive

# Check inputKey in job config matches folder name
# If you have folder "my_photos", inputKey should be "my_photos"
```

#### Problem: "No such file or directory: ./workflow-submissions"
**Cause**: Directory doesn't exist yet
**Solution**:
```bash
# Create it manually
mkdir workflow-submissions

# Or let Python create it (newer versions do)
python submit_property_job.py ...
```

---

### Processing Issues

#### Problem: Job status stuck on "RUNNING" for >2 hours
**Cause**: Long processing time or infrastructure issue
**Solution**:
```bash
# Check CloudWatch logs
# AWS Console → CloudWatch → Log Groups
# Look for /aws/sagemaker/... logs
# Search for "error" or "exception"

# If no progress, may need:
# - Higher instance type (ml.g5.12xlarge)
# - Fewer images (reduce maxNumImages)
# - Different image set (current images may be problematic)
```

#### Problem: "TASK_FAILED: Image processing error"
**Cause**: Image quality or format issues
**Solution**:
1. Check image format: JPG/PNG only
2. Try smaller image resolution: 2000x2000 instead of 5000x5000
3. Verify images are not corrupted
4. For video: ensure MP4/MOV format, H.264 codec

#### Problem: "SfM failed to generate camera poses"
**Cause**: Insufficient image overlap or feature matching failed
**Solution**:
1. Ensure 60-80% image overlap between consecutive images
2. Try different matching method: `"matchingMethod": "spatial"`
3. Use higher resolution images
4. Reduce image count to focus on best images
5. Try different property/scene

#### Problem: Point cloud is very sparse or empty
**Cause**: SfM couldn't match features
**Solution**:
1. Check image quality: should be clear, well-lit
2. Verify overlap: retake images with more coverage
3. Check for mostly uniform surfaces (white walls, blank areas)
4. Try higher resolution images
5. Add more varied viewing angles

---

### Notification Issues

#### Problem: No email notification after job completes
**Cause**: SNS subscription or email configuration issue
**Solution**:
1. Check AWS SNS console:
   - AWS Console → SNS → Topics
   - Find topic named "3dgs-notifications" or similar
   - Verify your email is subscribed (Status should be "Confirmed", not "PendingConfirmation")
2. If still pending:
   - Check spam folder for SNS confirmation email
   - Click confirmation link in email
3. Verify email in config.json was correct
4. Send test notification via SNS console to test

#### Problem: Spam folder receiving notifications
**Cause**: Email filtering
**Solution**:
- Mark SNS notifications as "Not Spam" in email client
- Or create email filter rule to allow notifications@sns.amazonaws.com

---

### Output Issues

#### Problem: "Access Denied" when downloading results
**Cause**: S3 permissions or wrong credentials
**Solution**:
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Verify bucket access
aws s3 ls s3://your-bucket/workflow-output/

# Re-configure if needed
aws configure
```

#### Problem: PLY file won't open in viewer
**Cause**: File corrupted or viewer incompatibility
**Solution**:
1. Verify file downloaded completely: `dir` shows correct size
2. Try different viewer: CloudCompare, Potree, MeshLab
3. Check file format: `file point_cloud.ply` should show ASCII/Binary
4. Re-download if corrupted

#### Problem: Gaussian splat (SPZ) file not useful
**Cause**: Mismatch between SfM output and splat training
**Solution**:
1. Verify SfM succeeded: check `metadata.json` in outputs
2. Increase training steps: `"maxSteps": "30000"`
3. Use `splatfacto-big` model for better quality
4. Check that PLY point cloud exists first

---

### AWS Cost Management

#### Problem: Unexpected high bill
**Cause**: Running many jobs, forgotten instances
**Solution**:
```bash
# Stop any stuck SageMaker training jobs
aws sagemaker list-training-jobs --status-equals InProgress

# Cancel job if necessary
aws sagemaker stop-training-job --training-job-name <job-name>

# Set up budget alerts
# AWS Console → Budgets → Create Budget
# Set monthly limit and alert threshold
```

---

## Performance Optimization Tips

### For Faster Processing:
1. Use `--quality fast` mode for testing
2. Reduce `maxNumImages`: 150-200 instead of 300
3. Disable unnecessary processing:
   - Set `removeBackground: false` if not needed
   - Set `enableEnhancedFeatureExtraction: false`
4. Use faster SfM: `softwareName: glomap`, `matchingMethod: sequential`

### For Better Quality:
1. Use `--quality high` mode
2. Increase `maxSteps` to 30000+
3. Use `model: splatfacto-big`
4. Enable enhanced extraction: `enableEnhancedFeatureExtraction: true`
5. Use exhaustive matching: `matchingMethod: exhaustive`
6. Higher resolution images: 3000x3000+

### For Lower Costs:
1. Start with fast mode ($3-4)
2. Combine multiple small jobs if possible
3. Use `ml.g6e.4xlarge` instance instead of `ml.g5.12xlarge`
4. Reduce image count for iteration, increase only for final
5. Monitor with CloudWatch alarms

---

## Getting Help

### Resources:
- **GitHub**: Report issues at the repository
- **AWS Documentation**: https://docs.aws.amazon.com/
- **Open Source Project Docs**:
  - COLMAP: https://colmap.github.io/
  - Glomap: https://github.com/colmap/glomap
  - NerfStudio: https://docs.nerf.studio/
  - gsplat: https://github.com/nerfstudio-project/gsplat

### Debugging Steps:
1. Check CloudWatch logs for detailed errors
2. Review job metadata.json for processing info
3. Verify intermediate outputs exist (check S3)
4. Try simple test case first (few images)
5. Check AWS service quotas if hitting limits

---

## Quick Debugging Checklist

When something goes wrong:
- [ ] Check AWS CloudWatch logs
- [ ] Verify AWS credentials: `aws sts get-caller-identity`
- [ ] Confirm S3 bucket and paths
- [ ] Verify image format and quality
- [ ] Check image overlap and count
- [ ] Review error code and message
- [ ] Check AWS service quotas
- [ ] Try with minimal test images first
- [ ] Review configuration JSON for typos
- [ ] Consult the open source project documentation

---

**Remember**: Most issues are easily resolved by checking CloudWatch logs first!
