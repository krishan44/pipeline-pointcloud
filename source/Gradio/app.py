#!/usr/bin/env python3
"""
Gradio Web UI for 3D Gaussian Splatting Reconstruction
Allows users to upload 360° panorama images and submit jobs to AWS
"""

import os
import json
import uuid
import boto3
import logging
import gradio as gr
from datetime import datetime
from botocore.exceptions import ClientError
from pathlib import Path
import zipfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Client Setup
s3_client = boto3.client('s3')
stepfunctions_client = boto3.client('stepfunctions')

# Configuration
CONFIG = {
    'REGION': os.getenv('AWS_REGION', 'eu-west-2'),
    'S3_BUCKET': os.getenv('S3_BUCKET', '3dgs-bucket-uqgdnb'),
    'STEP_FUNCTION_ARN': os.getenv('STEP_FUNCTION_ARN', 'arn:aws:states:eu-west-2:762233765429:stateMachine:3dgs-sfn-uqgdnb'),
    'ECR_IMAGE_URI': os.getenv('ECR_IMAGE_URI', '762233765429.dkr.ecr.eu-west-2.amazonaws.com/3dgs-ecr-repo-uqgdnb:latest'),
    'CONTAINER_ROLE_ARN': os.getenv('CONTAINER_ROLE_ARN', 'arn:aws:iam::762233765429:role/3dgs-container-role-uqgdnb'),
    'LAMBDA_COMPLETE_NAME': os.getenv('LAMBDA_COMPLETE_NAME', 'GSWorkflowBaseStack-LambdaWorkflowCompleteConstruc-jz8aScDJQjAJ'),
    'SNS_TOPIC_ARN': os.getenv('SNS_TOPIC_ARN', 'arn:aws:sns:eu-west-2:762233765429:GSWorkflowBaseStack-NotificationConstructNotificationTopic211862B9-iZhUklp0F1q7'),
    'MAX_UPLOAD_SIZE_MB': 500,  # 500 MB max
    # Support both lowercase and uppercase extensions (Gradio file validation is case-sensitive)
    'ALLOWED_FORMATS': ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.mp4', '.MP4', '.mov', '.MOV']
}

IMAGE_FORMATS = {'.png', '.jpg', '.jpeg'}
VIDEO_FORMATS = {'.mp4', '.mov'}
ALLOWED_FORMATS_SET = {ext.lower() for ext in CONFIG['ALLOWED_FORMATS']}
MIN_IMAGES_FOR_SFM = 3
MIN_PANORAMAS_FOR_SPHERICAL_SFM = 3

# If CDK outputs are available, prefer them over hard-coded defaults / env vars.
try:
    outputs_path = Path(__file__).resolve().parents[2] / 'deployment' / 'cdk' / 'outputs.json'
    env_update_path = Path(__file__).resolve().parents[2] / 'env-update.json'
    has_explicit_ecr_image = False

    # Prefer explicit image URI from deployment env update file when available.
    if env_update_path.exists():
        with env_update_path.open('r', encoding='utf-8') as f:
            env_update_data = json.load(f)
        variables = env_update_data.get('Variables', {}) if isinstance(env_update_data, dict) else {}
        if isinstance(variables, dict):
            updated_image_uri = variables.get('ECR_IMAGE_URI')
            if updated_image_uri:
                CONFIG['ECR_IMAGE_URI'] = updated_image_uri
                has_explicit_ecr_image = True

    if outputs_path.exists():
        with outputs_path.open('r', encoding='utf-8') as f:
            outputs_data = json.load(f)

        if 'GSWorkflowBaseStack' in outputs_data:
            base = outputs_data['GSWorkflowBaseStack']
            # Region
            CONFIG['REGION'] = base.get('Region', CONFIG['REGION'])
            # S3 bucket
            CONFIG['S3_BUCKET'] = base.get('S3BucketName', base.get('S3ConstructBucketName77DC70F6', CONFIG['S3_BUCKET']))
            # ECR image (construct using account from role ARN if present)
            container_role = base.get('ContainerRoleArn')
            account = None
            if container_role and isinstance(container_role, str):
                parts = container_role.split(':')
                if len(parts) >= 5:
                    account = parts[4]

            state_machine_name = base.get('StateMachineName')
            if account and state_machine_name:
                CONFIG['STEP_FUNCTION_ARN'] = f"arn:aws:states:{CONFIG['REGION']}:{account}:stateMachine:{state_machine_name}"

            ecr_name = base.get('ECRRepoName')
            if account and ecr_name and not has_explicit_ecr_image:
                CONFIG['ECR_IMAGE_URI'] = f"{account}.dkr.ecr.{CONFIG['REGION']}.amazonaws.com/{ecr_name}:latest"

            # Role / lambda / sns
            if base.get('ContainerRoleArn'):
                CONFIG['CONTAINER_ROLE_ARN'] = base['ContainerRoleArn']
            if base.get('LambdaWorkflowCompleteFunctionName'):
                CONFIG['LAMBDA_COMPLETE_NAME'] = base['LambdaWorkflowCompleteFunctionName']
            if base.get('SnsTopicArn'):
                CONFIG['SNS_TOPIC_ARN'] = base['SnsTopicArn']

except Exception:
    # Non-fatal: fall back to env/defaults
    pass

logger.info(f"Using ECR image for training jobs: {CONFIG['ECR_IMAGE_URI']}")

class GaussianSplattingUI:
    """Manages the Gaussian Splatting reconstruction workflow"""
    
    def __init__(self):
        self.upload_dir = Path('./uploads')
        self.upload_dir.mkdir(exist_ok=True)
        # Ephemeral in-process history only. This is not persisted across
        # restarts or shared across multiple workers.
        self.job_history = []
        
    def validate_inputs(self, email, job_name):
        """Validate user inputs"""
        if not email or '@' not in email:
            return False, "Invalid email address"
        if not job_name or len(job_name.strip()) == 0:
            return False, "Job name cannot be empty"
        if not CONFIG['S3_BUCKET']:
            return False, "S3 bucket not configured. Contact admin."
        if not CONFIG['STEP_FUNCTION_ARN']:
            return False, "Step Function not configured. Contact admin."
        return True, "Valid"
    
    def validate_files(self, files):
        """Validate uploaded files"""
        if not files:
            return False, "No files uploaded"
        total_size = 0
        file_exts = []
        for f in files:
            try:
                local_path, orig_name = self._resolve_file(f)
            except Exception:
                return False, "Could not read uploaded file metadata"

            if not local_path.exists():
                return False, f"Uploaded file missing: {orig_name}"

            file_size = local_path.stat().st_size
            total_size += file_size

            # Check file extension using original filename when available
            ext = Path(orig_name).suffix.lower()
            file_exts.append(ext)
            if ext not in ALLOWED_FORMATS_SET:
                return False, f"File format {ext} not supported. Allowed: {', '.join(CONFIG['ALLOWED_FORMATS'])}"

        has_images = any(ext in IMAGE_FORMATS for ext in file_exts)
        has_videos = any(ext in VIDEO_FORMATS for ext in file_exts)
        if has_images and has_videos:
            return False, "Please upload either image(s) or one video, not a mixed set"
        if has_videos and len(files) > 1:
            return False, "Please upload only one video file (.mp4 or .mov)"
        
        # Check total size
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > CONFIG['MAX_UPLOAD_SIZE_MB']:
            return False, f"Total upload size ({total_size_mb:.1f} MB) exceeds limit ({CONFIG['MAX_UPLOAD_SIZE_MB']} MB)"
        
        return True, f"Valid ({total_size_mb:.1f} MB)"
    
    def upload_to_s3(self, job_uuid, files):
        """Upload files to S3"""
        try:
            resolved_files = [self._resolve_file(f) for f in files]
            file_exts = [Path(orig_name).suffix.lower() for _, orig_name in resolved_files]
            all_images = all(ext in IMAGE_FORMATS for ext in file_exts)

            if all_images:
                archive_name = "input_images.zip"
                zip_path = self.upload_dir / f"{job_uuid}_{archive_name}"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for local_path, orig_name in resolved_files:
                        zf.write(str(local_path), arcname=orig_name)

                s3_key = f"input/{job_uuid}/{archive_name}"
                logger.info(f"Uploading {zip_path} to s3://{CONFIG['S3_BUCKET']}/{s3_key}")
                s3_client.upload_file(str(zip_path), CONFIG['S3_BUCKET'], s3_key)
                return True, f"Successfully uploaded image archive to S3", archive_name

            local_path, orig_name = resolved_files[0]
            s3_key = f"input/{job_uuid}/{orig_name}"
            logger.info(f"Uploading {local_path} to s3://{CONFIG['S3_BUCKET']}/{s3_key}")
            s3_client.upload_file(str(local_path), CONFIG['S3_BUCKET'], s3_key)
            return True, "Successfully uploaded video file to S3", orig_name
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return False, f"S3 upload failed: {str(e)}", None

    def _resolve_file(self, f):
        """Resolve different Gradio file input shapes to a local Path and original filename.

        Supports:
        - string paths
        - dicts with 'tmp_path'/'tempfile' and 'name'
        - file-like objects with .name
        """
        # Plain string path
        if isinstance(f, str):
            p = Path(f)
            return p, p.name

        # dict (Gradio may pass {'name':..., 'tmp_path':...})
        if isinstance(f, dict):
            tmp = f.get('tmp_path') or f.get('tempfile') or f.get('file')
            name = f.get('name') or (Path(tmp).name if tmp else None)
            if tmp:
                return Path(tmp), name
            # fallback to name only (not ideal)
            return Path(name), name

        # file-like object
        if hasattr(f, 'name'):
            try:
                p = Path(f.name)
                return p, p.name
            except Exception:
                pass

        # Unknown format
        raise ValueError('Unsupported file input format')

    def _count_uploaded_images(self, files):
        """Count uploaded image files from Gradio file payload."""
        image_count = 0
        for f in files or []:
            try:
                _, orig_name = self._resolve_file(f)
            except Exception:
                continue
            if Path(orig_name).suffix.lower() in IMAGE_FORMATS:
                image_count += 1
        return image_count
    
    def submit_job(self, email, job_name, files, 
                   filter_blurry=True, run_sfm=True, sfm_software="glomap",
                   generate_splat=True, max_steps=30000, remove_background=False,
                   spherical_camera=False, rotate_splat=True, instance_type="ml.g5.xlarge",
                   optimize_sequential_spherical_frame_order=True,
                   spherical_use_oval_nodes=False,
                   spherical_angled_up_views=False,
                   spherical_angled_down_views=False,
                   use_tripod_scale=False,
                   tripod_height_m=1.6,
                   enable_semantic_object_layer=False):
        """Submit reconstruction job to Step Functions"""
        
        # Validate inputs
        valid, msg = self.validate_inputs(email, job_name)
        if not valid:
            return f"❌ Validation Error: {msg}"
        
        # Validate files
        valid, msg = self.validate_files(files)
        if not valid:
            return f"❌ File Validation Error: {msg}"

        # Fast-fail underconstrained SfM jobs (common source of COLMAP sparse model failures)
        image_count = self._count_uploaded_images(files)
        if run_sfm and image_count > 0:
            if spherical_camera and image_count < MIN_PANORAMAS_FOR_SPHERICAL_SFM:
                return (
                    "❌ Input Validation Error: Spherical SfM needs at least "
                    f"{MIN_PANORAMAS_FOR_SPHERICAL_SFM} panorama viewpoints. "
                    f"You uploaded {image_count}. "
                    "Add more distinct camera positions (not just rotations) and try again."
                )
            if image_count < MIN_IMAGES_FOR_SFM:
                return (
                    "❌ Input Validation Error: SfM needs at least "
                    f"{MIN_IMAGES_FOR_SFM} images with overlap. "
                    f"You uploaded {image_count}."
                )
        
        # Generate job UUID
        job_uuid = str(uuid.uuid4())
        
        logger.info(f"Submitting job {job_uuid} for {email}")
        
        # Upload files to S3
        success, upload_msg, input_filename = self.upload_to_s3(job_uuid, files)
        if not success:
            return f"❌ Upload Failed: {upload_msg}"
        
        # Prepare Step Function input
        resolved_tripod_height = 0.0
        if use_tripod_scale:
            try:
                resolved_tripod_height = float(tripod_height_m)
            except (TypeError, ValueError):
                resolved_tripod_height = 0.0

        step_function_input = {
            "UUID": job_uuid,
            "EMAIL": email,
            "JOB_NAME": job_name,
            "MODEL_INPUT": f"s3://{CONFIG['S3_BUCKET']}/models/models.tar.gz",
            "S3_INPUT": f"s3://{CONFIG['S3_BUCKET']}/input/{job_uuid}",
            "S3_OUTPUT": f"s3://{CONFIG['S3_BUCKET']}/output/{job_uuid}",
            "FILENAME": input_filename,
            "INSTANCE_TYPE": instance_type,
            "FILTER_BLURRY_IMAGES": str(filter_blurry).lower(),
            "RUN_SFM": str(run_sfm).lower(),
            "SFM_SOFTWARE_NAME": sfm_software,
            "GENERATE_SPLAT": str(generate_splat).lower(),
            "MAX_STEPS": str(int(max_steps)),
            "REMOVE_BACKGROUND": str(remove_background).lower(),
            "SPHERICAL_CAMERA": str(spherical_camera).lower(),
            "OPTIMIZE_SEQUENTIAL_SPHERICAL_FRAME_ORDER": str(optimize_sequential_spherical_frame_order).lower(),
            "SPHERICAL_USE_OVAL_NODES": str(spherical_use_oval_nodes).lower(),
            "SPHERICAL_ANGLED_UP_VIEWS": str(spherical_angled_up_views).lower(),
            "SPHERICAL_ANGLED_DOWN_VIEWS": str(spherical_angled_down_views).lower(),
            "ROTATE_SPLAT": str(rotate_splat).lower(),
            "MEASURE_REFERENCE_TYPE": "tripod_height" if use_tripod_scale else "none",
            "TRIPOD_HEIGHT_M": str(resolved_tripod_height if use_tripod_scale else 0.0),
            "ENABLE_SEMANTIC_OBJECT_LAYER": str(enable_semantic_object_layer).lower(),
            # Configuration passed to Step Function for SageMaker training job
            "ECR_IMAGE_URI": CONFIG['ECR_IMAGE_URI'],
            "CONTAINER_ROLE_ARN": CONFIG['CONTAINER_ROLE_ARN'],
            "LAMBDA_COMPLETE_NAME": CONFIG['LAMBDA_COMPLETE_NAME'],
            "SNS_TOPIC_ARN": CONFIG['SNS_TOPIC_ARN']
        }
        
        try:
            response = stepfunctions_client.start_execution(
                stateMachineArn=CONFIG['STEP_FUNCTION_ARN'],
                name=f"3dgs-{job_uuid}",
                input=json.dumps(step_function_input)
            )
            
            execution_arn = response['executionArn']
            self.job_history.append({
                'uuid': job_uuid,
                'name': job_name,
                'email': email,
                'submitted_at': datetime.now().isoformat(),
                'execution_arn': execution_arn,
                'status': 'RUNNING'
            })
            
            logger.info(f"Job submitted successfully. Execution ARN: {execution_arn}")
            
            return (f"✅ Job Submitted Successfully!\n\n"
                   f"**Job UUID:** {job_uuid}\n"
                   f"**Job Name:** {job_name}\n"
                   f"**Email:** {email}\n"
                   f"**Execution ARN:** {execution_arn}\n\n"
                   f"You will receive email updates at {email}")
        
        except Exception as e:
            logger.error(f"Step Function submission failed: {str(e)}")
            return f"❌ Job Submission Failed: {str(e)}"
    
    def get_job_status(self, job_uuid):
        """Get status of a reconstruction job"""
        try:
            for job in self.job_history:
                if job['uuid'] == job_uuid:
                    execution_arn = job['execution_arn']
                    response = stepfunctions_client.describe_execution(executionArn=execution_arn)
                    
                    status = response['status']
                    output = response.get('output', 'N/A')
                    
                    status_text = f"**Status:** {status}\n"
                    if status == 'SUCCEEDED':
                        status_text += f"✅ Job completed successfully!\n"
                        status_text += f"**Output:** {output}\n"
                    elif status == 'FAILED':
                        cause = response.get('cause', 'Unknown error')
                        status_text += f"❌ Job failed: {cause}\n"
                    elif status == 'RUNNING':
                        status_text += f"🔄 Processing...\n"
                    
                    return status_text
            
            return f"❌ Job {job_uuid} not found"
        
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            return f"❌ Error fetching status: {str(e)}"
    
    def list_job_results(self, job_uuid):
        """List available results for a job from S3"""
        try:
            if not job_uuid.strip():
                return "❌ Please enter a Job UUID"
            
            prefix = f"output/{job_uuid}/"
            response = s3_client.list_objects_v2(
                Bucket=CONFIG['S3_BUCKET'],
                Prefix=prefix
            )
            
            if 'Contents' not in response or len(response['Contents']) == 0:
                return "❌ No results found for this Job UUID. Job may still be processing."
            
            files = []
            total_size = 0
            for obj in response['Contents']:
                key = obj['Key']
                size = obj['Size']
                filename = key.replace(prefix, '')

                # Skip empty folder markers
                if filename and not filename.endswith('/'):
                    size_mb = size / (1024 * 1024)
                    # Include the S3 key so users can copy the exact path if needed
                    files.append(f"{filename} ({size_mb:.2f} MB) — s3://{CONFIG['S3_BUCKET']}/{key}")
                    total_size += size
            
            if not files:
                return "❌ No result files found for this Job UUID."
            
            result_text = f"✅ Found {len(files)} file(s) - Total: {total_size / (1024 * 1024):.2f} MB\n\n"
            result_text += "**Available Files:**\n"
            for f in files:
                result_text += f"• {f}\n"
            
            return result_text
        
        except Exception as e:
            logger.error(f"Failed to list job results: {str(e)}")
            return f"❌ Error listing results: {str(e)}"
    
    def download_job_result(self, job_uuid, filename):
        """Download a specific result file from S3"""
        try:
            s3_key = None
            if not job_uuid.strip() or not filename.strip():
                return None, "❌ Please enter both Job UUID and filename"
            
            # Extract just the filename if it includes size info
            if ' (' in filename:
                filename = filename.split(' (')[0]
            # Build S3 key flexibly to accept several user inputs and fall back to suffix-matching
            filename = filename.strip().lstrip('/')

            # Normalize input that may include the size text
            if ' (' in filename:
                filename = filename.split(' (')[0]

            tried_keys = []

            def _download_key(bucket, key):
                download_path = Path('./downloads')
                prefix = f"output/{job_uuid}/"
                # Create a safe local path: if key is under the job prefix, preserve subfolders
                if key.startswith(prefix):
                    rel = key[len(prefix):]
                    local_file = download_path / rel
                else:
                    # fallback to using only the filename portion
                    local_file = download_path / Path(key).name

                local_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading s3://{bucket}/{key} to {local_file}")
                s3_client.download_file(bucket, key, str(local_file))
                return str(local_file)

            # If user provided a full key starting with 'output/', use it directly
            if filename.startswith("output/"):
                s3_key = filename
                tried_keys.append(s3_key)
                try:
                    local = _download_key(CONFIG['S3_BUCKET'], s3_key)
                    return local, f"✅ Downloaded: {Path(s3_key).name}"
                except ClientError:
                    # continue to fallback
                    pass

            # If user pasted a full s3:// URI, parse bucket and key
            if filename.startswith("s3://"):
                # strip scheme
                rest = filename[5:]
                parts = rest.split('/', 1)
                if len(parts) == 2:
                    bucket_name, obj_key = parts[0], parts[1]
                    tried_keys.append(f"s3://{bucket_name}/{obj_key}")
                    try:
                        local = _download_key(bucket_name, obj_key)
                        return local, f"✅ Downloaded: {Path(obj_key).name}"
                    except ClientError:
                        # fall through to other heuristics
                        pass

            # If user provided a path starting with the job UUID, prepend 'output/'
            if filename.startswith(f"{job_uuid}/"):
                s3_key = f"output/{filename}"
                tried_keys.append(s3_key)
                try:
                    local = _download_key(CONFIG['S3_BUCKET'], s3_key)
                    return local, f"✅ Downloaded: {Path(s3_key).name}"
                except ClientError:
                    pass

            # Primary candidate: output/<job_uuid>/<filename>
            s3_key = f"output/{job_uuid}/{filename}"
            tried_keys.append(s3_key)
            try:
                local = _download_key(CONFIG['S3_BUCKET'], s3_key)
                return local, f"✅ Downloaded: {Path(s3_key).name}"
            except ClientError:
                # Not found: try suffix-matching across objects under the job prefix
                prefix = f"output/{job_uuid}/"
                response = s3_client.list_objects_v2(Bucket=CONFIG['S3_BUCKET'], Prefix=prefix)
                matches = []
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        if key.endswith(filename):
                            matches.append(key)

                if len(matches) == 1:
                    s3_key = matches[0]
                    tried_keys.append(s3_key)
                    try:
                        local = _download_key(CONFIG['S3_BUCKET'], s3_key)
                        return local, f"✅ Downloaded: {Path(s3_key).name} (resolved from nested path)"
                    except ClientError as e:
                        logger.error(f"Failed to download resolved key {s3_key}: {str(e)}")
                        return None, f"❌ Download failed after resolving nested key: {str(e)}\nTried keys: {tried_keys}"
                elif len(matches) > 1:
                    # Multiple candidates found — show the options so user can pick the correct one
                    pretty = '\n'.join([f"- s3://{CONFIG['S3_BUCKET']}/{k}" for k in matches])
                    return None, (f"❌ Multiple files match '{filename}'. Please specify the full key or choose one of:\n{pretty}")
                else:
                    return None, f"❌ Download failed: object not found. Tried keys: {tried_keys}"
        
        except Exception as e:
            logger.error(f"Failed to download result for job {job_uuid} filename {filename}: {str(e)}")
            # Help the user by suggesting the exact S3 key attempted
            return None, f"❌ Download failed: {str(e)}\nTried S3 key: {s3_key or 'unknown'}"

# Initialize UI
ui = GaussianSplattingUI()

# Define Gradio interface
def create_gradio_interface():
    """Create the Gradio web UI"""
    
    with gr.Blocks(title="3D Gaussian Splatting Reconstruction", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🎬 3D Gaussian Splatting Reconstruction
        
        Upload your **360° panorama images or video** and we'll reconstruct a high-quality 3D Gaussian Splatting model!
        
        """)
        
        with gr.Tabs():
            
            # ===== SUBMIT JOB TAB =====
            with gr.TabItem("📤 Submit Job"):
                gr.Markdown("### Upload your 360° images or video")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        email = gr.Textbox(
                            label="Email Address",
                            placeholder="your-email@example.com",
                            info="We'll send you updates on job progress"
                        )
                        job_name = gr.Textbox(
                            label="Job Name",
                            placeholder="e.g., My 360 Office Building",
                            info="Descriptive name for this reconstruction"
                        )
                    
                    with gr.Column(scale=1):
                        uploaded_files = gr.File(
                            label="Upload 360° Images or Video",
                            file_count="multiple",
                            file_types=CONFIG['ALLOWED_FORMATS'],
                            interactive=True
                        )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    
                    with gr.Row():
                        with gr.Column():
                            instance_type = gr.Dropdown(
                                choices=[
                                    "ml.g5.xlarge",    # 1x A10G GPU - $1.41/hr
                                    "ml.g5.2xlarge",   # 1x A10G GPU - $1.52/hr  
                                    "ml.g5.4xlarge",   # 1x A10G GPU - $2.03/hr
                                    "ml.g5.12xlarge",  # 4x A10G GPU - $7.09/hr
                                    "ml.g6e.xlarge",   # 1x L4 GPU - $1.11/hr
                                    "ml.g6e.2xlarge",  # 1x L4 GPU - $1.35/hr
                                    "ml.g6e.4xlarge",  # 1x L4 GPU - $1.83/hr
                                    "ml.p3.2xlarge",   # 1x V100 GPU - $3.83/hr
                                ],
                                value="ml.g5.xlarge",
                                label="GPU Instance Type",
                                info="Select compute instance - affects cost and speed"
                            )
                            filter_blurry = gr.Checkbox(
                                label="Filter Blurry Images",
                                value=True,
                                info="Automatically remove blurry images"
                            )
                            run_sfm = gr.Checkbox(
                                label="Run SfM (Structure from Motion)",
                                value=True,
                                info="Extract camera poses from images"
                            )
                            sfm_software = gr.Radio(
                                ["glomap", "colmap"],
                                value="glomap",
                                label="SfM Software",
                                info="glomap: faster, colmap: more accurate"
                            )
                        
                        with gr.Column():
                            generate_splat = gr.Checkbox(
                                label="Generate Gaussian Splat",
                                value=True,
                                info="Train 3D Gaussian Splatting model"
                            )
                            max_steps = gr.Slider(
                                minimum=1000,
                                maximum=100000,
                                value=1500,
                                step=1000,
                                label="Training Steps",
                                info="1000-1500 for quick preview, higher for final quality"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            remove_background = gr.Checkbox(
                                label="Remove Background",
                                value=False,
                                info="Use SAM2 to segment and remove background"
                            )
                            spherical_camera = gr.Checkbox(
                                label="Spherical Camera Mode",
                                value=False,
                                info="For 360° panorama images - converts to 6 cube faces"
                            )
                            optimize_seq_spherical = gr.Checkbox(
                                label="Optimize Spherical Sequence",
                                value=True,
                                info="Keeps view ordering optimized for SfM"
                            )
                            spherical_use_oval_nodes = gr.Checkbox(
                                label="Spherical Oval Nodes",
                                value=False,
                                info="Adds extra node frames (can improve robustness, slower)"
                            )
                            spherical_angled_up_views = gr.Checkbox(
                                label="Spherical Angled Up Views",
                                value=False,
                                info="Adds extra upward views (slower)"
                            )
                            spherical_angled_down_views = gr.Checkbox(
                                label="Spherical Angled Down Views",
                                value=False,
                                info="Adds extra downward views (slower)"
                            )
                            rotate_splat = gr.Checkbox(
                                label="Rotate Splat",
                                value=True,
                                info="Align splat to standard orientation"
                            )
                            use_tripod_scale = gr.Checkbox(
                                label="Use Tripod Height for Scale",
                                value=False,
                                info="Estimate metric scale from camera height above floor"
                            )
                            tripod_height_m = gr.Number(
                                label="Tripod Height (m)",
                                value=1.60,
                                precision=3,
                                info="Measure floor to camera optical center (meters)"
                            )
                            enable_semantic_object_layer = gr.Checkbox(
                                label="Include Semantic Object Layer",
                                value=False,
                                info="Adds object-footprint overlay outputs for floorplans"
                            )
                
                submit_btn = gr.Button("🚀 Submit Reconstruction Job", size="lg", variant="primary")
                
                submission_output = gr.Markdown()
                
                submit_btn.click(
                    fn=ui.submit_job,
                    inputs=[email, job_name, uploaded_files, filter_blurry, run_sfm, 
                           sfm_software, generate_splat, max_steps, remove_background, 
                              spherical_camera, rotate_splat, instance_type,
                              optimize_seq_spherical, spherical_use_oval_nodes,
                              spherical_angled_up_views, spherical_angled_down_views,
                              use_tripod_scale, tripod_height_m, enable_semantic_object_layer],
                    outputs=submission_output
                )
            
            # ===== CHECK STATUS TAB =====
            with gr.TabItem("📊 Check Job Status"):
                gr.Markdown("### Check the status of your submitted jobs")
                
                with gr.Row():
                    job_uuid_input = gr.Textbox(
                        label="Job UUID",
                        placeholder="Paste your Job UUID here (from submission)",
                        info="Find this in your submission confirmation"
                    )
                    status_btn = gr.Button("Get Status", variant="primary", size="lg")
                
                status_output = gr.Markdown()
                
                status_btn.click(
                    fn=ui.get_job_status,
                    inputs=job_uuid_input,
                    outputs=status_output
                )
                
                gr.Markdown("""
                ### How to track your job:
                1. Submit a job in the "Submit Job" tab
                2. Copy the **Job UUID** from the confirmation message
                3. Paste it here and click "Get Status"
                4. Check back periodically for updates
                """)
            
            # ===== DOWNLOAD RESULTS TAB =====
            with gr.TabItem("⬇️ Download Results"):
                gr.Markdown("### Download your 3D model results")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        job_uuid_dl = gr.Textbox(
                            label="Job UUID",
                            placeholder="Paste your Job UUID here",
                            info="Find this in your submission confirmation"
                        )
                    with gr.Column(scale=1):
                        list_btn = gr.Button("📋 List Files", variant="secondary", size="lg")
                
                files_output = gr.Markdown()
                
                list_btn.click(
                    fn=ui.list_job_results,
                    inputs=job_uuid_dl,
                    outputs=files_output
                )
                
                gr.Markdown("---")
                gr.Markdown("### Download a specific file")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        filename_input = gr.Textbox(
                            label="Filename",
                            placeholder="e.g., point_cloud.ply or gaussian_splat.spz",
                            info="Name of the file to download (without the Job UUID path)"
                        )
                    with gr.Column(scale=1):
                        download_btn = gr.Button("⬇️ Download File", variant="primary", size="lg")
                
                with gr.Row():
                    download_file = gr.File(label="Downloaded File", visible=True)
                    download_status = gr.Markdown()
                
                download_btn.click(
                    fn=ui.download_job_result,
                    inputs=[job_uuid_dl, filename_input],
                    outputs=[download_file, download_status]
                )
                
                gr.Markdown("""
                ### Available output files:
                - **point_cloud.ply** - Point cloud (large, 10-500 MB)
                - **gaussian_splat.spz** - Compressed 3D Gaussian Splat model
                - **transforms.json** - Camera poses and metadata
                - **training_logs.txt** - Processing logs
                
                **Tip:** Click "List Files" first to see all available files for your job!
                """)
            
            # ===== INFO TAB =====
            with gr.TabItem("ℹ️ Info & Help"):
                gr.Markdown(f"""
                ### Configuration
                - **AWS Region:** {CONFIG['REGION']}
                - **S3 Bucket:** {CONFIG['S3_BUCKET']}
                - **ECR Image:** {CONFIG['ECR_IMAGE_URI']}
                - **Max Upload Size:** {CONFIG['MAX_UPLOAD_SIZE_MB']} MB
                - **Supported Formats:** {', '.join(CONFIG['ALLOWED_FORMATS'])}
                
                ### Pipeline Steps
                1. **Upload** - Your images/video to S3
                2. **Process** - Extract frames, filter, segment
                3. **SfM** - Estimate camera poses (GLOMAP/COLMAP)
                4. **Train** - Generate Gaussian Splatting model (NerfStudio)
                5. **Export** - Convert to PLY/SPZ format
                6. **Download** - Retrieve your model from S3
                
                ### Best Practices for 360° Images
                - **Format:** PNG or JPG (300+ images recommended)
                - **Resolution:** 3840x1920 or higher
                - **Coverage:** Full 360° panorama with overlap
                - **Equirectangular:** Recommended for panorama mode
                
                ### Support
                For issues or questions, contact: support@example.com
                """)
    
    return app

# Create and launch the app
if __name__ == "__main__":
    
    # Check configuration
    if not CONFIG['S3_BUCKET']:
        print("⚠️ Warning: S3_BUCKET not set. Set AWS environment variables:")
        print("   export S3_BUCKET=your-bucket-name")
        print("   export STEP_FUNCTION_ARN=arn:aws:states:region:account:stateMachine:name")
    
    app = create_gradio_interface()
    print("\n" + "="*60)
    print("🚀 Starting 3D Gaussian Splatting UI...")
    print("="*60)
    print("\nOpen your browser to: http://localhost:7860\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False
    )
