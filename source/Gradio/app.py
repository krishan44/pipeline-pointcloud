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

class GaussianSplattingUI:
    """Manages the Gaussian Splatting reconstruction workflow"""
    
    def __init__(self):
        self.upload_dir = Path('./uploads')
        self.upload_dir.mkdir(exist_ok=True)
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
            if ext not in CONFIG['ALLOWED_FORMATS']:
                return False, f"File format {ext} not supported. Allowed: {', '.join(CONFIG['ALLOWED_FORMATS'])}"
        
        # Check total size
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb > CONFIG['MAX_UPLOAD_SIZE_MB']:
            return False, f"Total upload size ({total_size_mb:.1f} MB) exceeds limit ({CONFIG['MAX_UPLOAD_SIZE_MB']} MB)"
        
        return True, f"Valid ({total_size_mb:.1f} MB)"
    
    def upload_to_s3(self, job_uuid, files):
        """Upload files to S3"""
        try:
            uploaded_count = 0
            for f in files:
                local_path, orig_name = self._resolve_file(f)
                s3_key = f"input/{job_uuid}/{orig_name}"
                logger.info(f"Uploading {local_path} to s3://{CONFIG['S3_BUCKET']}/{s3_key}")
                s3_client.upload_file(str(local_path), CONFIG['S3_BUCKET'], s3_key)
                uploaded_count += 1
            
            return True, f"Successfully uploaded {uploaded_count} file(s) to S3"
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return False, f"S3 upload failed: {str(e)}"

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
    
    def submit_job(self, email, job_name, files, 
                   filter_blurry=True, run_sfm=True, sfm_software="glomap",
                   generate_splat=True, max_steps=30000, remove_background=False,
                   spherical_camera=False, rotate_splat=True):
        """Submit reconstruction job to Step Functions"""
        
        # Validate inputs
        valid, msg = self.validate_inputs(email, job_name)
        if not valid:
            return f"❌ Validation Error: {msg}"
        
        # Validate files
        valid, msg = self.validate_files(files)
        if not valid:
            return f"❌ File Validation Error: {msg}"
        
        # Generate job UUID
        job_uuid = str(uuid.uuid4())
        
        logger.info(f"Submitting job {job_uuid} for {email}")
        
        # Upload files to S3
        success, upload_msg = self.upload_to_s3(job_uuid, files)
        if not success:
            return f"❌ Upload Failed: {upload_msg}"
        
        # Prepare Step Function input
        _, input_filename = self._resolve_file(files[0])  # Use first file's original name as primary input
        
        step_function_input = {
            "UUID": job_uuid,
            "EMAIL": email,
            "JOB_NAME": job_name,
            "S3_INPUT": f"s3://{CONFIG['S3_BUCKET']}/input/{job_uuid}",
            "S3_OUTPUT": f"s3://{CONFIG['S3_BUCKET']}/output/{job_uuid}",
            "FILENAME": input_filename,
            "FILTER_BLURRY_IMAGES": str(filter_blurry).lower(),
            "RUN_SFM": str(run_sfm).lower(),
            "SFM_SOFTWARE_NAME": sfm_software,
            "GENERATE_SPLAT": str(generate_splat).lower(),
            "MAX_STEPS": str(int(max_steps)),
            "REMOVE_BACKGROUND": str(remove_background).lower(),
            "SPHERICAL_CAMERA": str(spherical_camera).lower(),
            "ROTATE_SPLAT": str(rotate_splat).lower(),
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
                    files.append(f"{filename} ({size_mb:.2f} MB)")
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
            if not job_uuid.strip() or not filename.strip():
                return None, "❌ Please enter both Job UUID and filename"
            
            # Extract just the filename if it includes size info
            if ' (' in filename:
                filename = filename.split(' (')[0]
            
            s3_key = f"output/{job_uuid}/{filename}"
            download_path = Path('./downloads')
            download_path.mkdir(exist_ok=True)
            
            local_file = download_path / filename
            logger.info(f"Downloading {s3_key} to {local_file}")
            
            s3_client.download_file(CONFIG['S3_BUCKET'], s3_key, str(local_file))
            
            return str(local_file), f"✅ Downloaded: {filename}"
        
        except Exception as e:
            logger.error(f"Failed to download result: {str(e)}")
            return None, f"❌ Download failed: {str(e)}"

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
                                value=30000,
                                step=1000,
                                label="Training Steps",
                                info="More steps = better quality, longer training"
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
                            rotate_splat = gr.Checkbox(
                                label="Rotate Splat",
                                value=True,
                                info="Align splat to standard orientation"
                            )
                
                submit_btn = gr.Button("🚀 Submit Reconstruction Job", size="lg", variant="primary")
                
                submission_output = gr.Markdown()
                
                submit_btn.click(
                    fn=ui.submit_job,
                    inputs=[email, job_name, uploaded_files, filter_blurry, run_sfm, 
                           sfm_software, generate_splat, max_steps, remove_background, 
                           spherical_camera, rotate_splat],
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
