# 🎬 3D Gaussian Splatting Gradio Web UI

A web-based interface for uploading 360° panorama images and reconstructing high-quality 3D Gaussian Splatting models.

## Features

✨ **Easy Upload** - Drag & drop 360° images or video files  
🎯 **Advanced Controls** - Fine-tune SfM, training steps, background removal, and more  
📊 **Job Tracking** - Monitor reconstruction progress with real-time status updates  
☁️ **AWS Integration** - Built-in S3 and Step Functions workflow  
🤖 **GPU-Optimized** - Runs on GPU-enabled infrastructure for fast processing  

## Quick Start

### Prerequisites

- Python 3.11+
- AWS credentials configured (with S3 and Step Functions access)
- Gradio and boto3 installed

### Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set your AWS environment variables:

```bash
export AWS_REGION=eu-west-2
export S3_BUCKET=your-bucket-name
export STEP_FUNCTION_ARN=arn:aws:states:eu-west-2:ACCOUNT:stateMachine:3dgs-workflow
```

Authenticate with AWS:

```bash
aws configure
# or verify existing credentials:
aws sts get-caller-identity
```

## Running the Application

### Local Development

```bash
python app.py
```

Open browser to: **http://localhost:7860**

- Start the Gradio interface:

  ```bash
  cd source/Gradio
  python generate_splat_gradio.py
  ```

- Open your web browser and navigate to the URL displayed in the terminal (typically `http://127.0.0.1:7860`)

## Interface Components

The interface is organized into several sections:

### Processing Components

- `AWS Workload`: For AWS Settings specific to the infrastructure and services.
- `Advanced Settings`: Settings to guide the processing of the gaussian splat
  Key Settings include:
  - `Max Number of Images`: Set the maximum number of images to process from a video (default: 300)
  - `Filter Blurry Images`: Toggle to filter out blurry images
  - `Background Removal`: Toggle to remove background of objects
  - `Use Pose Priors`: Enable/disable pose priors for camera positioning
  - `Select SfM Software and Parameters`: Optimize the 3D reconstruction initialization
  - `Select Splat Training Software and Parameters` : Use models and parameters suitable for your use-case
- `S3 Browser`: ability to interrogate the contents of the S3 bucket chosen for the pipeline and download and view assets from the S3 bucket
- `Debug:` a quick access to view the `.json` payload which will be sent with the job.

## How It Works

1. **Input**: Users upload video/images and configure processing parameters through the web interface.
2. **Processing**: The application processes the inputs and generates a configuration JSON that defines:
   - Image processing settings
   - Camera configuration
   - Model generation parameters
   - Video payload
   - Submission: Clicking the **Upload to AWS** button will generate the required `.json` config file and upload both the `.json` and the video to AWS. A unique UUID will be included in this file to differentiate the jobs.
   - **NB:** The upload sequencing to AWS is important, the video must be uploaded first followed by the `.json`. The S3 bucket has a S3 Trigger which is actioned by the file type `.json`. If the video has not been uploaded yet, you will receive an error the video file not available. The Gradio file already caters for this, but when building your own interface make sure to keep this in mind.
3. **Output**: The system generates a 3D Gaussian Splatting model based on the settings provided in the `.json`. and will be placed in your chosen output folder location as your S3 Output Prefix

## Tips for Best Results

- Start by capturing objects, recording a video while orbiting the object
- Ensure video taken has good consistent lighting and are in focus
- Provide sufficient overlap for better reconstruction
- Use consistent camera settings across all shots
- Avoid reflective or transparent surfaces for best results

## Troubleshooting

If you encounter package-related errors, try:

```bash
pip uninstall gradio gradio-client packaging -y
pip install packaging --upgrade
pip install gradio --upgrade
```
