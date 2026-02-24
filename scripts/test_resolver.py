import os
import tarfile
import zipfile
from pathlib import Path

def resolve_input_file_path(dataset_path: str, filename: str, s3_input: str = ""):
    normalized_filename = os.path.normpath(str(filename).strip())
    file_basename = os.path.basename(normalized_filename)
    channel_roots = [dataset_path, "/opt/ml/input/data/training", "/opt/ml/input/data/train"]
    candidate_paths = []
    for root in channel_roots:
        if not root:
            continue
        candidate_paths.append(os.path.join(root, normalized_filename))
        candidate_paths.append(os.path.join(root, file_basename))
    if s3_input and s3_input.startswith("s3://"):
        s3_object_name = os.path.basename(s3_input.rstrip("/"))
        if s3_object_name:
            for root in channel_roots:
                if root:
                    candidate_paths.append(os.path.join(root, s3_object_name))
    for candidate in candidate_paths:
        if os.path.isfile(candidate):
            return candidate, os.path.dirname(candidate)
    search_roots = ["/opt/ml/input/data", dataset_path]
    extensions = {".zip", ".mp4", ".mov"}
    prioritized_names = [normalized_filename, file_basename]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for current_root, _, files in os.walk(root):
            for entry in files:
                ext = os.path.splitext(entry)[1].lower()
                if ext not in extensions:
                    continue
                full_path = os.path.join(current_root, entry)
                if entry in prioritized_names or full_path.endswith(normalized_filename):
                    return full_path, os.path.dirname(full_path)
    raise FileNotFoundError(f"Could not resolve input media '{filename}' in SageMaker input channels.")

# Prepare test environment
root = Path('.').resolve()
opt_ml = root / 'opt' / 'ml' / 'input' / 'data' / 'training'
opt_ml.mkdir(parents=True, exist_ok=True)
# create a small dummy image file and zip it
sample_dir = root / 'test_input'
sample_dir.mkdir(exist_ok=True)
(sample_dir / 'img1.jpg').write_bytes(b'\xff\xd8\xff')
zip_path = opt_ml / 'input_images.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write(sample_dir / 'img1.jpg', arcname='img1.jpg')
print(f"Created test zip at: {zip_path}")

# Run resolver
try:
    resolved, resolved_dir = resolve_input_file_path(str(root / 'opt' / 'ml' / 'input' / 'data' / 'training'), 'input_images.zip', '')
    print('Resolved path:', resolved)
    print('Resolved dataset dir:', resolved_dir)
except Exception as e:
    print('Resolver failed:', e)
