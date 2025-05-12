import os
import zipfile
import shutil
import gdown

url = "https://drive.google.com/uc?id=1FqDDDvXwCDOzWM87Chnp6sgHPaY_4ayE"
output_zip = "models_download.zip"
gdown.download(url, output_zip, quiet=False)

unzip_dir = "temp_models"
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

target_dir = "models"
os.makedirs(target_dir, exist_ok=True)

subfolders = ["patchcore", "rf_detr"]
for root, dirs, files in os.walk(unzip_dir):
    for folder_name in subfolders:
        if folder_name in dirs:
            src = os.path.join(root, folder_name)
            dst = os.path.join(target_dir, folder_name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            print(f"Moved {folder_name} to {dst}")

os.remove(output_zip)
shutil.rmtree(unzip_dir)
print("All models set up successfully.")
