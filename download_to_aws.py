import os
import boto3
import subprocess

# AWS S3 Configuration
s3 = boto3.client('s3')
bucket_name = 'snapsorttestbucket2'  # Replace with your S3 bucket name


def download_packages(requirements_file):
    # Download all packages specified in requirements.txt
    subprocess.run(['pip', 'download', '-r', requirements_file, '-d', 'packages'], check=True)


def upload_to_s3(local_dir, bucket_name):
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            file_path = os.path.join(root, file)
            s3_key = os.path.relpath(file_path, local_dir)  # Maintain folder structure in S3
            try:
                s3.upload_file(file_path, bucket_name, s3_key)
                print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
            except Exception as e:
                print(f"Failed to upload {file_path}: {e}")


if __name__ == '__main__':
    requirements_file = 'requirements.txt'

    # Step 1: Download packages locally
    download_packages(requirements_file)

    # Step 2: Upload downloaded packages to S3
    upload_to_s3('packages', bucket_name)
