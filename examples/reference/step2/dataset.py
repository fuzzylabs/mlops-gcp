import glob
import os
from google.cloud import storage

client = storage.Client(
    project="fuzzylabs"
)


def upload_from_directory(directory_path: str, dest_bucket_name: str, dest_blob_name: str):
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    bucket = client.get_bucket(dest_bucket_name)
    for local_file in rel_paths:
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def upload_dataset(src="../data/fashion_mnist", dest_bucket="fashion_mnist", dest_dir="dataset/"):
    upload_from_directory(src, dest_bucket, dest_dir)


