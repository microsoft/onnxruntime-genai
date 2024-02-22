import argparse

import os
import requests
import zipfile
import tarfile


def download_extract_rename(url, extracted_folder):
    # Extract file extension
    file_extension = url.split('.')[-1]
    # Name of the downloaded file
    filename = url.split('/')[-1]

    # Download the file
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)

    # Extract the contents based on file extension
    if file_extension == 'zip':
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall()
    elif file_extension == 'tgz' or file_extension == 'gz':
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
    else:
        print("Unsupported file format.")
        return

    # Rename the extracted folder
    dest_folder = os.path.splitext(filename)[0]  # remove the file extension
    os.rename(dest_folder, extracted_folder)

    # Clean up the downloaded file
    os.remove(filename)

    print("Downloading Onnxruntime, extraction, and renaming completed successfully.")


def ort_url_generator(ort_version, is_gpu, is_next_gen) -> str:
    gpu_suffix = "-cuda12" if is_next_gen else "-gpu"
    return (f'https://github.com/microsoft/onnxruntime/releases/download/'
            f'v{ort_version}'
            f'/onnxruntime-osx-arm64{gpu_suffix if is_gpu else ""}-{ort_version}.tgz')


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ort_version", "-v", type=str, default="1.17.0")
    parser.add_argument("--is_gpu", "-g", type=bool, default=False,
                        help="Whether to use GPU or not. Default is False.")
    parser.add_argument("--next_gen_cuda", "-n", type=bool, default=False,
                        help="Whether to use next generation CUDA or not. Default is False.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    # URL of the file to download
    url = ort_url_generator(args.ort_version, args.is_gpu, args.next_gen_cuda)
    # Name of the folder after extraction
    extracted_folder_name = f"{os.path.abspath(__file__)}/../ort"

    download_extract_rename(url, extracted_folder_name)
    # get_models(args)
