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
    print("filename", filename)

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
    print("dest_folder", dest_folder)
    os.rename(dest_folder, extracted_folder)

    # Clean up the downloaded file
    os.remove(filename)

    print("Downloading Onnxruntime, extraction, and renaming completed successfully.")


def ort_url_generator(ort_version, is_gpu, alt_cuda_version) -> str:
    gpu_suffix = f"-cuda{alt_cuda_version}" if alt_cuda_version is not None else "-gpu"
    return (f'https://github.com/microsoft/onnxruntime/releases/download/'
            f'v{ort_version}'
            f'/onnxruntime-osx-arm64{gpu_suffix if is_gpu else ""}-{ort_version}.tgz')


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ort_version", "-v", type=str, default="1.17.0")
    parser.add_argument("--is_gpu", "-g", type=bool, default=False,
                        help="Whether to use GPU or not. Default is False.")
    parser.add_argument("--alt_cuda_version", "-a", type=str, default=None,
                        help="Whether to use next alternative CUDA or not. Default is None.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    # URL of the file to download
    ort_url = ort_url_generator(args.ort_version, args.is_gpu, args.alt_cuda_version)
    # Name of the folder after extraction
    extracted_folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ort")
    print("ort_url:", ort_url)
    print("extracted_folder_name", extracted_folder_name)
    download_extract_rename(ort_url, extracted_folder_name)
