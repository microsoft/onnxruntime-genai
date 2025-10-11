'''
Use this script to leverage python to find the stubgen installation.
'''

from pathlib import Path
from argparse import ArgumentParser

def install_package(package_name: str, version: str | None = None) -> bool:
    import subprocess
    import sys
    # Check if the package is already installed, is so, return False, meaning we did not install it.
    try:
        __import__(package_name)
        return False
    except ImportError:
        pass
    # First try install with --user, otherwise try install without --user.
    package_spec = f"{package_name}=={version}" if version else package_name
    print(f"Installing package: {package_spec}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package_spec])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])
    return True

def try_uninstall_package(package_name: str) -> None:
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])
    except subprocess.CalledProcessError:
        pass

def generate_stubs(package: str, output: Path) -> None:
    from mypy.stubgen import main as stubgen_main
    stubgen_main(['-p', package, '-o', str(output)])

def fix_stubs(root: Path) -> None:
    # Try fix all .pyi files, though only the one generated from the native library needs fixing.
    stub_files = root.rglob('*.pyi')
    for stub_file in stub_files:
        content = stub_file.read_text()
        fixed_content = content.replace("Oga", "")
        stub_file.write_text(fixed_content)

def clean_up_pycache(root: Path) -> None:
    import shutil
    pycache_dirs = root.rglob('__pycache__')
    for pycache_dir in pycache_dirs:
        shutil.rmtree(pycache_dir, ignore_errors=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('-r', '--wheel_root', type=Path, required=True, help='Wheel root directory containing __init__.py')
    parser.add_argument('-v', '--ort-version', type=str, help='onnxruntime version to install, if not specified, the latest version will be installed')
    args = parser.parse_args()

    try:
        onnxruntime_installed_by_us = install_package('onnxruntime', args.ort_version)
        mypy_installed_by_us = install_package('mypy')
    except Exception as e:
        print(f"Failed to install dependencies on this platform:\n{e}")
        print("Skipping type stub generation.")
        return

    wheel_root = Path(args.wheel_root).resolve()
    generate_stubs(wheel_root.name, wheel_root.parent)
    fix_stubs(wheel_root)
    clean_up_pycache(wheel_root)

    if onnxruntime_installed_by_us:
        try_uninstall_package('onnxruntime')
    if mypy_installed_by_us:
        try_uninstall_package('mypy')

if __name__ == '__main__':
    main()
