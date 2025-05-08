#!/bin/bash
set -e -x
pushd .
PYTHON_EXES=("/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12")

popd
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r requirements.txt
  if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep release | sed 's/.*release //' | cut -d',' -f1)
    if [[ "$cuda_version" == 11* ]]; then
      ${PYTHON_EXE} -m pip install cupy-cuda11x
    elif [[ "$cuda_version" == 12* ]]; then
      ${PYTHON_EXE} -m pip install cupy-cuda12x
    fi
  fi
done

# Install Rust
export RUSTUP_HOME=/usr/.rustup 
export CARGO_HOME=/usr/.cargo 
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=1.86.0
chmod -R 777 /usr/.rustup
chmod -R 777 /usr/.cargo
