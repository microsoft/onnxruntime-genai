#!/bin/bash
set -e -x
pushd .
PYTHON_EXES=("/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12" "/opt/python/cp313-cp313/bin/python3.13" "/opt/python/cp314-cp314/bin/python3.14")

popd
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r requirements.txt
done

# Install Rust
export RUSTUP_HOME=/usr/.rustup 
export CARGO_HOME=/usr/.cargo 
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=1.86.0
chmod -R 777 /usr/.rustup
chmod -R 777 /usr/.cargo
