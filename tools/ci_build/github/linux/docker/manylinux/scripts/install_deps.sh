#!/bin/bash
set -e -x
pushd .
PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12")

popd
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r requirements.txt
done

# No release binary for ccache aarch64, so we need to build it from source.
if ! [ -x "$(command -v ccache)" ]; then
    ccache_url="https://github.com/ccache/ccache/archive/refs/tags/v4.8.tar.gz"
    pushd .
    curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o ccache_src.tar.gz $ccache_url
    mkdir ccache_main
    cd ccache_main
    tar -zxf ../ccache_src.tar.gz --strip=1

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local _DCMAKE_BUILD_TYPE=Release ..
    make
    make install
    which ccache
    popd
    rm -f ccache_src.tar.gz
    rm -rf ccache_src
fi

# Download a file from internet
function GetFile {
  local uri=$1
  local path=$2
  local force=${3:-false}
  local download_retries=${4:-5}
  local retry_wait_time_seconds=${5:-30}

  if [[ -f $path ]]; then
    if [[ $force = false ]]; then
      echo "File '$path' already exists. Skipping download"
      return 0
    else
      rm -rf "$path"
    fi
  fi

  if [[ -f $uri ]]; then
    echo "'$uri' is a file path, copying file to '$path'"
    cp "$uri" "$path"
    return $?
  fi

  echo "Downloading $uri"
  curl "$uri" -sSL --retry $download_retries --retry-delay $retry_wait_time_seconds --create-dirs -o "$path" --fail
  return $?
}
mkdir -p /tmp/src

cd /tmp/src

CPU_ARCH=$(uname -m)

# Install CMake
echo "Installing cmake"
GetFile "https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-$CPU_ARCH.tar.gz" "/tmp/src/cmake.tar.gz"
tar -zxf /tmp/src/cmake.tar.gz --strip=1 -C /usr

# Install Ninja
echo "Installing Ninja"
GetFile https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
pushd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin
popd

# Install Rust
export RUSTUP_HOME=/usr/.rustup 
export CARGO_HOME=/usr/.cargo 
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=1.82.0
chmod -R 777 /usr/.rustup
chmod -R 777 /usr/.cargo

cd /
rm -rf /tmp/src
