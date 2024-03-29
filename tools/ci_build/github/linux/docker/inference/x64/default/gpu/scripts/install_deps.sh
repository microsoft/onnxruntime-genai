#!/bin/bash
set -e -x

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

cd /
rm -rf /tmp/src
