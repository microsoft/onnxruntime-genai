#!/bin/bash

# This script will run build_aar_package.py to build Android AAR and copy all the artifacts
# to a given folder for publishing to Maven Central, or building nuget package
# This script is intended to be used in CI build only

set -e
set -x

# TODO: Update to the most recent python version we can use
export PATH=/opt/python/cp38-cp38/bin:$PATH

ls /build

# TODO: Fetch onnxruntime-android package or add to docker image in /ort_home

python3 /ort_genai_src/tools/ci_build/github/android/build_aar_package.py \
    --build_dir /build \
    --config $BUILD_CONFIG \
    --android_sdk_path /android_home \
    --android_ndk_path /ndk_home \
    --ort_home /ort_home \
    /ort_genai_src/tools/ci_build/github/android/default_aar_build_settings.json

# Copy the built artifacts to convenient folder for publishing
BASE_PATH=/build/aar_out/${BUILD_CONFIG}/com/microsoft/onnxruntime/genai/${PACKAGE_NAME}/${ORT_VERSION}
mkdir /build/artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}-javadoc.jar  /build/artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}-sources.jar  /build/artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}.aar          /build/artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}.pom          /build/artifacts
