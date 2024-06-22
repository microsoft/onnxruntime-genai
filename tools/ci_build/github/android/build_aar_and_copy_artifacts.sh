#!/bin/bash

# This script will run build_aar_package.py to build Android AAR and copy all the artifacts
# to a given folder for publishing to Maven Central, or building nuget package
# This script is intended to be used in CI build only

set -e -x

python3 tools/ci_build/github/android/build_aar_package.py \
    --build_dir $BUILD_DIR \
    --config $BUILD_CONFIG \
    tools/ci_build/github/android/default_aar_build_settings.json

# Copy the built artifacts to the artifacts staging directory
BASE_PATH=$BUILD_DIR/aar_out/${BUILD_CONFIG}/com/microsoft/onnxruntime/${PACKAGE_NAME}/${GENAI_VERSION}
cp ${BASE_PATH}/${PACKAGE_NAME}-${GENAI_VERSION}-javadoc.jar  $ARTIFACTS_DIR
cp ${BASE_PATH}/${PACKAGE_NAME}-${GENAI_VERSION}-sources.jar  $ARTIFACTS_DIR
cp ${BASE_PATH}/${PACKAGE_NAME}-${GENAI_VERSION}.aar          $ARTIFACTS_DIR
cp ${BASE_PATH}/${PACKAGE_NAME}-${GENAI_VERSION}.pom          $ARTIFACTS_DIR
