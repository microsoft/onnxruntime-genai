# Use a base image that supports multiple architectures
FROM ubuntu:24.04

# Ensure that all interactions for apt are non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages for building gRPC and C++ applications
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    cmake \
    build-essential \
    ninja-build \
    automake \
    autoconf \
    libtool \
    pkg-config \
    unzip \
    wget \
    vim \
    nano \
    curl \
    openjdk-11-jdk \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    python3-requests \
    && rm -rf /var/lib/apt/lists/*

# Install Android SDK
ENV ANDROID_HOME=/opt/android-sdk

RUN mkdir -p $ANDROID_HOME && \
    wget -O cmd.zip https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip && \
    unzip cmd.zip -d $ANDROID_HOME && rm cmd.zip && \
    yes | $ANDROID_HOME/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_HOME --licenses && \
    $ANDROID_HOME/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_HOME 'platforms;android-27' && \
    $ANDROID_HOME/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_HOME 'build-tools;30.0.2' && \
    $ANDROID_HOME/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_HOME 'ndk;27.2.12479018' 
