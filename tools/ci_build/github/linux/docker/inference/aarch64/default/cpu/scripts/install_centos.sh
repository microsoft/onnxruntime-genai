#!/bin/bash
set -e -x

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"
dnf install -y \
  python3-devel python3-pip python3-wheel\
  glibc-langpack-\* glibc-locale-source which redhat-lsb-core \
  expat-devel tar unzip zlib-devel make bzip2 bzip2-devel \
  java-11-openjdk-devel graphviz \
  git
locale
