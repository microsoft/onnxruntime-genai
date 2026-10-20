#!/bin/bash
set -e -x

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"
lsb_packages=()
if [ "$os_major_version" -lt 9 ]; then
  lsb_packages=(redhat-lsb-core)
fi
dnf install -y \
  glibc-langpack-\* glibc-locale-source which "${lsb_packages[@]}" \
  perl-IPC-Cmd openssl-devel wget \
  expat-devel tar unzip zlib-devel make bzip2 bzip2-devel \
  readline-devel
locale