#!/bin/bash
set -e -x

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"
dnf install -y \
  python39-devel python39-pip python39-wheel\
  glibc-langpack-\* glibc-locale-source which redhat-lsb-core \
  perl-IPC-Cmd openssl-devel wget \
  expat-devel tar unzip zlib-devel make bzip2 bzip2-devel
locale