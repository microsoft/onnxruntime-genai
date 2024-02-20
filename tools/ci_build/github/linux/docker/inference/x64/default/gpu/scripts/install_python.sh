#!/bin/bash
set -e -x
os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)
echo "installing Python for CentOS version : $os_major_version"

dnf install -y \
  python3.8-devel python3.8-pip python3.8-wheel\
  python3.9-devel python3.9-pip python3.9-wheel\
  python3.10-devel python3.10-pip python3.10-wheel\
  python3.11-devel python3.11-pip python3.11-wheel\
  python3.12-devel python3.12-pip python3.12-wheel
