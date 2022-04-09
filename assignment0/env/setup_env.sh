#!/usr/bin/env bash

# This script file is copied from https://github.com/CSAILVision/gandissect
# Bash script to set up an anaconda python-based deep learning environment
# that has support for pytorch and some other things.

# This should not require root.  However, it does copy and build a lot of
# binaries into your ~/.conda directory.  If you do not want to store
# these in your homedir disk, then ~/.conda can be a symlink somewhere else.
# (At MIT CSAIL, you should symlink ~/.conda to a directory on NFS or local
# disk instead of leaving it on AFS, or else you will exhaust your quota.)

# Start from parent directory of script
#cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Default RECIPE 'environment' can be overridden by 'RECIPE=foo setup.sh'
RECIPE=${RECIPE:-environment.yml}
# Default ENV_NAME 'hab' can be overridden by 'ENV_NAME=foo setup.sh'
ENV_NAME="${ENV_NAME:-deep-learning-19}"
ANACONDA_ENV_PATH="${HOME}/anaconda3/envs"
echo "Creating conda environment ${ENV_NAME}"

if [[ ! $(type -P conda) ]]
then
    echo "conda not in PATH"
    echo "read: https://conda.io/docs/user-guide/install/index.html"
    exit 1
fi

# Uninstall existing environment
if [[ -e "${ANACONDA_ENV_PATH}/${ENV_NAME}" ]]
then
 if [[ "$1" == "rebuild" ]]
 then
  echo "Deleting and rebuilding existing environment ${ENV_NAME}."
 else
  echo "Conda environment ${ENV_NAME} already exists."
  echo "Use '$0 rebuild' if you want to delete it and rebuild."
  exit 1
 fi
fi

rm -rf ${ANACONDA_ENV_PATH}/${ENV_NAME}

# Build new environment: torch and torch vision from source
conda env create -n=${ENV_NAME} -f ${RECIPE}

# Set up CUDA_HOME to set itself up correctly on every source activate
# https://stackoverflow.com/questions/31598963
# CUDA_HOME is needed for building packages with the right CUDA tool versions.
# https://github.com/rusty1s/pytorch_scatter/issues/19#issuecomment-449735614
mkdir -p ${ANACONDA_ENV_PATH}/${ENV_NAME}/etc/conda/activate.d
echo "export CUDA_HOME=/usr/local/cuda-9.0" > \
    ${ANACONDA_ENV_PATH}/${ENV_NAME}/etc/conda/activate.d/CUDA_HOME.sh
