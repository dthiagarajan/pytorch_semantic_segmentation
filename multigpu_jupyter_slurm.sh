#!/bin/bash
#SBATCH -J segmentation_dt372    # Job name
#SBATCH -o segmentation_dt372.%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e segmentation_dt372.%j    # Name of stderr output file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16GB

# Load necessary modules (uncomment when needed)
# module use ~fw245/modulefiles
# module rm cuda cudnn
# module add cuda/9.0 cudnn/v7.0-cuda-9.0

#Go to the folder you wanna run jupyter in
cd .
export XDG_RUNTIME_DIR="./"

#Pick a random or predefined port
port=8080

#Forward the picked port to the prince on the same port. Here log-x is set to be the prince login node.
/usr/bin/ssh -N -f -R $port:localhost:$port graphite

#Start the notebook
jupyter notebook --no-browser --port $port
