#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=35:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ds5749@nyu.edu
#SBATCH -c20
#SBATCH --gres=gpu:a100:1


module purge
singularity exec --nv --overlay /scratch/ds5749/environments/ego.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash  -c "

source /ext3/env.sh;
python train.py --epochs 4 --distinct_count 1000
"
