#!/bin/bash
#SBATCH --job-name="leafAngle"
#SBATCH --output="leafAngle.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --export=ALL
#SBATCH -t 01:30:00

#This job runs with 1 nodes, 20 cores per node for a total of 20 cores.
#ibrun in verbose mode will give binding detail

PROJECT_DIR=/oasis/scratch/comet/slb197/temp_project
CODE_DIR=/home/slb197/pyWorkSpace/leafAngleDistribution
module purge
module load singularity
singularity exec $PROJECT_DIR/terra-batch.img $CODE_DIR/comet_batch_process.sh
