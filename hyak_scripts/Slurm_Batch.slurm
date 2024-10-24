#!/bin/bash

#SBATCH --job-name=Fruitfly     ### Job Name
#SBATCH --partition=gpu-l40s    ### Parition to run job: gpu-l40s, ckpt-g2
#SBATCH --account=portia      ### Account to run job
#SBATCH --time=2-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --cpus-per-task=16    ### Number of CPU's per task
#SBATCH --gpus=8              ### General REServation of gpu:number of gpus
#SBATCH --mem=128G            ### Memory in GB
# #SBATCH --array=10 ###0-63 ###  ### Array index
#SBATCH --verbose  
#SBATCH --exclude=g3090,g3107,g3097 ### Exclude nodes
#SBATCH -o ./OutFiles/slurm-%A_%a.out  ### File in which to store job stdout

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=EMAIL@EMAIL.COM  ### E-mail to which notifications will be sent

###### Load necessary modules: This will depend on what you are working on ######
module load cuda/12.4.1
set -x
source ~/.bashrc
nvidia-smi
###### Load conda environment to run code ######
conda activate stac-mjx-env
python -u main_requeue.py paths=hyak train.note=hyak_ckpt version=ckpt train=train_fly dataset=fly train.num_envs=8192 num_gpus=8 run_id=$SLURM_JOB_ID 

###### Useful SLurm commands ######
### check job status: squeue -u $USER
### check hyak allocation: hyakalloc
### info on resources on a partition: sinfo -p ckpt-g2 -O nodehost,cpusstate,freemem,gres,gresused -S nodehost | grep -v null 
#### cancel all jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel
### python scripts/slurm-run_bbrunton.py paths=hyak train=train_fly_run dataset=fly_run train.note=hyak train.num_envs=1024 gpu=0
