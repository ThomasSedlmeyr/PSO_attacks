#!/bin/bash
#SBATCH -N 1 # dont touch this
#SBATCH --ntasks=100   # number of processor cores (i.e. tasks)
#SBATCH --gpus=0 # number of gpus. Typically 1 except you do a cpu only job then 0
#SBATCH --job-name=syn_gen # you can choose an arbitrary name here (just for logging purposes)
#SBATCH --output=/vol/aimspace/users/sedlmeyr/PSO_attacks/slurm_output/%x-%j.out # the output of your job gets saved to this file (make sure the path exists otherwise the job will fail)
#SBATCH --mem=128g # RAM you need. Dont use too much otherwise other people hate you
#SBATCH -t 2-00:00:00 # maximum time your job runs (7 days is the maximum)

# from here on arbitrary commands

source activate /vol/aimspace/users/sedlmeyr/virtual_envs/pso-attacks-aimspace-env
export PYTHONPATH=$PYTHONPATH:/vol/aimspace/users/sedlmeyr/PSO_attacks
cd /vol/aimspace/users/sedlmeyr/PSO_attacks/src/analysis
python perform_singling_out_attacks.py /vol/aimspace/users/sedlmeyr/PSO_attacks/src/config_syn_gen.yaml

