#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N run_learner_speaker_diff_lexicons_p_prior_egoc_sp_literal_P1_a1_lr_perspective-taking_a1_lr_sp_hyp_literal_30_C_2_R
#$ -cwd
#$ -pe sharedmem 1
#$ -l h_vmem=1.5G
#$ -l h_rt=02:00:00
#$ -t 1-343
#  These options are:
#  job name: -N
#  use the current working output_pickle_file_directory: -cwd
#  1 compute cores: -pe sharedmem
#  memory limit of in Gbyte: -l h_vmem
#  runtime limit of in hours: -l h_rt
#  Tell SGE that this is an array job, with "tasks" numbered: -t


# Initialise the environment modules
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# Load Python
module load python/2.7.10

# Run the program
python run_learner_speaker_diff_lexicons_p_prior_egoc_sp_literal_P1_a1_lr_perspective-taking_a1_lr_sp_hyp_literal_30_C_2_R.py $SGE_TASK_ID

