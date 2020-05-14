# Very dumb script to create multiple jobs with parameters
import os
import time

params = range(16)
#params = range(1)

for i in params:
    template = """#!/bin/sh
#SBATCH -J unet-%d-f
#SBATCH --gres=gpu:rtx2080ti:1
# #SBATCH --mail-type=ALL --mail-user=carroloi@tcd.ie

cd ~

python2 udocker/udocker run --volume=/home/oisin/dissert:/dissert \\
    --env="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \\
    ml-2 bash -c \\
    "source ~/.bashrc; \\
    cd dissert/Unet/; \\
    python3 train_test.py %d fast"
""" % (i, i)

    with open('test_job.job', 'w') as f:
        f.write(template)

    # Wait for filesystem latency.
    time.sleep(1)

    os.popen('sbatch test_job.job')

    # And again...
    time.sleep(1)
