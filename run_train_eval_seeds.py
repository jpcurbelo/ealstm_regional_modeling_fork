import subprocess
import os
import glob

CAMELS_ROOT = "../../../../../gladwell/hydrology/SUMMA/summa-ml-models/CAMELS_US"
NO_STATIC = True

if NO_STATIC:
    modelname = "lstm"
else:
    modelname = "ealstm"

# List of seeds
# seeds = [333, 444, 555]
seeds = [666, 777, 888]

# Base command for training
train_command = [
    "python", "main.py", "train",
    "--camels_root", CAMELS_ROOT,
    "--cache_data", "True",
    "--no_static", str(NO_STATIC)
]

# Base command for evaluation
evaluate_command_base = [
    "python", "main.py", "evaluate",
    "--camels_root", CAMELS_ROOT 
]

# Function to find the latest run directory
def find_latest_run_dir(seed):
    run_dir_pattern = f"runs/run_{modelname}*_seed{seed}"
    matching_dirs = glob.glob(run_dir_pattern)
    if not matching_dirs:
        return None
    latest_run_dir = max(matching_dirs, key=os.path.getmtime)
    return latest_run_dir

# Loop over the seeds, run training and evaluation
for seed in seeds:
    print(f"Running training with seed {seed}")
    train_cmd = train_command + ["--seed", str(seed)]
    subprocess.run(train_cmd)
    
    # Find the latest run directory for this seed
    latest_run_dir = find_latest_run_dir(seed)
    if latest_run_dir:
        print(f"Evaluating the model for seed {seed} using run directory {latest_run_dir}")
        evaluate_cmd = evaluate_command_base + ["--run_dir", latest_run_dir]
        subprocess.run(evaluate_cmd)
    else:
        print(f"No run directory found for seed {seed}")