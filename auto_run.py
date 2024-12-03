import os
import subprocess

if __name__ == "__main__":
    log_folder = './logs'

    os.makedirs(log_folder, exist_ok=True)
    # learning_rates = [1e-2, 2e-2, 5e-2, 1e-5, 2e-5, 5e-5]
    learning_rates = [1e-5, 2e-5, 5e-5]
    batch_sizes = [32, 128]
    # num_epochs = [25, 50, 100]
    num_epochs = [50]
    num_samples = [5, 10, 20, 100, 200]
    seeds = [1, 2, 4, 8, 16]
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for num_epoch in num_epochs:
                for num_sample in num_samples:
                    # if batch size is 128 and num sample is less than 32 then 32 already run them
                    if batch_size == 128 and num_sample < 32:
                        continue
                    for seed in seeds:
                        print(f'Running with setting: lr = {lr}, batch_size = {batch_size}, num_epoch = {num_epoch}, seed = {seed}, num_sample = {num_sample}')
                        os.makedirs(f'{log_folder}/cora_pseudo_label_{num_epoch}_{lr}_{batch_size}_{num_sample}', exist_ok=True)
                        command_run = f'python main_test_pseudo_label.py --ft_epoch {num_epoch} --prompt_lr {lr} --batch_size {batch_size} --seed {seed} --num_sample {num_sample}'
                        with open(f'{log_folder}/cora_pseudo_label_{num_epoch}_{lr}_{batch_size}_{num_sample}/seed_{seed}_stdout.log', 'w') as f_stdout, open(f'{log_folder}/cora_pseudo_label_{num_epoch}_{lr}_{batch_size}_{num_sample}/seed_{seed}_stderr.log', 'w') as f_stderr:
                            subprocess.run(command_run, stdout=f_stdout, stderr=f_stderr, shell=True)