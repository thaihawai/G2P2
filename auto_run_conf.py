import os
import subprocess

if __name__ == "__main__":
    log_folder = './logs_w_conf'

    os.makedirs(log_folder, exist_ok=True)
    # learning_rates = [1e-2, 2e-2, 5e-2, 1e-5, 2e-5, 5e-5]
    learning_rates = [5e-5]
    batch_sizes = [32, 64, 128]
    # num_epochs = [25, 50, 100]
    num_epochs = [50]
    # num_samples = [5, 10, 20, 100, 200]
    num_samples = [50]
    confs = [0.85, 0.9, 0.95]
    seeds = [1, 2, 4, 8, 16]
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for num_epoch in num_epochs:
                for num_sample in num_samples:
                    for conf in confs:
                        for seed in seeds:
                            print(f'Running with setting: lr = {lr}, batch_size = {batch_size}, num_epoch = {num_epoch}, seed = {seed}, num_sample = {num_sample}, conf = {conf}')
                            os.makedirs(f'{log_folder}/cora_pseudo_label_{num_epoch}_{lr}_{batch_size}_{num_sample}_{conf}', exist_ok=True)
                            command_run = f'python main_test_pseudo_label_filter_conf.py --ft_epoch {num_epoch} --prompt_lr {lr} --batch_size {batch_size} --seed {seed} --num_sample {num_sample} --conf {conf}'
                            with open(f'{log_folder}/cora_pseudo_label_{num_epoch}_{lr}_{batch_size}_{num_sample}_{conf}/seed_{seed}_stdout.log', 'w') as f_stdout, open(f'{log_folder}/cora_pseudo_label_{num_epoch}_{lr}_{batch_size}_{num_sample}_{conf}/seed_{seed}_stderr.log', 'w') as f_stderr:
                                subprocess.run(command_run, stdout=f_stdout, stderr=f_stderr, shell=True)