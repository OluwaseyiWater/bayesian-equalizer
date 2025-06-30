import csv
import wandb

def init_wandb(project: str, run_name: str, config: dict):
    wandb.init(project=project, name=run_name, config=config)

def log_wandb(metrics: dict, step: int = None):
    wandb.log(metrics, step=step)

def save_csv(metrics: list, filepath: str, header: list):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(metrics)
