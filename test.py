from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch.utils.data as data
import wandb
from absl import app, logging
from ml_collections import config_flags

from config import load_cfgs
from dataset.dataset import Dataset
from dataset.utils import numpy_collate
from trainer import TrainerModule

_CFG_FILE = config_flags.DEFINE_config_file("config", default="config/config.py:test")


def main(_):
    cfg = load_cfgs(_CFG_FILE)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    vis_folder = Path(cfg.experiment_dir) / Path(f"{cfg.model_name}")
    vis_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Storing results in {checkpoint_dir}.")
    logging.info(f"Loaded config: {cfg}")

    u0_path = Path("./pendulum_data/pendulum_dataset_u0.txt")
    T_path = Path("./pendulum_data/pendulum_dataset_T.txt")

    if cfg.wandb.wandb_log:
        wandb.init(project=cfg.wandb.project_name, name=cfg.model_name, config=cfg)

    train_dataset = Dataset(
        train_lines=cfg.dataset.train_lines,
        num_lines=cfg.dataset.num_lines,
        u0_path=u0_path,
        T_path=T_path,
        train=True,
    )
    val_dataset = Dataset(
        train_lines=cfg.dataset.train_lines,
        num_lines=cfg.dataset.num_lines,
        u0_path=u0_path,
        T_path=T_path,
        train=False,
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, collate_fn=numpy_collate
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=cfg.dataset.num_lines - cfg.dataset.batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
    )

    trainer = TrainerModule(
        model_name=cfg.model_name,
        num_layers_flow=cfg.model.num_layers_flow,
        num_layers=cfg.model.num_layers,
        num_hidden=cfg.model.num_hidden,
        d=cfg.model.d,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        lr=cfg.train.lr,
        decay_rate=cfg.train.decay_rate,
        wandb_log=cfg.wandb.wandb_log,
        end_lr=cfg.train.end_lr,
        seed=cfg.seed,
    )

    trainer.load_checkpoint()

    @jax.jit
    def run_flow(x):
        return trainer.forward(x)

    predictions = [
        jnp.array([[0.0, i * 0.05] for i in range(10)] + [[i * 0.06, 0.0] for i in range(10)])
    ]

    for _ in range(30):
        predictions.append(run_flow(predictions[-1]))

    plt.figure(figsize=(8, 8))
    for pred in predictions:
        plt.scatter([x[0] for x in pred], [x[1] for x in pred], c="r", s=0.1)
    plt.scatter([x[0] for x in predictions[0]], [x[1] for x in predictions[0]], c="b")
    plt.savefig(vis_folder / Path("predictions.png"))
    plt.show()


if __name__ == "__main__":
    app.run(main)
