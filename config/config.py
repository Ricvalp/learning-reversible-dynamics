import os
import pathlib
from datetime import datetime
from typing import Literal, Type

from absl import logging
from ml_collections import ConfigDict


def get_config(mode: Literal["train", "test_last", "test_specific"] = None):
    if mode is None:
        mode = "train"
        logging.info(f"No mode provided, using '{mode}' as default")

    cfg = ConfigDict()
    cfg.model_name = "model"
    cfg.checkpoint_dir = (
        pathlib.Path("./checkpoints")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        / pathlib.Path(cfg.model_name)
    )

    cfg.overwite_checkpoint = True
    cfg.seed = 42

    cfg.wandb = ConfigDict()
    cfg.wandb.wandb_log = False
    cfg.wandb.project_name = "learning_reversible_dynamics"

    cfg.dataset = ConfigDict()
    cfg.dataset.batch_size = 150
    cfg.dataset.train_lines = 150
    cfg.dataset.num_lines = 200

    cfg.model = ConfigDict()
    cfg.model.num_layers_flow = 5
    cfg.model.num_layers = 2
    cfg.model.num_hidden = 32
    cfg.model.d = 1

    cfg.train = ConfigDict()
    cfg.train.lr = 0.001
    cfg.train.decay_rate = 0.01
    cfg.train.transition_steps = 10000
    cfg.train.num_epochs = 10000
    cfg.train.end_lr = 0.0001

    cfg.experiment_dir = pathlib.Path("visualizations") / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    if mode == "test_last":
        date_and_time = max(os.listdir(pathlib.Path("./checkpoints")))
        cfg.checkpoint_dir = (
            pathlib.Path("./checkpoints") / date_and_time / pathlib.Path(cfg.model_name)
        )

    if mode == "test_specific":
        cfg.date_and_time = "2023-10-10_19-15-40"
        cfg.checkpoint_dir = (
            pathlib.Path("./checkpoints")
            / pathlib.Path(cfg.date_and_time)
            / pathlib.Path(cfg.model_name)
        )

    logging.debug(f"Loaded config: {cfg}")

    return cfg
