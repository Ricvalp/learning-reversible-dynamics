import helpers as ai
import torch.utils.data as data
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time

import os
import shutil

import torch.utils.data as data
import numpy as np
import argparse
import wandb

import jax.numpy as jnp
import codecs, json 

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))

u0_path = os.path.join(current_dir, "./pendulum_data/pendulum_dataset_u0.txt")
T_path = os.path.join(current_dir, "./pendulum_data/pendulum_dataset_T.txt")


parser = argparse.ArgumentParser()

parser.add_argument("model_name", type=str,
                    help="model name",
                    )

parser.add_argument("N", type=int,
                    help="number of layers",
                    )

parser.add_argument("num_hidden", type=int,
                    help="hidden dimension"
                    )

### ### ###

parser.add_argument("--batch_size", type=int,
                    help="batch_size: 500",
                    default=500)

parser.add_argument("--num_hidden", type=int,
                    help="number of hidden : 30",
                    default=30)

parser.add_argument("--N", type=int,
                    help="number of hidden layers of the flow: 29",
                    default=29)

parser.add_argument("--lr", type=float,
                    help="Adam initial learning rate: 0.0001",
                    default=0.01)

parser.add_argument("--d", type=float,
                    help="half the dimension of the system",
                    default=1)

parser.add_argument("--decay_rate", type=float,
                    help="optax exponential decay rate: 0.01",
                    default=0.01)

parser.add_argument("--transition_steps", type=float,
                    help="optax exponential decay transition steps: 1e06",
                    default=1e06)

parser.add_argument("--seed", type=int,
                    help="random seed: 42",
                    default=42)

parser.add_argument("--num_epochs", type=int,
                    help="number of epochs: 200",
                    default=2000)

parser.add_argument("--wandb_log", type=str,
                    help="if 'yes' logs with wandb: 'no'",
                    default="no")
        
parser.add_argument("--project_name", type=str,
                    help="wandb project model name",
                    default="unnamed_project")

parser.add_argument("--train_lines", type=int,
                    help="number of points to use ad train dataset: 100",
                    default=100)

parser.add_argument("--num_lines", type=int,
                    help="total rnumber of points in the dataset: 200",
                    default=200)

parser.add_argument("--overwite_checkpoint", type=bool,
                    help="if True overwites existing checkpoints with same model name",
                    default=True)

parser.add_argument("--end_lr", type=float,
                    help="optax exponential decay end learning rate: 1e-05",
                    default=1e-05)


args = parser.parse_args()

with open('commandline_args_' + args.model_name + '.txt', 'w') as f:
   json.dump(args.__dict__, f, indent = 2)

# path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./cp_" + args.model_name

# overwrite existing checkpoints
if os.path.exists(CHECKPOINT_PATH) and args.overwite_checkpoint==True:
    shutil.rmtree(CHECKPOINT_PATH)

N = args.N
batch_size = args.batch_size
num_hidden = args.num_hidden
lr = args.lr
num_epochs = args.num_epochs
seed = args.seed
model_name = args.model_name
wandb_log = args.wandb_log
project_name = args.project_name
train_lines = args.train_lines
num_lines = args.num_lines
d = args.d
decay_rate = args.decay_rate
end_lr = args.end_lr

if wandb_log=="yes":
        wandb.init(project=project_name, name=model_name, config = {
                "N": N,
                "batch_size": batch_size,
                "lr": lr,
                "decay_rate": decay_rate,
                "num_epochs": num_epochs,
                "seed": seed,
                "train_lines": train_lines,
                "num_lines": num_lines
        })

### ### ### ### ### ### ### ### ### ###
###  ###  ###   DATASET   ###  ###  ###
### ### ### ### ### ### ### ### ### ###

### train and evaluation dataset
train_dataset = ai.Dataset(train_lines=train_lines, num_lines=num_lines, u0_path=u0_path, T_path=T_path, train=True)
val_dataset = ai.Dataset(train_lines=train_lines, num_lines=num_lines, u0_path=u0_path, T_path=T_path, train=False)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ai.numpy_collate)
val_loader = data.DataLoader(val_dataset, batch_size=num_lines-batch_size, shuffle=True, collate_fn=ai.numpy_collate)

data_inputs, data_outputs = next(iter(train_loader))

plt.figure(figsize=(8,8))
plt.scatter([x[0] for x in data_inputs], [x[1] for x in data_inputs], c='r', label = "x")
plt.scatter([x[0] for x in data_outputs], [x[1] for x in data_outputs], c='b', label = "T(x)")
plt.show()

### ### ### ### ###  ### ### ### ### ###
### ### ### TRAIN AI-SAMPLER ### ### ###
### ### ### ### ###  ### ### ### ### ###

trainer = ai.TrainerModule(model_name=model_name,
                                N=N,
                                num_hidden=num_hidden,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                CHECKPOINT_PATH=CHECKPOINT_PATH,
                                d=d,
                                lr=lr,
                                decay_rate=decay_rate,
                                wandb_log=wandb_log,
                                end_lr = end_lr,
                                seed=seed)

trainer.train_model(num_epochs=num_epochs)