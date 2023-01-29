import helpers as ai
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
import jax
import argparse
import json
import os
import jax.numpy as jnp

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

parser = argparse.ArgumentParser()

parser.add_argument("model_name",
                    help="model name"
                    )

parser.add_argument("cp",
                    nargs='?',
                    help="step checkpoint ai-sampler"
                    )

args = parser.parse_args()
arg_dict = vars(args)

current_dir = os.path.dirname(os.path.abspath(__file__))
commands_path = os.path.join(current_dir, './commandline_args_' + args.model_name + '.txt')
with open(commands_path, 'r') as f:
   json_dict = json.load(f)

arg_dict.update(json_dict)
args.__dict__ = arg_dict
print("Arguments: ", args)

model_name = args.model_name

CHECKPOINT_PATH = os.path.join(current_dir, './cp_' + model_name)

N = args.N
batch_size = args.batch_size
num_hidden = args.num_hidden
seed = args.seed
model_name = args.model_name
wandb_log = args.wandb_log
project_name = args.project_name
train_lines = args.train_lines
num_lines = args.num_lines
d = args.d
cp = args.cp

### ### ### LOAD MODEL ### ### ###

### train and avaluation dataset
u0_path = os.path.join(current_dir, "./pendulum_data/pendulum_dataset_u0.txt")
T_path = os.path.join(current_dir, "./pendulum_data/pendulum_dataset_T.txt")

train_dataset = ai.Dataset(train_lines=train_lines, num_lines=num_lines, u0_path=u0_path, T_path=T_path, train=True)
val_dataset = ai.Dataset(train_lines=train_lines, num_lines=num_lines, u0_path=u0_path, T_path=T_path, train=False)
train_loader = data.DataLoader(train_dataset, batch_size=train_lines, shuffle=True, collate_fn=ai.numpy_collate)
val_loader = data.DataLoader(val_dataset, batch_size=num_lines-train_lines, shuffle=True, collate_fn=ai.numpy_collate)

trainer = ai.TrainerModule(model_name=model_name,
                                N=N,
                                num_hidden=num_hidden,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                CHECKPOINT_PATH=CHECKPOINT_PATH,
                                d=d,
                                wandb_log=wandb_log,
                                seed=seed)

flow = trainer.load_model(cp)

@jax.jit
def run_flow(x):
    return flow(x)


### ### ### RUN MODEL ### ### ###

predictions = [jnp.array([[0., i*0.05] for i in range(10)] + [[i*0.06, 0.] for i in range(10)])]
# predictions = [np.random.uniform(low=-0.3, high=0.3, size=(s, 2)) + np.array([[0., 0.] for _ in range(s)])]

for _ in range(300):
    predictions.append(run_flow(predictions[-1]))

plt.figure(figsize=(8,8))
for pred in predictions:
    plt.scatter([x[0] for x in pred], [x[1] for x in pred], c='r', s=0.1)
plt.scatter([x[0] for x in predictions[0]], [x[1] for x in predictions[0]], c='b')
# plt.savefig("trajectories")
plt.show()