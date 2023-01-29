from sre_constants import RANGE
from tkinter import N
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import jax
import jax.numpy as jnp
from flax import linen as nn
import torch.utils.data as data
from flax.training import checkpoints
import optax
from flax.training import checkpoints, train_state
import wandb

class CouplingLayer(nn.Module):

    network_s : nn.Module  # NN to use in the flow for predicting mu and sigma
    network_t : nn.Module  # NN to use in the flow for predicting mu and sigma

    mask : np.ndarray  # Binary mask where 0 denotes that the element should be transformed, and 1 not.
    c_in : int  # Number of input channels

    def setup(self):
        
        self.scaling_factor = self.param('scaling_factor',
                                         nn.initializers.zeros,
                                         (self.c_in,))

    def __call__(self, z, reverse=False, orig_img=None): # ...,ldj, ...

        # Apply network to masked input
        z_in = z*self.mask
        s = self.network_s(z_in)
        t = self.network_t(z_in)

        # Stabilize scaling output
        s_fac = jnp.exp(self.scaling_factor)
        s = nn.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:

            z = (z + t) * jnp.exp(s)

        else:

            z = (z * jnp.exp(-s)) - t

        return z
class VP_CouplingLayer(nn.Module):

    network_t : nn.Module

    mask : np.ndarray
    c_in : int

    def setup(self):

        pass
        

    def __call__(self, z, reverse=False, orig_img=None):

        z_in = z*self.mask
        t = self.network_t(z_in)

        t = t * (1 - self.mask)

        if not reverse:
            
            z = z + t
        else:
            z = z - t
            
        return z
class SimpleMLP(nn.Module):

    num_hidden : int
    num_outputs : int

    def setup(self):

        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_hidden)
        self.linear3 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):

        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        x = nn.tanh(x)
        x = self.linear3(x)

        return x
class FlowModel(nn.Module):

    d : int
    flows : Sequence[nn.Module]


    def setup(self):
        
        self.R = jax.numpy.array([1. for i in range(self.d)] + [-1. for i in range(self.d)])


    def __call__(self, x):

        z = x

        for flow in self.flows:
            z = flow(z, reverse=False)

        z = z*self.R

        for flow in reversed(self.flows):
            z = flow(z, reverse=True)

        z = z*self.R

        return z
class Dataset(data.Dataset):

    def __init__(self, train_lines, num_lines, u0_path, T_path, train=True):
        super().__init__()
        self.train_lines = train_lines
        self.num_lines = num_lines
        self.u0_path = u0_path
        self.T_path = T_path
        self.train = train
        self.generate_data()

    def generate_data(self):

        data_input = []
        data_output = []

        if self.train==True:

            with open(self.u0_path) as f:
                for _ in range(self.train_lines):
                    data_input.append([float(f.readline()), float(f.readline())])

            with open(self.T_path) as f:
                for _ in range(self.train_lines):
                    data_output.append([float(f.readline()), float(f.readline())])

            self.data = np.array(data_input)
            self.label = np.array(data_output)
        
        else:

            with open(self.u0_path) as f:
                for _ in range(self.train_lines):
                    float(f.readline())
                    float(f.readline())
                for _ in range(self.num_lines-self.train_lines):
                    data_input.append([float(f.readline()), float(f.readline())])

            with open(self.T_path) as f:
                for _ in range(self.train_lines):
                    float(f.readline())
                    float(f.readline())
                for _ in range(self.num_lines-self.train_lines):
                    data_output.append([float(f.readline()), float(f.readline())])

            self.data = np.array(data_input)
            self.label = np.array(data_output)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
class HenonLayer(nn.Module):

    V : nn.Module

    def setup(self):
        self.eta = self.param('eta', nn.initializers.zeros, (1,2))

    def __call__(self, z, reverse=False):

        if not reverse:

            X = jnp.matmul(z, jnp.array([[0., 1.], [0., 0.]]))
            Y = jnp.matmul(z, jnp.array([[0., 0.], [1., 0.]]))
            ETA = jnp.matmul(self.eta, jnp.array([[1., 0.], [0., 0.]]))
            V = jnp.matmul(self.V(Y), jnp.array([[0., 0.], [0., 1.]]))

            return -X + Y + ETA + V

        else:

            X = jnp.matmul(z, jnp.array([[0., 1.], [0., 0.]]))
            Y = jnp.matmul(z, jnp.array([[0., 0.], [1., 0.]]))
            ETA = jnp.matmul(self.eta, jnp.array([[0., 0.], [0., 1.]]))

            Xbar = jnp.matmul(X-ETA, jnp.array([[0., 0.], [1., 0.]]))
            V = jnp.matmul(self.V(Xbar), jnp.array([[0., 0.], [1., 0.]]))

            return X - Y - ETA + V



### ### ### models ### ### ###

def generate_mask(i, d):

    if i%2==0:
        return jax.numpy.array([0. for i in range(d)] + [1. for i in range(d)])
    else:
        return jax.numpy.array([1. for i in range(d)] + [0. for i in range(d)])
def create_flow(N, num_hidden, d):

    flow_layers = []

    flow_layers += [CouplingLayer(network_s=SimpleMLP(num_hidden=num_hidden, num_outputs=d), network_t=SimpleMLP(num_hidden=num_hidden, num_outputs=d), mask=generate_mask(i, d), c_in=1) for i in range(N)]

    flow_model = FlowModel(d, flow_layers)

    return flow_model
def create_VP_flow(N, num_hidden, d):

    flow_layers = []

    flow_layers += [VP_CouplingLayer(network_t=SimpleMLP(num_hidden=num_hidden, num_outputs=d), mask=generate_mask(i, d), c_in=1) for i in range(N)]

    flow_model = FlowModel(d, flow_layers)

    return flow_model
def create_henon_flow(N, num_hidden, d):

    flow_layers = []

    flow_layers += [HenonLayer(SimpleMLP(num_hidden=num_hidden, num_outputs=2)) for _ in range(N)]

    flow_model = FlowModel(d, flow_layers)

    return flow_model
def numpy_collate(batch):

    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
def create_henon_layer(num_hidden):

    V = SimpleMLP(num_hidden=num_hidden, num_outputs=2)
    
    hanon_layer = HenonLayer(V)

    return hanon_layer



### ### ### training ### ### ###

def calculate_loss(state, params, batch):

    data_input, data_output = batch

    prediction = state.apply_fn(params, data_input)
    
    loss =  jnp.mean(jnp.square(data_output - prediction))
    return loss
@jax.jit
def train_step(state, batch):


    grad = jax.value_and_grad(calculate_loss,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=False  # Function has additional outputs, e.g., accuracy
                                )

    loss, grads = grad(state, state.params, batch)
    state = state.apply_gradients(grads=grads)

    return state, loss
@jax.jit
def eval_step(state, batch):

    eval_loss = calculate_loss(state, state.params, batch)
    return eval_loss
def train_model(state, train_data_loader, eval_data_loader, check_every=20, num_epochs=100):
    
    # step = jax.jit(train_step, static_argnums=(3,))
    loss = []
    eval_loss = []
    saved = 0

    checkpoint_eval = 100

    for epoch in range(num_epochs):

        for batch in train_data_loader:
            state, l = train_step(state, batch)
        
        loss.append(l)

        eval_batch = next(iter(eval_data_loader))
        eval_loss.append(eval_step(state, eval_batch))

        if (epoch%check_every==0 and eval_loss[-1]<checkpoint_eval) :

            checkpoints.save_checkpoint(ckpt_dir='my_checkpoints/',  # Folder to save checkpoint in
                            target=state,  # What to save. To only save parameters, use model_state.params
                            step=epoch,  # Training step or other metric to save best model on
                            prefix='my_model',  # Checkpoint file name prefix
                            overwrite=True   # Overwrite existing checkpoint files
                           )
            
            checkpoint_eval=eval_loss[-1]
            saved = saved + 1
        
        print(epoch, "loss:  ", l, "eval: ", eval_loss[-1], "saved: ", saved)
    
    print("checkpoint evaluation loss: ", checkpoint_eval)

    return state, loss, eval_loss



### ### ### trainer ### ### ###

class TrainerModule:

    def __init__(self, 
                    model_name,
                    N,
                    num_hidden,
                    train_loader,
                    val_loader,
                    CHECKPOINT_PATH,
                    d,
                    decay_rate = 0.01,
                    transition_steps = 1e06,
                    lr=1e-4,
                    wandb_log = "no",
                    seed=42):

        super().__init__()

        self.model_name = model_name
        self.N = N
        self.num_hidden = num_hidden
        self.lr = lr
        self.seed = seed
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.CHECKPOINT_PATH=CHECKPOINT_PATH
        self.d = d
        self.decay_rate = decay_rate
        self.wandb_log = wandb_log
        self.transition_steps = transition_steps

        # Create model
        self.model = create_henon_flow(N=self.N, num_hidden=self.num_hidden, d=self.d)

        # Batch from evaluation for shape initialization
        self.val_batch = next(iter(val_loader))
        
        # Create jitted training and eval functions
        self.create_functions()
        
        # Initialize model
        self.init_model()

        # Current validation loss for checkpoints
        self.current_val_loss = self.eval_step(self.model_state, self.val_batch)

    def create_functions(self):

        # Training function
        def train_step(model_state, batch):
    
            grad = jax.value_and_grad(calculate_loss, 
                                        argnums=1,
                                        has_aux=False
                                        )


            loss, grads = grad(model_state, model_state.params, batch)
            model_state = model_state.apply_gradients(grads=grads)

            # print(grads)

            return model_state, loss

        self.train_step = jax.jit(train_step)

        # Eval function
        def eval_step(model_state, batch):
            
            loss = calculate_loss(model_state, model_state.params, batch)

            return loss

        self.eval_step = jax.jit(eval_step)

    def init_model(self):

        # Initialize model

        rng = jax.random.PRNGKey(self.seed)
        rng, inp_model = jax.random.split(rng, 2)

        model_params = self.model.init(inp_model, self.val_batch[0])  # ['params']

        # Optimizer

        # EXPONENTIAL DECAY LEARNING RATE
        init_learning_rate = self.lr # initial learning rate for Adam
        exponential_decay_scheduler = optax.exponential_decay(init_value=init_learning_rate, transition_steps=self.transition_steps,
                                                            decay_rate=self.decay_rate, transition_begin=50,
                                                            staircase=False)
        optimizer = optax.adam(learning_rate=exponential_decay_scheduler)

        # SIMPLE ADAM
        # optimizer = optax.adam(learning_rate=self.lr)

        # GRADIENT CLIPPING + ADAM
        # optimizer = optax.chain(
        #     optax.clip(self.clip_at),
        #     optax.adam(learning_rate=self.lr))

        # Initialize training state
        self.model_state = train_state.TrainState.create(apply_fn=self.model.apply, params=model_params, tx=optimizer)

    def train_model(self, num_epochs=50):

        for epoch_idx in range(1, num_epochs+1):

            loss = self.train_epoch()

            print("epoch: ", epoch_idx, ", loss : ", loss)

            # Saving checkpoint if validation loss improves
            val = self.eval_step(self.model_state, self.val_batch)
            if val<self.current_val_loss:
                self.current_val_loss = val
                self.save_model(step=epoch_idx)

    def train_epoch(self):

        plot_count = 0

        for batch in self.train_loader:

            self.model_state, loss = self.train_step(self.model_state, batch)

            if self.wandb_log=="yes":

                wandb.log({
                    "loss": loss,
                })

            plot_count+=1

        return loss

    def save_model(self, step):

        # Save current model at certain training iteration

        checkpoints.save_checkpoint(ckpt_dir=self.CHECKPOINT_PATH + '/checkpoints/',  # Folder to save checkpoint in
                            target=self.model_state,  # What to save. To only save parameters, use model_state.params
                            step=step,  # Training step or other metric to save best model on
                            prefix='model',  # Checkpoint file name prefix
                            keep=10,
                            overwrite=True   # Overwrite existing checkpoint files
                           )

    def load_model(self, step):

        model_state = checkpoints.restore_checkpoint(
                                        ckpt_dir=self.CHECKPOINT_PATH +'/checkpoints/',   # Folder with the checkpoints
                                        target=self.model_state,   # (optional) matching object to rebuild state in
                                        prefix='model',
                                        step=step
                                        )
        
        trained_model = self.model.bind(model_state.params)
     
        return trained_model
    
    def get_model(self):

        return self.model

"""
### ### ### plot ### ### ###
def plot_lr_scheduler(epochs, init_learning_rate, decay_rate, transition_steps, log=False):


    decayed_values = [init_learning_rate*decay_rate**(i/transition_steps) for i in range(epochs)]
    plt.figure(figsize=(10,5))
    plt.plot(decayed_values)
    plt.title("Learning rate scheduler")
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    if log==True:
        plt.yscale('log')
    plt.show()
"""