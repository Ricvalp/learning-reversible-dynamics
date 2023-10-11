import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state
from tqdm import tqdm

from models.HenonFlow import create_henon_flow


def calculate_loss(state, params, batch):
    data_input, data_output = batch

    prediction = state.apply_fn(params, data_input)

    loss = jnp.mean(jnp.square(data_output - prediction))
    return loss


class TrainerModule:
    def __init__(
        self,
        model_name,
        num_layers_flow,
        num_layers,
        num_hidden,
        train_loader,
        val_loader,
        checkpoint_dir,
        d,
        decay_rate=0.01,
        transition_steps=1e06,
        lr=1e-4,
        end_lr=1e-05,
        wandb_log="no",
        seed=42,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_layers_flow = num_layers_flow
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.lr = lr
        self.seed = seed
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.d = d
        self.decay_rate = decay_rate
        self.wandb_log = wandb_log
        self.transition_steps = transition_steps
        self.end_lr = end_lr

        # Create model
        self.model = create_henon_flow(
            num_layers_flow=self.num_layers_flow,
            num_layers=self.num_layers,
            num_hidden=self.num_hidden,
            d=self.d,
        )

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
            grad = jax.value_and_grad(calculate_loss, argnums=1, has_aux=False)

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
        # init_learning_rate = self.lr # initial learning rate for Adam
        # exponential_decay_scheduler = optax.exponential_decay(init_value=init_learning_rate, transition_steps=self.transition_steps,
        #                                                     decay_rate=self.decay_rate, transition_begin=50,  end_value=self.end_lr,
        #                                                     staircase=False)
        # optimizer = optax.adam(learning_rate=exponential_decay_scheduler)

        # SIMPLE ADAM
        optimizer = optax.adam(learning_rate=self.lr)

        # GRADIENT CLIPPING + ADAM
        # optimizer = optax.chain(
        #     optax.clip(self.clip_at),
        #     optax.adam(learning_rate=self.lr))

        # Initialize training state
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=model_params, tx=optimizer
        )

    def train_model(self, num_epochs=50):
        t = tqdm(range(1, num_epochs + 1), unit="step")
        for epoch_idx in t:
            loss = self.train_epoch()
            t.set_description(f"loss: {loss:.6f}")

            # Saving checkpoint every 10 epochs if validation loss improves
            if epoch_idx % 10 == 0:
                val = self.eval_step(self.model_state, self.val_batch)
                if self.wandb_log:
                    wandb.log(
                        {
                            "val_loss": val,
                        }
                    )
                if val < self.current_val_loss:
                    self.current_val_loss = val
                    self.save_checkpoint(step=epoch_idx)

    def train_epoch(self):
        plot_count = 0

        for batch in self.train_loader:
            self.model_state, loss = self.train_step(self.model_state, batch)

            if self.wandb_log:
                wandb.log(
                    {
                        "train_loss": loss,
                    }
                )

            plot_count += 1

        return loss

    def save_checkpoint(self, step):
        checkpoint = {
            "model_params": self.model_state.params,
            "step": step,
        }
        with open(Path(self.checkpoint_dir) / Path("checkpoint"), "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self):
        with open(self.checkpoint_dir / Path("checkpoint"), "rb") as f:
            checkpoint = pickle.load(f)

        self.model_state = self.model_state.replace(params=checkpoint["model_params"])

    def forward(self, x):
        return self.model_state.apply_fn(self.model_state.params, x)
