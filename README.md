# Learning Reversible Symplectic Dynamics (JAX)

## JAX implementation of [time-reversible neural networks](https://arxiv.org/abs/2204.12323).

### jax_env.yml provides the dependencies.

## To train the model run 

```console
python train.py model_name N num_hidden
```

## where 

1. `model_name`  model name (also used for wandb logging)
2. `N`  number of hidden layers normalizing flow
3. `num_hidden`  number of neurons in the MLP in the last layer of the discriminator

## The following arguments can be parsed 

3. `--batch_size`  batch size: 100
4. `--lr`  initial learning rate: 0.01
5. `--num_epochs`  number of epochs: 2000
6. `--seed`  random seed: 42
7. `--transition_steps`  transition steps in the exponential decay lerning rate (see [documentation](https://optax.readthedocs.io/en/latest/api.html?highlight=exponential#optax.exponential_decay)): 1e06
8. `--decay_rate`  decay learning rate (see [documentation](https://optax.readthedocs.io/en/latest/api.html?highlight=exponential#optax.exponential_decay)): 0.01
9. `--end_lr`  end learning rate (see [documentation](https://optax.readthedocs.io/en/latest/api.html?highlight=exponential#optax.exponential_decay)): 1e-05
10. `--wandb_log`  logging using wandb (set to 'yes' for logging): 'no'
11. `--project_name`  name of the wandb project: unnamed_project

---

## To test the model after training run

```console
python run.py model_name cp
```

## where `model_name` is the name of the trained model, and `cp` is the checkpoint step to load.