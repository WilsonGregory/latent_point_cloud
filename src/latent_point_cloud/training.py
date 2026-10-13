import enum
import json
import pathlib
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb
from jaxtyping import Array, Float
from torch.utils.data import DataLoader

# ~~~~~~~~~~~~~~~~~~~~~~ Saving/Loading ~~~~~~~~~~~~~~~~~~~~~~


def save(filename: str | pathlib.Path, model: eqx.Module, further_args: dict | None = None) -> None:
    """
    New version of save, allows you to save any serializable args.

    args:
        filename: the file to save the model to
        model: the model to save
        further_args: more values to save, as a dictionary
    """
    further_args = {} if further_args is None else further_args
    with open(filename, "wb") as f:
        further_args_str = json.dumps(further_args)
        f.write((further_args_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename: str | pathlib.Path, model: eqx.Module) -> tuple[eqx.Module, dict]:
    """
    Load an equinox model.

    args:
        filename: the file to load the model from
        model: the type of model we are loading, the parameter values will be set to the loaded ones

    returns:
        the loaded model
    """
    with open(filename, "rb") as f:
        further_args = json.loads(f.readline().decode())
        return eqx.tree_deserialise_leaves(f, model), further_args


# ~~~~~~~~~~~~~~~~~~~~~~ Training Functions ~~~~~~~~~~~~~~~~~~~~~~


class Loss(ABC):
    # a generic loss functor

    @abstractmethod
    def __call__(self: Self, x: Float[Array, "..."], y: Float[Array, "..."]) -> Float[Array, ""]:
        pass


class Mapper:
    """
    Functor for map_and_loss in train, map_loss_in_batches, etc, where arguments can be provided
    beforehand. In this case, it is useful for smse vs relative error, and whether to learn the
    residual or not.
    """

    losses: Sequence[Loss]

    def __init__(self: Self, losses: Loss | Sequence[Loss]) -> None:
        """
        Docstring for __init__

        args:
            losses: a sequence of losses, must be at least 1
            residual: Whether the network should learn the residual, defaults to False
            reduce: How to reduce the batch dimension, defaults to 'mean' but can also be None
            eps: epsilon value to use for nrmse and lr_rel, avoid dividing by 0
        """
        losses = [losses] if isinstance(losses, Loss) else losses
        assert len(losses) > 0, "Mapper::init: At least one loss required."
        self.losses = losses

    @eqx.filter_jit
    def map(
        self: Self, model: eqx.Module, x: Float[Array, "batch in_c spatial"]
    ) -> Float[Array, "batch out_c ..."]:
        """
        Vmap the model so it processes batches, then send the input data through.
        The map function using the model and the input data.
        """
        assert callable(model)
        return jax.vmap(model, axis_name="batch")(x)

    @eqx.filter_jit
    def __call__(
        self: Self,
        model: eqx.Module,
        x: Float[Array, "batch channels spatial"],
        y: Float[Array, "batch out_c ..."],
    ) -> Float[Array, ""]:
        """
        Equivalent of the map_and_loss function.
        """
        pred_y = self.map(model, x)
        loss_outputs = [loss_f(pred_y, y) for loss_f in self.losses]

        # if we aren't reducing the batch dimension, we don't want to squeeze it out
        loss_outputs = jnp.stack(loss_outputs, axis=-1)
        if len(self.losses) == 1:
            squeeze_outputs = jnp.squeeze(loss_outputs, axis=0)
        else:
            squeeze_outputs = loss_outputs

        return squeeze_outputs


def evaluate(
    model: eqx.Module, mapper: Mapper, x: Float[Array, "channels spatial"], y: Float[Array, "..."]
) -> Float[Array, ""]:
    """
    Evaluate the model using the mapper on the input data x and output data y. This pmaps over
    multiple GPUs if available. The number of gpus must evenly divide batch_size as well as as
    any remainder of the MultiImage.

    args:
        model: the model to run through map_and_loss
        mapper: the mapper object
        x: input data
        y: target output data

    Returns:
        Average loss over the entire input data.
    """
    # TODO: do we want to do this here?
    devices = jax.devices()
    x = x.reshape((len(devices), len(x) // len(devices)) + x.shape[1:])
    y = y.reshape((len(devices), len(y) // len(devices)) + y.shape[1:])

    inference_model = eqx.nn.inference_mode(model)
    compute_loss_pmap = eqx.filter_pmap(mapper, axis_name="pmap_batch", in_axes=(None, 0, 0))
    loss = compute_loss_pmap(inference_model, x, y)
    return jnp.mean(loss, axis=0)


def evaluate_loop(mapper: Mapper, model: eqx.Module, dataloader: DataLoader) -> jax.Array:
    """
    Runs map_and_loss for the entire x, y, splitting into batches if the MultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the MultiImage.

    args:
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model: the model to run through map_and_loss
        dataloader: the dataloader for input and output multi image data
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        reduce: how to reduce between batches, defaults to mean

    Returns:
        Average loss over the entire BatchMultiImage
    """
    losses = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.numpy()  # assume that dataloader returns tensors
        y_batch = y_batch.numpy()
        losses.append(evaluate(model, mapper, x_batch, y_batch))

    return jnp.mean(jnp.stack(losses, axis=0), axis=0)


def train_step(
    mapper: Mapper,
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: Any,
    x: Float[Array, "batch channels spatial"],
    y: Float[Array, "batch channels ..."],
) -> tuple[eqx.Module, Any, Float[Array, ""]]:
    """
    Perform one step and gradient update of the model. Uses filter_pmap to use multiple gpus.

    args:
        mapper: map and loss function where the input is a model pytree, x, y, and
            aux_data, and returns a float loss and aux_data
        model: the model
        optim: the optimizer
        opt_state:
        x: input data
        y: target data

    returns:
        model, opt_state, loss_value
    """
    # TODO: do we want to do this here?
    devices = jax.devices()
    x = x.reshape((len(devices), len(x) // len(devices)) + x.shape[1:])
    y = y.reshape((len(devices), len(y) // len(devices)) + y.shape[1:])

    # NOTE: do not `jit` over `pmap` see (https://github.com/google/jax/issues/2926)
    loss_grad = eqx.filter_value_and_grad(mapper)
    compute_loss_pmap = eqx.filter_pmap(loss_grad, axis_name="pmap_batch", in_axes=(None, 0, 0))
    loss, grads = compute_loss_pmap(model, x, y)
    loss = jnp.mean(loss, axis=0)

    get_weights = lambda m: jax.tree_util.tree_leaves(m, is_leaf=eqx.is_array)
    new_grad_arrays = [jnp.mean(x, axis=0) for x in get_weights(grads)]
    grads = eqx.tree_at(get_weights, grads, new_grad_arrays)

    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_loop(
    mapper: Mapper,
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    train_dataloader: DataLoader,
) -> tuple[eqx.Module, Any, Float[Array, ""]]:
    """
    Run through the train_dataloader once, running train step on each mini-batch.
    """
    epoch_loss = []
    for x, y in train_dataloader:
        x = x.numpy()  # assuming that train_dataloader returns tensors
        y = y.numpy()
        model, opt_state, loss_value = train_step(mapper, model, optimizer, opt_state, x, y)
        epoch_loss.append(loss_value)

    return model, opt_state, jnp.mean(jnp.stack(epoch_loss))


def train(
    train_dataloader: DataLoader,
    mapper: Mapper,
    model: eqx.Module,
    total_epochs: int,
    optimizer: optax.GradientTransformation,
    val_dataloader: DataLoader | None = None,
    val_mapper: Mapper | None = None,
    save_model: str | None = None,
    verbose: int = 1,
    is_wandb: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    model_name: str = "",
    args: dict | None = None,
) -> tuple[eqx.Module, Float[Array, ""] | None, Float[Array, ""] | None, float]:
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters that minimize the loss function. The model is returned. This function automatically
    pmaps over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    This version uses pytorch datasets and dataloaders.

    args:
        train_dataloader: dataloader for train input and target data. Each is a MultiImage by k of
            (images, channels, (N,)*D, (D,)*k)
        map_and_loss: function that takes in model, X_batch, Y_batch, and aux_data and
            returns the loss and aux_data.
        model: Model pytree
        stop_condition: when to stop the training process, currently only 1 condition
            at a time
        batch_size: the size of each mini-batch in SGD
        optimizer: optimizer
        val_dataloader: dataloader for val input and target data. Each is a MultiImage by k of
            (images, channels, (N,)*D, (D,)*k)
        save_model: if string, save model every 10 epochs, defaults to None
        aux_data: initial aux data passed in to map_and_loss when has_aux is true.
        is_wandb: whether wandb experiment tracking has been initiated and should be logged to

    returns:
        A tuple of best model in inference mode, aux_data, epoch loss, val loss, and train_time
    """
    if is_wandb:
        args = {} if args is None else args
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=model_name,
            settings=wandb.Settings(),
        )
        type_list = [str, int, float, bool]

        def display_val(val):
            if isinstance(val, enum.Enum):
                return val.name
            elif type(val) in type_list or val is None:
                return val
            else:
                return type(val)

        wandb.config.update(
            {key: display_val(val) for key, val in args.items()},
            allow_val_change=True,
        )

    val_mapper = mapper if val_mapper is None else val_mapper

    total_train_time = 0
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    epoch_loss = None
    epoch_val_loss = None
    for epoch in range(total_epochs):
        start_time = time.time()
        model, opt_state, epoch_loss = train_loop(
            mapper, model, optimizer, opt_state, train_dataloader
        )

        total_train_time += time.time() - start_time
        log = {"train/loss": epoch_loss}

        # We evaluate the validation loss in batches for memory reasons.
        if val_dataloader is not None:
            epoch_val_loss = evaluate_loop(val_mapper, model, val_dataloader)
            log["val/loss"] = epoch_val_loss

        if is_wandb:
            wandb.log(log)

        epoch_time = time.time() - start_time

        if verbose == 2 or (verbose == 1 and epoch % (total_epochs / 10) == 0):
            if epoch_val_loss is not None:
                print(
                    f"Epoch {epoch} Train: {epoch_loss:.7f} Val: {epoch_val_loss:.7f} Epoch time: {epoch_time:.5f}",
                )
            else:
                print(f"Epoch {epoch} Train: {epoch_loss:.7f} Epoch time: {epoch_time:.5f}")

    if save_model:
        save(save_model, model)

    if is_wandb:
        wandb.finish()

    return model, epoch_loss, epoch_val_loss, total_train_time
