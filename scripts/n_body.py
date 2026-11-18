import argparse
import itertools as it
import math
import os
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray
from matplotlib.collections import LineCollection
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from latent_point_cloud import training, utils

G = 1  # the gravitational constant. These are not in actual units yet, so we can use 1.
TINY = 1e-5  # avoid divide by 0 issues
LETTERS = "abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_pos(
    image_dir: pathlib.Path,
    data_name: str,
    positions: Float[Array, "timestep n_bodies D"],
    velocities: Float[Array, "timestep n_bodies D"],
    gravity_accs: Float[Array, "timestep n_bodies D"],
    ts: Float[Array, " timestep"],
    masses: Float[Array, " n_bodies"],
) -> None:
    D = positions.shape[-1]

    for step in range(len(positions)):
        current_pos = positions[step]

        ax = plt.subplot(1, 1, 1)
        assert isinstance(ax, matplotlib.axes.Axes)
        ax.scatter(current_pos[..., 0], current_pos[..., 1], s=masses * 30, alpha=0.8)
        for one_pos, vel, force in zip(
            current_pos, velocities[step], gravity_accs[step]
        ):  # loop over n_bodies
            ax.annotate("", xytext=one_pos, xy=one_pos + force, arrowprops={"arrowstyle": "->"})
            ax.annotate("", xytext=one_pos, xy=one_pos + vel, arrowprops={"arrowstyle": "->"})

        if step > 0:
            begin = max(0, step - 200)
            start_increments = positions[begin:step].reshape((-1, D))
            stop_increments = positions[begin + 1 : (step + 1)].reshape((-1, D))
            increments = [[start, stop] for start, stop in zip(start_increments, stop_increments)]
            lc = LineCollection(increments, alpha=0.5, linewidth=2, color="red")
            ax.add_collection(lc)

        ax.set_title(f"t={ts[step]:.3f}")
        extent = float(jnp.max(jnp.abs(positions))) * 1.1  # 10% increase on the extent bounds
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        plt.savefig(image_dir / f"gravity_step{step}.png")
        plt.close()

    images = []
    frame_images = [x for x in os.listdir(image_dir) if "gravity_step" in x]
    for file_name in sorted(
        frame_images,
        key=lambda x: int(x[x.rfind("gravity_step") + len("gravity_step") : x.rfind(".png")]),
    ):
        images.append(Image.open(image_dir / file_name))
        os.remove(image_dir / file_name)

    # 2. Save as an animated GIF
    images[0].save(
        image_dir / f"{data_name}_animation.gif",
        save_all=True,  # Ensures all frames are included, not just the first one
        append_images=images[1:],  # Appends the rest of the frames
        duration=40,  # Duration of each frame in milliseconds (e.g., 200ms = 5 FPS)
        loop=0,  # 0 means infinite loop; omit or change for specific iterations
    )


# DATA GENERATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_indices(
    spatial_dims: tuple[int, ...],
) -> tuple[Int[Array, "spatial D"], Float[Array, "spatial D"]]:
    D = len(spatial_dims)
    if spatial_dims != (spatial_dims[0],) * D:
        raise NotImplementedError(f"get_indices: spatial_dims must be square, got {spatial_dims}")

    spatial_range = [jnp.arange(N) for N in spatial_dims]
    indices = jnp.stack(jnp.meshgrid(*spatial_range, indexing="ij"), axis=-1)

    spatial_linspace = [jnp.linspace(-1, 1, num=N) for N in spatial_dims]
    positions = jnp.stack(jnp.meshgrid(*spatial_linspace, indexing="ij"), axis=-1)
    # indices and positions need to use the same indexing so that rotations work properly.
    # But then when they are plotted (especially if using imshow), you need to make sure to swap
    # everything properly.
    return indices, positions


def get_initial_bodies_cloud(
    N: int, D: int, n_bodies: int, key: PRNGKeyArray, uniform_mass: bool, initial_velocity: bool
) -> tuple[
    Int[Array, "n_bodies D"],
    Float[Array, "n_bodies D"],
    Float[Array, " n_bodies"],
    Float[Array, "n_bodies D"],
]:
    """
    Generate the masses, positions, and initial velocities for n_bodies. If uniform_mass is false,
    the masses are sorted from highest mass to lowest mass.

    args:
        N: the size of the image to place them in.
        D: the dimension of the space
        n_bodies: the number of bodies in the simulation
        key: key for randomness
        uniform_mass: if true, all bodies have the same mass, otherwise they are uniform [0.5,1]
        initial_velocity: if true, generate random velocities from a normal distribution

    returns:
        the positions in D dimensions, masses, and velocities in D dimensions
    """
    if (N % 2) != 0:
        raise ValueError(f"Side length must be even, got {N}")

    subkey1, subkey2, subkey3 = random.split(key, num=3)
    indices, positions = get_indices((N,) * D)

    # Sample uniformly the cells: (spatial,D)
    choices = random.choice(subkey1, N**D, shape=(n_bodies, 1), replace=False, axis=0)
    indices = jnp.take_along_axis(indices.reshape(-1, D), choices, axis=0)
    positions = jnp.take_along_axis(positions.reshape(-1, D), choices, axis=0)

    # generate masses where the total mass is 20
    if uniform_mass:
        masses = 20 * jnp.ones(n_bodies) / n_bodies  # particles equal in size
    else:
        masses = jnp.sort(random.uniform(subkey2, shape=(n_bodies,), minval=0.5), descending=True)
        masses = masses * (2 / jnp.sum(masses))  # masses sum to 2

    if initial_velocity:
        velocities = random.normal(subkey3, shape=(n_bodies, D))
        velocities -= jnp.mean(masses[:, None] * velocities, 0) / jnp.mean(masses)
    else:
        velocities = jnp.zeros((n_bodies, D))

    return indices, positions, masses, velocities


def get_input_image(
    N: int, indices: Int[Array, "n_bodies D"], masses: Float[Array, " n_bodies"]
) -> Float[Array, "1 spatial"]:
    D = indices.shape[1]
    # likely a nice jax way of doing this, but this works
    point_mass = jnp.zeros((N,) * D)
    for (x, y), mass in zip(indices, masses):
        point_mass = point_mass.at[x, y].set(mass)

    return point_mass[None]


def compute_gravity_force(
    bodies_pos: Float[Array, "n_bodies D"],
    bodies_mass: Float[Array, " n_bodies"],
    softening: float,
) -> Float[Array, "n_bodies D"]:
    """
    Softening is an idea from https://github.com/pmocz/nbody-python/blob/master/nbody.py
    """
    n_bodies = len(bodies_pos)

    # along the diagonal of the first 2 elements will be the dist to itself, which should be 0
    r_vecs = bodies_pos[None] - bodies_pos[:, None]  # (n_bodies,n_bodies,D) antisymmetric
    r_cubed = jnp.sum(r_vecs**2, axis=-1, keepdims=True) + (softening**2)  # (n_bodies,n_bodies,1)
    # r_norm = jnp.linalg.norm(r_vecs, axis=-1, keepdims=True)  # (n_bodies,n_bodies,1) symmetric
    # r_vecs_unit = r_vecs / (r_norm + TINY)

    # (n_bodies,n_bodies,D)
    gravity_force = (G * bodies_mass[None, :, None] * r_vecs) / r_cubed
    diags = jnp.full(gravity_force.shape, jnp.eye(n_bodies)[..., None] != 0)
    gravity_force = jnp.where(diags, 0, gravity_force)  # set the diag vectors to 0
    return jnp.sum(gravity_force, axis=1)  # (n_bodies,D)


def integrate_over_time(
    positions: Float[Array, "n_bodies D"],
    masses: Float[Array, " n_bodies"],
    velocities: Float[Array, "n_bodies D"],
    dt: float,
    T: float,
) -> tuple[
    Float[Array, "timestep n_bodies D"],
    Float[Array, "timestep n_bodies D"],
    Float[Array, "timestep n_bodies D"],
    Float[Array, " timestep"],
]:
    """
    Given starting positions and masses, run the n-body simulation for a total of T time, in dt increments.
    There is probably very accurate code out there to do this I imagine?
    """

    gravity_acc = compute_gravity_force(positions, masses, 0.1)
    pos_history = [positions]
    vel_history = [velocities]
    gravity_history = [gravity_acc]
    t = 0.0
    ts = [t]
    while t < T:
        # kick-drift-kick from https://github.com/pmocz/nbody-python/blob/master/nbody.py
        # (1/2) kick
        velocities += gravity_acc * dt / 2.0

        # drift
        positions += velocities * dt

        # update accelerations
        gravity_acc = compute_gravity_force(positions, masses, 0.1)

        # (1/2) kick
        velocities += gravity_acc * dt / 2.0

        # update time
        t += dt

        pos_history.append(positions)
        vel_history.append(velocities)
        gravity_history.append(gravity_acc)
        ts.append(t)

    return (
        jnp.stack(pos_history),
        jnp.stack(vel_history),
        jnp.stack(gravity_history),
        jnp.stack(ts),
    )


class NBodyDataset(Dataset):
    input_images: Float[Array, "n_samples channels spatial"]
    last_positions: Float[Array, "n_samples n_bodies 1 D"]
    last_velocities: Float[Array, "n_samples n_bodies 1 D"]
    masses: Float[Array, "n_samples n_bodies"]

    def __init__(
        self: Self,
        root: pathlib.Path,
        data_type: Literal["train", "val", "test"],
        D: int,
        N: int,
        n_samples: int,
        n_bodies: int,
        uniform_mass: bool,
        initial_velocity: bool,
        dt: float,
        T: float,
        key: PRNGKeyArray,
        image_dir: pathlib.Path | None = None,
    ) -> None:
        uniform_mass_str = "unif-mass" if uniform_mass else "no-unif-mass"
        initial_velocity_str = "initial-velocity" if initial_velocity else "no-initial-velocity"
        data_name = f"{D}D_L{n_samples}_N{N}_nbodies{n_bodies}_{uniform_mass_str}_{initial_velocity_str}_dt{dt}_T{T}"
        file_path = root / data_type / f"{data_name}.npy"
        (root / data_type).mkdir(exist_ok=True)  # construct train/val/test dir if its missing

        if file_path.is_file():
            data_dict = jnp.load(file_path, allow_pickle=True).item()
            masses = data_dict["masses"]
            indices = data_dict["indices"]  # (n_samples, n_bodies, D)
            positions = data_dict["positions"]  # (n_samples, timesteps, n_bodies, tensor)
            velocities = data_dict["velocities"]
            gravity_accs = data_dict["gravity_accs"]
            ts = data_dict["ts"]
        else:
            indices, initial_positions, masses, initial_velocities = eqx.filter_vmap(
                get_initial_bodies_cloud
            )(N, D, n_bodies, random.split(key, num=n_samples), uniform_mass, initial_velocity)

            # (n_samples, timesteps, n_bodies, tensor)
            positions, velocities, gravity_accs, ts = eqx.filter_vmap(integrate_over_time)(
                initial_positions, masses, initial_velocities, dt, T
            )

            # save positions, velocities, masses, and output positions, construct images at runtime
            jnp.save(
                file_path,
                {
                    "indices": indices,
                    "positions": positions,
                    "masses": masses,
                    "velocities": velocities,
                    "gravity_accs": gravity_accs,
                    "ts": ts,
                },
                allow_pickle=True,
            )

        self.masses = masses
        self.input_images = eqx.filter_vmap(get_input_image)(N, indices, self.masses)
        self.last_positions = positions[:, -1, :, None]
        self.last_velocities = velocities[:, -1, :, None]

        if image_dir is not None:
            plot_pos(
                image_dir,
                data_name,
                positions[0],
                velocities[0],
                gravity_accs[0],
                ts[0],
                masses[0],
            )

    def __len__(self: Self) -> int:
        return len(self.input_images)

    def __getitem__(self: Self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        # output as numpy array to be handled automatically by pytorch's dataloader
        return np.asarray(self.input_images[idx]), np.asarray(self.last_positions[idx])

        # TODO: velocities are currently unused
        # return np.asarray(self.input_images[idx]), np.asarray(
        #     jnp.concat([self.last_positions[idx], self.last_velocities[idx]], axis=1)
        # )


def get_data(
    data_dir: pathlib.Path,
    D: int,
    N: int,
    n_bodies: int,
    dt: float,
    T: float,
    n_train: int,
    n_val: int,
    n_test: int,
    uniform_mass: bool,
    initial_velocity: bool,
    key: PRNGKeyArray,
    batch_size: int,
    image_dir: pathlib.Path,
    plot_gif: bool,
) -> tuple[DataLoader[NBodyDataset], DataLoader[NBodyDataset], DataLoader[NBodyDataset]]:
    subkey1, subkey2, subkey3 = random.split(key, num=3)
    train_dataset = NBodyDataset(
        data_dir,
        "train",
        D,
        N,
        n_train,
        n_bodies,
        uniform_mass,
        initial_velocity,
        dt,
        T,
        subkey1,
        image_dir if plot_gif else None,
    )

    val_dataset = NBodyDataset(
        data_dir, "val", D, N, n_val, n_bodies, uniform_mass, initial_velocity, dt, T, subkey2
    )

    test_dataset = NBodyDataset(
        data_dir, "test", D, N, n_test, n_bodies, uniform_mass, initial_velocity, dt, T, subkey3
    )
    # TODO: no normalization?

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


# Model and Training


class MSE(training.Loss):
    def __call__(
        self: Self, x: Float[Array, "batch channels ..."], y: Float[Array, "batch channels ..."]
    ) -> Float[Array, ""]:
        # mean over batch, squared error over other stuff
        return jnp.mean(jnp.sum((x - y) ** 2, axis=tuple(range(1, x.ndim))))


class MSEPermPointLoss(training.Loss):
    """
    The MSE loss between points, taking the minimum over permutations of the points. This should be
    used as a metric at the end, not as a training loss.
    """

    def __call__(
        self: Self,
        x: Float[Array, "batch n_particles channels D"],
        y: Float[Array, "batch n_particles channels D"],
    ) -> Float[Array, ""]:
        losses = []
        for x_, y_ in zip(x, y):  # loop over batches
            losses_ = []
            for perm in it.permutations(range(len(x_))):
                x_perm = x_[jnp.array(perm)]
                losses_.append(jnp.sum((x_perm - y_) ** 2))

            losses.append(jnp.min(jnp.stack(losses_)))  # min over permutations of points

        return jnp.mean(jnp.stack(losses))  # mean over batch


def count_params(model: eqx.Module) -> int:
    get_array_sizes = lambda m: [
        x.size for x in jax.tree_util.tree_leaves(m, is_leaf=eqx.is_array) if eqx.is_array(x)
    ]
    return sum(get_array_sizes(model))


class MLPOnly(eqx.Module):
    """
    An even simpler model, an mlp from the image to positions.
    """

    D: int
    n_particles: int
    activation_f: Callable

    layers: list[eqx.nn.Linear]
    decoder: eqx.nn.Linear

    def __init__(
        self: Self,
        D: int,
        in_c: int,
        n_particles: int,
        width: int,
        n_hidden_layers: int,
        spatial_size: int,
        activation_f: Callable = jax.nn.gelu,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.D = D
        self.n_particles = n_particles
        self.activation_f = activation_f

        key, subkey = random.split(key)
        self.layers = [eqx.nn.Linear(spatial_size * in_c, width, key=subkey)]
        for _ in range(n_hidden_layers):
            key, subkey = random.split(key)
            self.layers.append(eqx.nn.Linear(width, width, key=subkey))

        key, subkey = random.split(key)
        self.decoder = eqx.nn.Linear(width, n_particles * D * 2, key=subkey)

    def __call__(self: Self, x: Float[Array, "in_c spatial"]) -> Float[Array, "n_particles 2 D"]:
        x = x.ravel()
        for layer in self.layers:
            x = self.activation_f(layer(x))

        x = self.decoder(x.ravel())
        positions = x[: self.n_particles * self.D].reshape(self.n_particles, self.D)
        velocities = x[self.n_particles * self.D : 2 * self.n_particles * self.D].reshape(
            self.n_particles, self.D
        )

        return positions[:, None]


class SimpleConv(eqx.Module):
    """
    A simple convolution model, takes in an image with in_c channels and outputs n_particles with
    positions and velocities. The model consists of convolution layers with nonlinearities,
    followed by a single fully connected linear layer to the output.
    """

    D: int
    n_particles: int
    activation_f: Callable

    layers: list[eqx.nn.Conv]
    decoder: eqx.nn.Linear

    def __init__(
        self: Self,
        D: int,
        in_c: int,
        n_particles: int,
        width: int,
        n_hidden_layers: int,
        spatial_size: int,
        kernel_size: int | Sequence[int] = 3,
        activation_f: Callable = jax.nn.gelu,
        *,
        key: PRNGKeyArray,
    ) -> None:
        assert n_hidden_layers > 0

        self.D = D
        self.n_particles = n_particles
        self.activation_f = activation_f

        key, subkey = random.split(key)
        self.layers = [eqx.nn.Conv(D, in_c, width, kernel_size, padding="SAME", key=subkey)]
        for _ in range(n_hidden_layers):
            key, subkey = random.split(key)
            self.layers.append(
                eqx.nn.Conv(D, width, width, kernel_size, padding="SAME", key=subkey)
            )

        key, subkey = random.split(key)
        self.decoder = eqx.nn.Linear(width * spatial_size, n_particles * D * 2, key=subkey)

    def __call__(self: Self, x: Float[Array, "in_c spatial"]) -> Float[Array, "n_particles 2 D"]:
        for layer in self.layers:
            x = self.activation_f(layer(x))

        x = self.decoder(x.ravel())
        positions = x[: self.n_particles * self.D].reshape(self.n_particles, self.D)
        velocities = x[self.n_particles * self.D : 2 * self.n_particles * self.D].reshape(
            self.n_particles, self.D
        )

        return positions[:, None]
        # return jnp.stack([positions, velocities], axis=1)


class LatentPointCloud(eqx.Module):
    D: int
    n_particles: int
    activation_f: Callable

    conv_layers: list[eqx.nn.Conv]
    connecter_layers: list[eqx.nn.Linear]

    def __init__(
        self: Self,
        D: int,
        image_c: int,
        n_particles: int,
        conv_width: int,
        n_hidden_conv: int,
        spatial_dims: Sequence[int],
        connecter_width: int,
        n_hidden_connecter: int,
        kernel_size: int | Sequence[int] = 3,
        activation_f: Callable = jax.nn.gelu,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.D = D
        self.n_particles = n_particles
        self.activation_f = activation_f

        key, subkey = random.split(key)
        layer_widths = [image_c] + [conv_width] * n_hidden_conv
        self.conv_layers = []
        for i in range(len(layer_widths) - 1):
            in_c, out_c = layer_widths[i : i + 2]
            key, subkey = random.split(key)
            self.conv_layers.append(
                eqx.nn.Conv(D, in_c, out_c, kernel_size, padding="SAME", key=subkey)
            )

        last_c = layer_widths[-1]
        spatial_size = math.prod(spatial_dims)
        in_c = math.comb(spatial_size, 2) + spatial_size + spatial_size * last_c
        layer_widths = (
            [in_c] + [connecter_width] * n_hidden_connecter + [spatial_size * n_particles]
        )
        self.connecter_layers = []
        for i in range(len(layer_widths) - 1):
            in_c, out_c = layer_widths[i : i + 2]
            key, subkey = random.split(key)
            self.connecter_layers.append(eqx.nn.Linear(in_c, out_c, key=subkey))

    def __call__(
        self: Self, x: Float[Array, "in_c spatial"]
    ) -> Float[Array, "n_particles channels D"]:
        # pass it through the conv layers
        for layer in self.conv_layers[:-1]:
            x = self.activation_f(layer(x))

        x = self.conv_layers[-1](x)

        spatial_dims = x.shape[1:]
        x = x.reshape(x.shape[0], -1)  # (channels,spatial) -> (channels,spatial_flat)

        # (spatial,D)
        _, grid_positions = get_indices(spatial_dims)

        grid_positions = grid_positions.reshape(-1, self.D)
        n_points = len(grid_positions)

        # TODO: grid_positions is fixed, so this will always be the same thing, seems silly.
        dots = grid_positions @ grid_positions.T  # (n_points,n_points)
        dots = dots[jnp.triu_indices(n_points)].reshape(-1)  # (n_points choose 2,)

        x = jnp.concat([dots, x.ravel()])

        for layer in self.connecter_layers[:-1]:
            x = self.activation_f(layer(x))

        coeffs = self.connecter_layers[-1](x).reshape(self.n_particles, math.prod(spatial_dims))
        # scalars are universal setup
        coeffs = coeffs / (jnp.sum(coeffs, axis=1, keepdims=True) + 1e-5)  # (in_c,)
        out = jnp.sum(coeffs[..., None] * grid_positions[None], axis=1)  # (n_particles,D)

        return out[:, None]


def train_and_eval(
    data: tuple[DataLoader[NBodyDataset], DataLoader[NBodyDataset], DataLoader[NBodyDataset]],
    model: eqx.Module,
    model_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    model_dir: pathlib.Path | None = None,
    overwrite_save_model: bool = False,
    images_dir: pathlib.Path | None = None,
    verbose: int = 1,
    is_wandb: bool = False,
    wandb_project: str = "",
) -> tuple[Float[Array, ""], ...]:
    if is_wandb:
        assert wandb_project != "", "wandb_project must be provided when is_wandb=True"

    train_dl, val_dl, test_dl = data
    assert isinstance(train_dl.dataset, NBodyDataset)

    mapper = training.Mapper(MSE())

    model_name_extended = f"{model_name}_L{len(train_dl.dataset)}_e{epochs}"
    print(f"{model_name_extended} params: {count_params(model):,}")

    model_path = model_dir / f"{model_name_extended}.eqx" if model_dir else None
    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model, _ = training.load(model_path, model)
    else:
        steps_per_epoch = math.ceil(len(train_dl.dataset) / batch_size)
        trained_model, _, _, _ = training.train(
            train_dl,
            mapper,
            model,
            epochs,
            optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            val_dl,
            val_mapper=mapper,
            save_model=model_path,
            verbose=verbose,
            is_wandb=is_wandb,
            wandb_project=wandb_project,
            model_name=model_name_extended,
        )

    train_loss = training.evaluate_loop(mapper, trained_model, train_dl)
    print(f"Train Loss: {train_loss}")
    val_loss = training.evaluate_loop(mapper, trained_model, val_dl)
    print(f"Val Loss: {val_loss}")
    test_loss = training.evaluate_loop(mapper, trained_model, test_dl)
    print(f"Test Loss: {test_loss}")

    perm_mapper = training.Mapper(MSEPermPointLoss())
    test_perm_loss = training.evaluate_loop(perm_mapper, trained_model, test_dl)
    print(f"Test Perm Loss: {test_perm_loss}")

    return train_loss, val_loss, test_loss


def handleArgs() -> argparse.Namespace:
    """
    time uv run python3 scripts/n_body.py \
    --data-dir data/n_body/ --n-train 128 --n-val 128 --n-test 128 -N 16 \
    --total-time 1.5 \
    --image-dir images/n_body/
    """
    parser = utils.get_common_parser()

    # Data generation
    parser.add_argument("-D", help="the dimension of the space", type=int, default=2)
    parser.add_argument("-N", "--side-length", help="image side lengths", type=int)
    parser.add_argument(
        "--n-bodies", help="number of bodies to orbit each other", type=int, default=10
    )
    parser.add_argument("--dt", help="time increment for the simulation", type=float, default=0.01)
    parser.add_argument(
        "--total-time", help="total time for the simulation to run", type=float, default=10
    )
    parser.add_argument(
        "--uniform-mass",
        help="use uniform masses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--initial-velocity",
        help="if true, give the bodies initial velocities",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--plot-gif",
        help="whether to plot the gif the input data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="n_body")

    return parser.parse_args()


if __name__ == "__main__":
    args = handleArgs()

    key = random.PRNGKey(time.time_ns())
    data_dir = pathlib.Path(args.data_dir)  # don't let it be None
    image_dir = pathlib.Path(args.image_dir)  # don't let it be None

    key, subkey = random.split(key)
    data = get_data(
        data_dir,
        args.D,
        args.side_length,
        args.n_bodies,
        args.dt,
        args.total_time,
        args.n_train,
        args.n_val,
        args.n_test,
        args.uniform_mass,
        args.initial_velocity,
        subkey,
        args.batch_size,
        image_dir,
        args.plot_gif,
    )

    kwargs = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "verbose": args.verbose,
        "is_wandb": args.wandb,
        "wandb_project": args.wandb_project,
    }

    key, *subkeys = random.split(key, num=10)
    model_list = [
        {
            **kwargs,
            "model_name": "simple_conv",
            "model": SimpleConv(
                args.D, 1, args.n_bodies, 64, 3, args.side_length**args.D, key=subkeys[0]
            ),
            "lr": 3e-4,
        },
        {
            **kwargs,
            "model_name": "mlp_only",
            "model": MLPOnly(
                args.D, 1, args.n_bodies, 64, 0, args.side_length**args.D, key=subkeys[1]
            ),
            "lr": 1e-2,
        },
        {
            **kwargs,
            "model_name": "LatentPointCloud",
            "model": LatentPointCloud(
                args.D, 1, args.n_bodies, 64, 2, (args.side_length,) * args.D, 32, 2, key=subkeys[1]
            ),
            "lr": 1e-3,
        },
    ]

    for model_kwargs in model_list:
        train_and_eval(data, **model_kwargs)
