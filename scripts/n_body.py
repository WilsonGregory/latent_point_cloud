import argparse
import os
import pathlib
import time
from typing import Literal, Self

import equinox as eqx
import jax.numpy as jnp
import matplotlib.axes
import matplotlib.pyplot as plt
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from matplotlib.collections import LineCollection
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from latent_point_cloud import utils

G = 1  # the gravitational constant. These are not in actual units yet, so we can use 1.
TINY = 1e-5  # avoid divide by 0 issues

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


def get_initial_bodies_cloud(
    N: int, D: int, n_bodies: int, key: PRNGKeyArray, uniform_mass: bool, zero_velocity: bool
) -> tuple[Float[Array, "n_bodies D"], Float[Array, " n_bodies"], Float[Array, "n_bodies D"]]:
    if (N % 2) != 0:
        raise ValueError(f"Side length must be even, got {N}")

    subkey1, subkey2, subkey3 = random.split(key, num=3)

    # Sample uniformly the cells: (spatial,D)
    indices = jnp.stack(jnp.meshgrid(*(jnp.arange(N),) * D, indexing="ij"), axis=-1) - N / 2
    positions = random.choice(
        subkey1, indices.reshape((-1, D)), shape=(n_bodies,), replace=False, axis=0
    )

    # generate masses where the total mass is 20
    if uniform_mass:
        masses = 20 * jnp.ones(n_bodies) / n_bodies  # particles equal in size
    else:
        masses = random.uniform(subkey2, shape=(n_bodies,), minval=0.5)
        masses = masses * (20 / jnp.sum(masses))

    if zero_velocity:
        velocities = jnp.zeros((n_bodies, D))
    else:
        velocities = random.normal(subkey3, shape=(n_bodies, D))
        velocities -= jnp.mean(masses[:, None] * velocities, 0) / jnp.mean(masses)

    return positions, masses, velocities


def get_input_image(
    N: int, positions: Float[Array, "n_bodies D"], masses: Float[Array, " n_bodies"]
) -> Float[Array, " spatial"]:
    if (N % 2) != 0:
        raise ValueError(f"Side length must be even, got {N}")

    positions = (positions + N / 2).astype(int)  # return to [0...N] coordinates
    D = positions.shape[1]
    # likely a nice jax way of doing this, but this works
    point_mass = jnp.zeros((N,) * D)
    for (x, y), mass in zip(positions, masses):
        point_mass = point_mass.at[x, y].set(mass)

    return point_mass


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
    input_image: Float[Array, "n_samples spatial"]
    last_positions: Float[Array, "n_samples n_bodies D"]
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
        zero_velocity: bool,
        dt: float,
        T: float,
        key: PRNGKeyArray,
        image_dir: pathlib.Path | None = None,
    ) -> None:
        data_name = (
            f"{D}D_L{n_samples}_nbodies{n_bodies}_{uniform_mass}_{zero_velocity}_dt{dt}_T{T}"
        )
        file_path = root / data_type / f"{data_name}.npy"
        (root / data_type).mkdir(exist_ok=True)  # construct train/val/test dir if its missing

        if file_path.is_file():
            data_dict = jnp.load(file_path, allow_pickle=True).item()
            self.masses = data_dict["masses"]
            positions = data_dict["positions"]
            velocities = data_dict["velocities"]
            gravity_accs = data_dict["gravity_accs"]
            ts = data_dict["ts"]

            self.input_images = eqx.filter_vmap(get_input_image)(N, positions[:, 0], self.masses)
            self.last_positions = positions[:, -1]
        else:
            initial_positions, masses, initial_velocities = eqx.filter_vmap(
                get_initial_bodies_cloud
            )(N, D, n_bodies, random.split(key, num=n_samples), uniform_mass, zero_velocity)
            self.input_images = eqx.filter_vmap(get_input_image)(N, initial_positions, masses)

            # (n_samples, timesteps, n_bodies, tensor)
            positions, velocities, gravity_accs, ts = eqx.filter_vmap(integrate_over_time)(
                initial_positions, masses, initial_velocities, dt, T
            )
            self.last_positions = positions[:, -1]
            self.masses = masses

            # save positions, velocities, masses, and output positions, construct images at runtime
            jnp.save(
                file_path,
                {
                    "positions": positions,
                    "masses": masses,
                    "velocities": velocities,
                    "gravity_accs": gravity_accs,
                    "ts": ts,
                },
                allow_pickle=True,
            )

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

    def __getitem__(
        self: Self, idx: int
    ) -> tuple[Float[Array, " spatial"], Float[Array, "n_bodies D"]]:
        return self.input_images[idx], self.last_positions[idx]


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
    zero_velocity: bool,
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
        zero_velocity,
        dt,
        T,
        subkey1,
        image_dir if plot_gif else None,
    )

    val_dataset = NBodyDataset(
        data_dir, "val", D, N, n_val, n_bodies, uniform_mass, zero_velocity, dt, T, subkey2
    )

    test_dataset = NBodyDataset(
        data_dir, "test", D, N, n_test, n_bodies, uniform_mass, zero_velocity, dt, T, subkey3
    )
    # TODO: no normalization?

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


def handleArgs() -> argparse.Namespace:
    """
    time uv run python3 scripts/n_body.py \
    --data-dir data/n_body/ --n-train 8 --n-val 8 --n-test 8 -N 16 \
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
        "--zero-velocity",
        help="use zero initial velocities",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        args.zero_velocity,
        key,
        args.batch_size,
        image_dir,
        args.plot_gif,
    )
