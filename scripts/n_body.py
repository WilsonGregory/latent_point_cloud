import argparse
import os
import time

import equinox as eqx
import jax.numpy as jnp
import matplotlib.axes
import matplotlib.pyplot as plt
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from matplotlib.collections import LineCollection
from PIL import Image

from latent_point_cloud import utils

G = 1  # the gravitational constant. These are not in actual units yet, so we can use 1.
TINY = 1e-5  # avoid divide by 0 issues

# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_pos(
    positions: Float[Array, "timestep n_bodies D"],
    velocities: Float[Array, "timestep n_bodies D"],
    gravity_forces: Float[Array, "timestep n_bodies D"],
    ts: Float[Array, " timestep"],
    masses: Float[Array, " n_bodies"],
) -> None:
    D = positions.shape[-1]
    save_loc = "./images/"

    for step in range(len(positions)):
        current_pos = positions[step]

        ax = plt.subplot(1, 1, 1)
        assert isinstance(ax, matplotlib.axes.Axes)
        ax.scatter(current_pos[..., 0], current_pos[..., 1], s=masses * 30, alpha=0.8)
        for one_pos, vel, force in zip(
            current_pos, velocities[step], gravity_forces[step]
        ):  # loop over n_bodies
            ax.annotate(
                "",
                xytext=one_pos,
                xy=one_pos + force,
                arrowprops={"arrowstyle": "->"},
                color="green",
            )
            ax.annotate(
                "", xytext=one_pos, xy=one_pos + vel, arrowprops={"arrowstyle": "->"}, color="blue"
            )

        if step > 0:
            begin = max(0, step - 150)
            start_increments = positions[begin:step].reshape((-1, D))
            stop_increments = positions[begin + 1 : (step + 1)].reshape((-1, D))
            increments = [[start, stop] for start, stop in zip(start_increments, stop_increments)]
            lc = LineCollection(increments, alpha=0.5, linewidth=2, color="red")
            ax.add_collection(lc)

        ax.set_title(f"t={ts[step]:.3f}")
        ax.set_xlim(-8, 24)
        ax.set_ylim(-8, 24)  # hardcoded, should be N
        plt.savefig(f"{save_loc}gravity_step{step}.png")
        plt.close()

    images = []
    frame_images = [x for x in os.listdir(save_loc) if "gravity_step" in x]
    for file_name in sorted(
        frame_images,
        key=lambda x: int(x[x.rfind("gravity_step") + len("gravity_step") : x.rfind(".png")]),
    ):
        file_path = os.path.join(save_loc, file_name)
        images.append(Image.open(file_path))
        os.remove(file_path)

    # 2. Save as an animated GIF
    images[0].save(
        f"{save_loc}gravity_animation.gif",
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
    subkey1, subkey2, subkey3 = random.split(key, num=3)

    # Sample uniformly the cells: (spatial,D)
    indices = jnp.stack(jnp.meshgrid(*(jnp.arange(N),) * D, indexing="ij"), axis=-1)
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
    N: int, bodies_pos: Float[Array, "n_bodies D"], bodies_mass: Float[Array, " n_bodies"]
) -> Float[Array, " spatial"]:
    D = bodies_pos.shape[1]
    # likely a nice jax way of doing this, but this works
    point_mass = jnp.zeros((N,) * D)
    for (x, y), mass in zip(bodies_pos, bodies_mass):
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
    vel_history = [jnp.zeros_like(positions)]
    gravity_history = [gravity_acc]
    t_history = []
    t = 0
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

        t_history.append(t)
        pos_history.append(positions)
        vel_history.append(velocities)
        gravity_history.append(gravity_acc)

    return (
        jnp.stack(pos_history),
        jnp.stack(vel_history),
        jnp.stack(gravity_history),
        jnp.stack(t_history),
    )


def get_data(
    N: int,
    D: int,
    n_bodies: int,
    dt: float,
    T: float,
    key: PRNGKeyArray,
    num_images: int,
    uniform_mass: bool,
    zero_velocity: bool,
) -> tuple[
    Float[Array, "n_images spatial"],
    Float[Array, "n_images n_bodies D"],
    Float[Array, "n_images n_bodies"],
]:
    positions, masses, velocities = eqx.filter_vmap(get_initial_bodies_cloud)(
        N, D, n_bodies, random.split(key, num=num_images), uniform_mass, zero_velocity
    )
    input_images = eqx.filter_vmap(get_input_image)(N, positions, masses)

    pos_history, vel_history, gravity_history, ts = eqx.filter_vmap(integrate_over_time)(
        positions, masses, velocities, dt, T
    )
    plot_pos(pos_history[0], vel_history[0], gravity_history[0], ts[0], masses[0])

    return input_images, pos_history[:, -1], masses


def handleArgs() -> argparse.Namespace:
    """
    CUDA_VISIBLE_DEVICES=2,3 time python3 scripts/n_body.py \
    --data /data/wgregor4/contractility/ZyxAct_16kPa_small/ \
    --n-train 232 --n-val 112 --n-test 8 -b 4 -e 10 \
    --rollout-steps 5 \
    --preprocess percell_nonmask_mean,percell_nonmask_scale,mean,std
    
    --model-dir /data/wgregor4/runs/contractility/ \
    --images-dir /data/wgregor4/images/contractility/ \

    Can do --n-val 128, but for speed do this
    """
    parser = utils.get_common_parser()
    parser.add_argument("-N", help="image side lengths", type=int)

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
    key = random.PRNGKey(time.time_ns())
    N = 16
    D = 2
    n_bodies = 10
    n_images = 1
    dt = 0.01
    T = 10

    data = get_data(N, D, n_bodies, dt, T, key, n_images)
