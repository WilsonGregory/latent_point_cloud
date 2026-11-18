import math
import time
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from ginjax import geometric as geom
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray

TINY = 1e-5


def vector_image_loglog(image: geom.GeometricImage) -> geom.GeometricImage:
    normed_image = jnp.linalg.norm(image.data, axis=-1, keepdims=True)
    return geom.GeometricImage(
        image.data * (jnp.log1p(jnp.log1p(normed_image)) / normed_image), 0, D
    )


def toroidal_center_of_mass(image: geom.GeometricImage) -> Float[Array, " D"]:
    spatial_dims = image.spatial_dims
    spatial_range = [jnp.arange(N) for N in spatial_dims]
    indices = jnp.stack(jnp.meshgrid(*spatial_range, indexing="ij"), axis=-1)  # [0...N-1]

    centralizer = (jnp.array(spatial_dims).reshape((1,) * image.D + (image.D,)) - 1) / 2
    centered_indices = indices - centralizer
    indices_norm = jnp.linalg.norm(centered_indices, axis=-1)
    ff = jnp.max(indices_norm) - indices_norm
    ff_norm = ff / jnp.sum(ff)
    conv_filter = geom.GeometricFilter(ff_norm, 0, D)

    convolved_image = image.convolve_with(conv_filter)
    idx = indices.reshape(-1, image.D)[jnp.argmax(convolved_image.data.ravel())]
    return idx


def getshiftA(image: geom.GeometricImage) -> tuple[Float[Array, " D"], Float[Array, "D D"]]:
    spatial_dims = image.spatial_dims
    spatial_range = [jnp.arange(N) for N in spatial_dims]
    # (spatial,D)
    indices = jnp.stack(jnp.meshgrid(*spatial_range, indexing="ij"), axis=-1)  # [0...N-1]

    spatial_dims_arr = jnp.array(spatial_dims)
    centralizer = (spatial_dims_arr - 1) / 2  # (D,)
    centered_indices = indices - centralizer.reshape((1,) * D + (D,))

    topk = image.D + 1
    # topk indices on the flattened image
    topk_pixel_idxs = jnp.argsort(image.data.ravel(), descending=True)[:topk]

    # topk indices as vectors on the d-dimensional image.
    topk_indices = centered_indices.reshape(-1, image.D)[topk_pixel_idxs]  # (topk,D)

    assert topk_indices.shape == (topk, image.D)

    shift = topk_indices[0]  # (D,)
    A = topk_indices[1:].T  # (topk-1,D) -> (D,topk-1)

    return shift, A


def shiftQR(
    image: geom.GeometricImage,
) -> tuple[Float[Array, " D"], Float[Array, "D D"], Float[Array, "D D"]]:
    shift, A = getshiftA(image)

    A_minus_shift = A - shift[:, None]
    if image.is_torus == (True,) * image.D:
        # indices are centralized, so uncentralize, get remainder, then recentralize
        spatial_dims_arr = jnp.array(image.spatial_dims)
        centralizer = (spatial_dims_arr - 1) / 2  # (D,)
        A_minus_shift = jnp.remainder(A_minus_shift + centralizer, spatial_dims_arr) - centralizer

    Q, R = jnp.linalg.qr(A_minus_shift)

    # set the diagonal to all positive
    sign_adjustment = jnp.diag(jnp.sign(jnp.diag(R)))
    Q = Q @ sign_adjustment
    R = sign_adjustment @ R

    return shift, Q, R


def testShiftQREquivariance(D: int, N: int, key: PRNGKeyArray) -> None:

    spatial_dims = (N,) * D
    spatial_dims_arr = jnp.array(spatial_dims)
    centralizer = (spatial_dims_arr - 1) / 2

    # First test the it is true on the torus
    key, subkey = random.split(key)
    img1 = geom.GeometricImage(random.uniform(subkey, shape=spatial_dims), 0, D, is_torus=True)

    translations = img1.key_array()
    rotations = geom.make_all_operators(D)

    for translation in translations:
        for rotation in rotations:
            shift, Q, R = shiftQR(img1)

            # rotate then translate, so inverse needs to translate, then rotate
            shift_, Q_, R_ = shiftQR(img1.times_gg_precise(rotation).translate(translation))

            # shift_ = rotation @ shift + translation
            transformed_shift = (
                jnp.remainder(rotation @ shift + translation + centralizer, spatial_dims_arr)
                - centralizer
            )

            assert jnp.allclose(transformed_shift, shift_), (
                f"Does {rotation} @ {shift} + {translation} = {shift_}?"
            )

            assert jnp.allclose(rotation @ Q, Q_), f"Does {rotation} @ {Q} = {Q_}?"

            # test that R is invariant to transformations
            assert jnp.allclose(R, R_), f"Does R={R} equal R_={R_}?"

    # Now test not on the torus.
    # pad N in every direction
    img2 = geom.GeometricImage(jnp.pad(img1.data, N), 0, D, is_torus=False)
    for translation in translations:
        for rotation in rotations:
            shift, Q, R = shiftQR(img2)
            transformed_img = img2.times_gg_precise(rotation).translate(translation)
            shift_, Q_, R_ = shiftQR(transformed_img)

            # shift_ = rotation @ shift + translation
            transformed_shift = rotation @ shift + translation
            if not jnp.allclose(transformed_shift, shift_):
                img2.plot()
                transformed_img.plot()
                print(translation)
                print(rotation)
                print(shift)
                print(shift_)
                plt.show()

            assert jnp.allclose(transformed_shift, shift_), (
                f"Does {rotation} @ {shift} + {translation} = {shift_}?"
            )

            if not jnp.allclose(rotation @ Q, Q_):
                img2.plot()
                transformed_img.plot()
                print(translation)
                print(rotation)
                print(Q)
                print(Q_)
                plt.show()

            assert jnp.allclose(rotation @ Q, Q_), f"Does {rotation} @ {Q} = {Q_}?"

            if not jnp.allclose(R, R_):
                img2.plot()
                transformed_img.plot()
                print(translation)
                print(rotation)
                print(R)
                print(R_)
                plt.show()

            # test that R is invariant to transformations
            assert jnp.allclose(R, R_), f"Does R={R} equal R_={R_}?"


def testCenterOfMassEquivariance(D: int, N: int) -> None:
    """
    Test that the center of mass function is translation equivariant, both on the torus and off
    the torus. The two below statements are equivalent.
    1. CoM(translated_A) = CoM(A) + translation (on or off torus)
    2. CoM(translated_A) - translation = CoM(A)
    """
    spatial_dims = (N,) * D
    spatial_dims_arr = jnp.array(spatial_dims)
    centralizer = (spatial_dims_arr - 1) / 2

    # First test the it is true on the torus
    img1 = geom.GeometricImage(random.uniform(subkey, shape=spatial_dims), 0, D, is_torus=True)

    translations = img1.key_array()
    rotations = geom.make_all_operators(D)

    for translation in translations:
        for rotation in rotations:
            idx = toroidal_center_of_mass(img1)
            # rotate then translate, so inverse needs to translate, then rotate
            transformed_idx = toroidal_center_of_mass(
                img1.times_gg_precise(rotation).translate(translation)
            )

            # do g^{-1} @ (centered_idx - translation)
            # translate back on the torus
            untranslate_idx = jnp.remainder(transformed_idx - translation, spatial_dims_arr)
            # rotate back
            restored_idx = rotation.T @ (untranslate_idx - centralizer) + centralizer

            assert jnp.allclose(restored_idx, idx), (
                f"Does {rotation.T} @ {transformed_idx - centralizer} - {translation} = {idx - centralizer}?"
            )

    # Now test not on the torus.
    # pad N in every direction
    img2 = geom.GeometricImage(jnp.pad(img1.data, N), 0, D, is_torus=False)
    centralizer = (jnp.array(img2.spatial_dims) - 1) / 2
    for translation in translations:
        for rotation in rotations:
            idx = toroidal_center_of_mass(img2)
            # rotate then translate, so inverse needs to translate, then rotate

            transformed_idx = toroidal_center_of_mass(
                img2.times_gg_precise(rotation).translate(translation)
            )

            # do g^{-1} @ (centered_idx - translation)
            # translate back, then rotate back
            restored_idx = rotation.T @ (transformed_idx - translation - centralizer) + centralizer

            assert jnp.allclose(restored_idx, idx), (
                f"Does {rotation.T} @ {transformed_idx - centralizer} - {translation} = {idx - centralizer}?"
            )


def testVecsOnlyEquivariance(D: int, N: int, key: PRNGKeyArray) -> None:
    """
    What I want:
    model(translated_A) = model(A) - translation
    """
    spatial_dims = (N,) * D
    key, subkey = random.split(key)
    vecs_model = VecsOnlyModel(math.prod(spatial_dims), 1, 32, subkey)

    # First test that it is true on the torus
    img1 = geom.GeometricImage(random.uniform(subkey, shape=spatial_dims), 0, D, is_torus=True)

    translations = img1.key_array()
    rotations = geom.make_all_operators(D)

    # Now test that it is true not on the torus.
    # pad N in every direction
    img2 = geom.GeometricImage(jnp.pad(img1.data, N), 0, D, is_torus=False)
    key, subkey = random.split(key)
    vecs_model = VecsOnlyModel(math.prod(img2.spatial_dims), 1, 32, subkey)
    centralizer = (jnp.array(img2.spatial_dims) - 1) / 2
    for translation in tqdm.tqdm(translations):
        for rotation in rotations:
            shift, Q, _ = shiftQR(img2)

            # rotate then translate
            transformed_img2 = img2.times_gg_precise(rotation).translate(translation)
            shift_, Q_, _ = shiftQR(transformed_img2)

            positions = img2.key_array().reshape(-1, D) - centralizer[None]  # (spatial_flat,D)

            out1 = vecs_model((Q @ positions.T).T - shift[None])  # (1,D)
            out2 = vecs_model((Q_ @ positions.T).T - shift_[None])

            # rotate then translate out1
            transformed_out1 = (
                jnp.einsum("ij,lj->li", rotation, out1, precision=jax.lax.Precision.HIGHEST)
                - translation[None]
            )

            if not jnp.allclose(out2, transformed_out1, rtol=1e-3, atol=1e-3):
                print("diff, translation, rotation, out1, out2, transformed_out1")
                print(jnp.max(jnp.abs(out2 - transformed_out1)))
                print(translation)
                print(rotation)
                print(out1)
                print(out2)
                print(transformed_out1)

            assert jnp.allclose(out2, transformed_out1, rtol=1e-3, atol=1e-3), (
                f"Does {out2} = {rotation} @ {out1} - {translation} ?"
            )


def testVecsScalarsEquivariance(D: int, N: int, key: PRNGKeyArray) -> None:
    """
    What I want:
    model(translated_A) = model(A) - translation
    """
    spatial_dims = (N,) * D

    # First test that it is true on the torus
    key, subkey = random.split(key)
    img1 = geom.GeometricImage(random.uniform(subkey, shape=spatial_dims), 0, D, is_torus=True)

    translations = img1.key_array()
    rotations = geom.make_all_operators(D)

    # Now test that it is true not on the torus.
    # pad N in every direction
    img2 = geom.GeometricImage(jnp.pad(img1.data, N), 0, D, is_torus=False)
    key, subkey = random.split(key)
    model = VecsScalarsModel(math.prod(img2.spatial_dims), 1, 32, subkey)
    centralizer = (jnp.array(img2.spatial_dims) - 1) / 2
    for translation in tqdm.tqdm(translations):
        for rotation in rotations:
            shift, Q, _ = shiftQR(img2)

            # rotate then translate
            transformed_img2 = img2.times_gg_precise(rotation).translate(translation)
            shift_, Q_, _ = shiftQR(transformed_img2)

            positions = img2.key_array().reshape(-1, D) - centralizer[None]  # (spatial_flat,D)

            out1 = model((Q @ positions.T).T - shift[None], img2.data.ravel())  # (1,D)
            out2 = model((Q_ @ positions.T).T - shift_[None], transformed_img2.data.ravel())

            # rotate then translate out1
            transformed_out1 = (
                jnp.einsum("ij,lj->li", rotation, out1, precision=jax.lax.Precision.HIGHEST)
                - translation[None]
            )

            if not jnp.allclose(out2, transformed_out1, rtol=1e-2, atol=1e-2):
                print("diff, translation, rotation, out1, out2, transformed_out1")
                print(jnp.max(jnp.abs(out2 - transformed_out1)))
                print(translation)
                print(rotation)
                print(out1)
                print(out2)
                print(transformed_out1)

            assert jnp.allclose(out2, transformed_out1, rtol=1e-2, atol=1e-2), (
                f"Does {out2} = {rotation} @ {out1} - {translation} ?"
            )


class VecsOnlyModel(eqx.Module):
    first: eqx.nn.Linear
    second: eqx.nn.Linear

    out_points: int

    def __init__(self: Self, n_points: int, out_points: int, width: int, key: PRNGKeyArray) -> None:
        self.out_points = out_points
        subkey1, subkey2 = random.split(key)
        in_c = math.comb(n_points, 2) + n_points
        self.first = eqx.nn.Linear(in_c, width, key=subkey1)
        self.second = eqx.nn.Linear(width, out_points * n_points, key=subkey2)

    def __call__(self: Self, vecs: Float[Array, "n_points D"]) -> Float[Array, "out_points D"]:
        n_points = vecs.shape[0]
        centered_vecs = vecs - jnp.mean(vecs, axis=0)
        dots = centered_vecs @ centered_vecs.T  # (n_points,n_points)
        dots = dots[jnp.triu_indices(n_points)].reshape(-1)  # (in_c,)

        coeffs = self.second(jax.nn.gelu(self.first(dots))).reshape(self.out_points, n_points)
        # Scalars are universal setup, coeffs have to add up to 1 for n_points
        coeffs = coeffs / (jnp.sum(coeffs, axis=1, keepdims=True) + 1e-5)  # (out_points,n_points)
        return coeffs @ vecs  # (out_points,D)


class VecsScalarsModel(eqx.Module):
    first: eqx.nn.Linear
    second: eqx.nn.Linear

    out_points: int

    def __init__(self: Self, n_points: int, out_points: int, width: int, key: PRNGKeyArray) -> None:
        self.out_points = out_points
        subkey1, subkey2 = random.split(key)
        in_c = math.comb(n_points, 2) + n_points + n_points
        self.first = eqx.nn.Linear(in_c, width, key=subkey1)
        self.second = eqx.nn.Linear(width, n_points * out_points, key=subkey2)

    def __call__(
        self: Self, vecs: Float[Array, "n_points D"], scalars: Float[Array, " n_points"]
    ) -> Float[Array, "out_c D"]:
        n_points = vecs.shape[0]
        centered_vecs = vecs - jnp.mean(vecs, axis=0)
        dots = centered_vecs @ centered_vecs.T  # (n_points,n_points)
        dots = dots[jnp.triu_indices(n_points)].reshape(-1)  # (n_points choose 2,)

        coeffs = self.second(jax.nn.gelu(self.first(jnp.concat([dots, scalars])))).reshape(
            self.out_points, n_points
        )
        # Scalars are universal setup, coeffs have to add up to 1 for n_points
        coeffs = coeffs / (jnp.sum(coeffs, axis=1, keepdims=True) + 1e-5)  # (out_points,n_points)
        return coeffs @ vecs  # (out_points,D)


D = 2
N = 9
n_points = 10

key = random.PRNGKey(seed=time.time_ns())
key, subkey = random.split(key)
vecs = random.normal(subkey, shape=(n_points, D))
scalars = random.normal(subkey, shape=(n_points,))

key, subkey = random.split(key)
vecs_scalars_model = VecsScalarsModel(n_points, 1, 32, subkey)

key, subkey1, subkey2 = random.split(key, num=3)
for translation, rotation in zip(
    random.normal(subkey1, shape=(5, D)), random.orthogonal(subkey2, D, shape=(5,))
):
    vecs_prime = jnp.einsum("ij,bj->bi", rotation, vecs) - translation[None]
    first = vecs_scalars_model(vecs_prime, scalars)[0]
    second = rotation @ vecs_scalars_model(vecs, scalars)[0] - translation
    assert jnp.allclose(first, second, rtol=1e-3, atol=1e-3), f"{jnp.max(jnp.abs(first - second))}"

testCenterOfMassEquivariance(D, N)
print("Passed Center of Mass equivariance checks.")

key, subkey = random.split(key)
testShiftQREquivariance(D, N, subkey)
print("Passed shiftQR equivariance checks.")

key, subkey = random.split(key)
testVecsOnlyEquivariance(D, N, subkey)
print("Passed VecsOnlyModel equivariance checks.")

key, subkey = random.split(key)
testVecsScalarsEquivariance(D, N, subkey)
print("Passed VecsScalarsModel equivariance checks.")

print("Passed all tests!")
