from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
FAST_PERIOD = 52.0

def fourier_wrapped_gaussian(t, mu, sigma, period, n_harmonics=20):
    t = jnp.asarray(t)
    mu = jnp.asarray(mu)[..., None]
    sigma = jnp.asarray(sigma)[..., None]
    k = jnp.arange(1, n_harmonics + 1, dtype=t.dtype)
    angle = 2.0 * jnp.pi * k * (t - mu) / period
    damp = jnp.exp(-0.5 * (2.0 * jnp.pi * k * sigma / period) ** 2)
    return (1.0 + 2.0 * jnp.sum(damp * jnp.cos(angle), axis=-1)) / period

def build_neighbor_index(lat: np.ndarray, lon: np.ndarray, n_neighbours: int=10):
    coords = np.column_stack([lat, lon]).astype(float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    k = min(n_neighbours, max(coords.shape[0] - 1, 1))
    nbr_idx = np.argsort(dist2, axis=1)[:, :k]
    d = np.take_along_axis(np.sqrt(dist2), nbr_idx, axis=1)
    w = 1.0 / np.maximum(d, 1e-06)
    w = w / w.sum(axis=1, keepdims=True)
    return (nbr_idx.astype(np.int32), w.astype(np.float32))

def _spatial_smooth_factor(name, u_city, nbr_idx, nbr_weights, tau):
    nbr_mean = jnp.sum(jnp.asarray(nbr_weights) * u_city[jnp.asarray(nbr_idx)], axis=1)
    return numpyro.factor(name, dist.Normal(nbr_mean, tau).log_prob(u_city).sum())
