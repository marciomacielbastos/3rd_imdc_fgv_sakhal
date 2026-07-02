"""
Direct hard-pulse dengue observation model.

This package is a simplified sibling of dengue_pulse.  Instead of sampling a
vector-competence surface and pushing it through latent transmission dynamics,
it models the expected reported cases directly as an annual wrapped-Gaussian
pulse.  Year-to-year amplitude is controlled by a city baseline, a national
year effect, and an empirical immunity memory updated from previous observed
burden.

The intent is pragmatic forecasting: fewer weakly identified parameters, no
separate phi/rho exposure-reporting pair, and no theta -> 0 escape hatch.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .seasonal import FAST_PERIOD, fourier_wrapped_gaussian, _spatial_smooth_factor

EPS = 1e-6
THETA_FIXED = 5.0


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _bounded_unit(name: str, center: float, scale: float, low: float, high: float):
    raw = numpyro.sample(name, dist.Normal(_logit((center - low) / (high - low)), scale))
    value = low + (high - low) * jax.nn.sigmoid(raw)
    numpyro.deterministic(name.removesuffix("_raw"), value)
    return value


def _sample_state_city_param(
    prefix: str,
    data: dict,
    nat_loc: float,
    nat_scale: float,
    state_scale_prior: float,
    city_scale_prior: float,
    spatial_scale_prior: float,
) -> jnp.ndarray:
    """National -> state -> city parameter with nearest-neighbor smoothing."""
    C = int(data["n_cities"])
    n_state = int(data["n_states"])
    state_idx = data["state_idx"]

    nat = numpyro.sample(f"{prefix}_nat", dist.Normal(nat_loc, nat_scale))
    tau_state = numpyro.sample(f"tau_{prefix}_state", dist.HalfNormal(state_scale_prior))
    with numpyro.plate(f"state_{prefix}", n_state):
        u_state = numpyro.sample(f"u_{prefix}_state", dist.Normal(0.0, tau_state))

    tau_city = numpyro.sample(f"tau_{prefix}_city", dist.HalfNormal(city_scale_prior))
    with numpyro.plate(f"city_{prefix}", C):
        u_city = numpyro.sample(f"u_{prefix}_city", dist.Normal(0.0, tau_city))

    tau_spatial = numpyro.sample(f"tau_{prefix}_spatial", dist.HalfNormal(spatial_scale_prior))
    _spatial_smooth_factor(
        f"{prefix}_spatial_smooth",
        u_city,
        data["nbr_idx"],
        data["nbr_weights"],
        tau_spatial,
    )

    value = nat + u_state[state_idx] + u_city
    numpyro.deterministic(f"{prefix}_city", value)
    return value


def _sample_city_param(
    prefix: str,
    data: dict,
    nat_loc: float,
    nat_scale: float,
    city_scale_prior: float,
    spatial_scale_prior: float,
) -> jnp.ndarray:
    """National -> city parameter with nearest-neighbor smoothing."""
    C = int(data["n_cities"])
    nat = numpyro.sample(f"{prefix}_nat", dist.Normal(nat_loc, nat_scale))
    tau_city = numpyro.sample(f"tau_{prefix}_city", dist.HalfNormal(city_scale_prior))
    with numpyro.plate(f"city_{prefix}", C):
        u_city = numpyro.sample(f"u_{prefix}_city", dist.Normal(0.0, tau_city))
    tau_spatial = numpyro.sample(f"tau_{prefix}_spatial", dist.HalfNormal(spatial_scale_prior))
    _spatial_smooth_factor(
        f"{prefix}_spatial_smooth",
        u_city,
        data["nbr_idx"],
        data["nbr_weights"],
        tau_spatial,
    )
    value = nat + u_city
    numpyro.deterministic(f"{prefix}_city", value)
    return value


def _sample_state_param(
    prefix: str,
    data: dict,
    nat_loc: float,
    nat_scale: float,
    state_scale_prior: float,
) -> jnp.ndarray:
    """National -> state parameter, then expanded to cities."""
    n_state = int(data["n_states"])
    state_idx = data["state_idx"]
    nat = numpyro.sample(f"{prefix}_nat", dist.Normal(nat_loc, nat_scale))
    tau_state = numpyro.sample(f"tau_{prefix}_state", dist.HalfNormal(state_scale_prior))
    with numpyro.plate(f"state_{prefix}", n_state):
        u_state = numpyro.sample(f"u_{prefix}_state", dist.Normal(0.0, tau_state))
    state_value = nat + u_state
    numpyro.deterministic(f"{prefix}_state", state_value)
    city_value = state_value[state_idx]
    numpyro.deterministic(f"{prefix}_city", city_value)
    return city_value


def _annual_burden_features(data: dict, obs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return year-start immunity and previous-year outbreak momentum.

    Both features are deterministic functions of observed annual incidence.
    Immunity is a decayed cumulative memory; momentum is only the immediately
    previous year's observed attack signal.  During pure forecasting with
    obs=None these features default to zero on the forecast grid.
    """
    C = int(data["n_cities"])
    Y = int(data["n_years"])
    T = int(data["weeks_per_year"])
    mask = jnp.asarray(data.get("obs_mask", jnp.ones((C, Y, T)))).astype(bool)
    y = jnp.where(mask, jnp.asarray(obs), 0.0) if obs is not None else jnp.zeros_like(mask, dtype=jnp.float32)
    pop = jnp.maximum(jnp.asarray(data["population"]).reshape(C, Y, T), 1.0)
    annual_cases = y.sum(axis=2)
    observed_weeks = jnp.maximum(mask.sum(axis=2), 1)
    annual_pop = jnp.maximum(jnp.where(mask, pop, 0.0).sum(axis=2) / observed_weeks, 1.0)
    attack = jnp.log1p(10_000.0 * annual_cases / annual_pop)

    omega = _bounded_unit("omega_raw", center=0.65, scale=0.7, low=0.30, high=0.95)

    memories = []
    momentums = []
    mem = jnp.zeros(C)
    prev_attack = jnp.zeros(C)
    for yi in range(Y):
        memories.append(mem)
        momentums.append(prev_attack)
        mem = omega * mem + attack[:, yi]
        prev_attack = attack[:, yi]
    return jnp.stack(memories, axis=1), jnp.stack(momentums, axis=1)


def _sample_city_pulse(data: dict) -> dict:
    mu_fast = _sample_state_city_param(
        "mu_fast", data, nat_loc=20.0, nat_scale=6.0,
        state_scale_prior=3.0, city_scale_prior=2.0, spatial_scale_prior=2.0,
    )
    lsig_fast = _sample_state_city_param(
        "lsig_fast", data, nat_loc=math.log(6.0), nat_scale=0.35,
        state_scale_prior=0.25, city_scale_prior=0.20, spatial_scale_prior=0.20,
    )
    log_rate_base = _sample_state_city_param(
        "log_rate_base", data, nat_loc=math.log(2e-6), nat_scale=0.50,
        state_scale_prior=0.35, city_scale_prior=0.45, spatial_scale_prior=0.40,
    )
    sigma_fast = jnp.clip(jax.nn.softplus(lsig_fast), 2.0, 18.0)
    return {
        "mu_fast": jnp.mod(mu_fast, FAST_PERIOD),
        "sigma_fast": sigma_fast,
        "log_rate_base": log_rate_base,
    }


def _sample_year_effect(Y: int) -> jnp.ndarray:
    rho_year = _bounded_unit("rho_year_raw", center=0.60, scale=0.75, low=-0.20, high=0.95)
    sigma_year = numpyro.sample("sigma_year", dist.HalfNormal(0.20))
    with numpyro.plate("year", Y):
        year_noise = numpyro.sample("year_noise", dist.Normal(0.0, 1.0))

    effects = []
    prev = jnp.asarray(0.0)
    for yi in range(Y):
        cur = rho_year * prev + sigma_year * year_noise[yi]
        effects.append(cur)
        prev = cur
    year_effect = jnp.stack(effects)
    numpyro.deterministic("year_effect", year_effect)
    return year_effect


def _expected_cases(data: dict, obs, pulse: dict) -> jnp.ndarray:
    C = int(data["n_cities"])
    Y = int(data["n_years"])
    T = int(data["weeks_per_year"])
    weeks = jnp.arange(T, dtype=jnp.float32)
    shape = fourier_wrapped_gaussian(
        weeks,
        pulse["mu_fast"],
        pulse["sigma_fast"],
        FAST_PERIOD,
        n_harmonics=20,
    )
    shape = jnp.clip(shape, 0.0, None)
    shape = shape / jnp.maximum(shape.mean(axis=1, keepdims=True), EPS)

    year_effect = _sample_year_effect(Y)

    xi_logit = _sample_state_param(
        "xi_logit", data, nat_loc=_logit(0.20), nat_scale=0.45, state_scale_prior=0.25
    )
    xi = jax.nn.sigmoid(xi_logit)
    numpyro.deterministic("xi", xi)

    beta_momentum = numpyro.sample("beta_momentum", dist.Normal(0.25, 0.20))
    sigma_amp = numpyro.sample("sigma_amp", dist.HalfNormal(0.20))
    with numpyro.plate("city_amp", C):
        with numpyro.plate("year_amp", Y):
            eps_amp_raw = numpyro.sample("eps_amp_raw", dist.Normal(0.0, 1.0))
    eps_amp = sigma_amp * eps_amp_raw.T
    numpyro.deterministic("eps_amp", eps_amp)

    immunity, momentum = _annual_burden_features(data, obs)
    pop = jnp.maximum(jnp.asarray(data["population"]).reshape(C, Y, T), 1.0)
    log_rate = (
        pulse["log_rate_base"][:, None]
        + year_effect[None, :]
        + eps_amp
        + beta_momentum * momentum
        - xi[:, None] * immunity
    )
    annual_mean = pop.mean(axis=2) * jnp.exp(log_rate)
    mu = annual_mean[:, :, None] * shape[:, None, :]
    return jnp.maximum(mu, EPS)


def model(data: dict, obs=None) -> None:
    C = int(data["n_cities"])
    Y = int(data["n_years"])
    T = int(data["weeks_per_year"])
    pulse = _sample_city_pulse(data)
    mu_obs = _expected_cases(data, obs, pulse)

    theta = jnp.asarray(THETA_FIXED, dtype=mu_obs.dtype)
    numpyro.deterministic("theta", theta)
    numpyro.deterministic("mu_obs", mu_obs)
    numpyro.deterministic("phi_mean", jnp.nan)
    numpyro.deterministic("rho_mean", jnp.nan)
    numpyro.deterministic("period_slow_years", jnp.nan)

    mask = jnp.asarray(data.get("obs_mask", jnp.ones((C, Y, T)))).astype(bool)
    numpyro.sample(
        "Y",
        dist.NegativeBinomial2(
            mean=mu_obs,
            concentration=jnp.broadcast_to(theta, mu_obs.shape),
        ).mask(mask).to_event(2),
        obs=obs,
    )
