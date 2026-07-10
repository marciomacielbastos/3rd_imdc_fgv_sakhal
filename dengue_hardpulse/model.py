from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from .seasonal import FAST_PERIOD, fourier_wrapped_gaussian, _spatial_smooth_factor
EPS = 1e-06
THETA_FIXED = 0.5
SEASON_START_WEEK = 41

def _logit(p: float) -> float:
    p = min(max(float(p), 1e-06), 1.0 - 1e-06)
    return math.log(p / (1.0 - p))

def _bounded_unit(name: str, center: float, scale: float, low: float, high: float):
    raw = numpyro.sample(name, dist.Normal(_logit((center - low) / (high - low)), scale))
    value = low + (high - low) * jax.nn.sigmoid(raw)
    numpyro.deterministic(name.removesuffix('_raw'), value)
    return value

def _bounded_log_rate_per_100k(name: str, center: float=200.0, low: float=1.0, high: float=3000.0, scale: float=0.75) -> jnp.ndarray:
    raw = numpyro.sample(f'{name}_raw', dist.Normal(_logit((center - low) / (high - low)), scale))
    rate_100k = low + (high - low) * jax.nn.sigmoid(raw)
    numpyro.deterministic(f'{name}_per_100k', rate_100k)
    return jnp.log(rate_100k / 100000.0)

def _sample_state_city_param(prefix: str, data: dict, nat_loc: float, nat_scale: float, state_scale_prior: float, city_scale_prior: float, spatial_scale_prior: float) -> jnp.ndarray:
    C = int(data['n_cities'])
    n_state = int(data['n_states'])
    state_idx = data['state_idx']
    nat = numpyro.sample(f'{prefix}_nat', dist.Normal(nat_loc, nat_scale))
    tau_state = numpyro.sample(f'tau_{prefix}_state', dist.HalfNormal(state_scale_prior))
    with numpyro.plate(f'state_{prefix}', n_state):
        u_state = numpyro.sample(f'u_{prefix}_state', dist.Normal(0.0, tau_state))
    tau_city = numpyro.sample(f'tau_{prefix}_city', dist.HalfNormal(city_scale_prior))
    with numpyro.plate(f'city_{prefix}', C):
        u_city = numpyro.sample(f'u_{prefix}_city', dist.Normal(0.0, tau_city))
    tau_spatial = numpyro.sample(f'tau_{prefix}_spatial', dist.HalfNormal(spatial_scale_prior))
    _spatial_smooth_factor(f'{prefix}_spatial_smooth', u_city, data['nbr_idx'], data['nbr_weights'], tau_spatial)
    value = nat + u_state[state_idx] + u_city
    numpyro.deterministic(f'{prefix}_city', value)
    return value

def _sample_city_param(prefix: str, data: dict, nat_loc: float, nat_scale: float, city_scale_prior: float, spatial_scale_prior: float) -> jnp.ndarray:
    C = int(data['n_cities'])
    nat = numpyro.sample(f'{prefix}_nat', dist.Normal(nat_loc, nat_scale))
    tau_city = numpyro.sample(f'tau_{prefix}_city', dist.HalfNormal(city_scale_prior))
    with numpyro.plate(f'city_{prefix}', C):
        u_city = numpyro.sample(f'u_{prefix}_city', dist.Normal(0.0, tau_city))
    tau_spatial = numpyro.sample(f'tau_{prefix}_spatial', dist.HalfNormal(spatial_scale_prior))
    _spatial_smooth_factor(f'{prefix}_spatial_smooth', u_city, data['nbr_idx'], data['nbr_weights'], tau_spatial)
    value = nat + u_city
    numpyro.deterministic(f'{prefix}_city', value)
    return value

def _sample_state_param(prefix: str, data: dict, nat_loc: float, nat_scale: float, state_scale_prior: float) -> jnp.ndarray:
    n_state = int(data['n_states'])
    state_idx = data['state_idx']
    nat = numpyro.sample(f'{prefix}_nat', dist.Normal(nat_loc, nat_scale))
    tau_state = numpyro.sample(f'tau_{prefix}_state', dist.HalfNormal(state_scale_prior))
    with numpyro.plate(f'state_{prefix}', n_state):
        u_state = numpyro.sample(f'u_{prefix}_state', dist.Normal(0.0, tau_state))
    state_value = nat + u_state
    numpyro.deterministic(f'{prefix}_state', state_value)
    city_value = state_value[state_idx]
    numpyro.deterministic(f'{prefix}_city', city_value)
    return city_value

def _standardize_city_vector(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    return (x - x.mean()) / jnp.maximum(x.std(), 1e-06)

def _sample_log_rate_base(data: dict) -> jnp.ndarray:
    C = int(data['n_cities'])
    Y = int(data['n_years'])
    T = int(data['weeks_per_year'])
    n_state = int(data['n_states'])
    state_idx = data['state_idx']
    with numpyro.plate('state_log_rate_base', n_state):
        log_rate_base_state = _bounded_log_rate_per_100k('log_rate_base_state', center=200.0, high=3000.0, scale=0.75)
    numpyro.deterministic('log_rate_base_state', log_rate_base_state)
    tau_city = numpyro.sample('tau_log_rate_base_city', dist.HalfNormal(1.0))
    with numpyro.plate('city_log_rate_base', C):
        u_city = numpyro.sample('u_log_rate_base_city', dist.Normal(0.0, tau_city))
    tau_spatial = numpyro.sample('tau_log_rate_base_spatial', dist.HalfNormal(1.0))
    _spatial_smooth_factor('log_rate_base_spatial_smooth', u_city, data['nbr_idx'], data['nbr_weights'], tau_spatial)
    pop = jnp.maximum(jnp.asarray(data['population']).reshape(C, Y, T).mean(axis=(1, 2)), 1.0)
    log_pop_z = _standardize_city_vector(jnp.log(pop))
    density = jnp.asarray(data['log_density']).reshape(C, Y, T).mean(axis=(1, 2))
    log_density_z = _standardize_city_vector(density)
    beta_log_pop = numpyro.sample('beta_log_pop_base', dist.Normal(0.0, 0.2))
    beta_log_density = numpyro.sample('beta_log_density_base', dist.Normal(0.0, 0.2))
    value = log_rate_base_state[state_idx] + u_city + beta_log_pop * log_pop_z + beta_log_density * log_density_z
    numpyro.deterministic('log_rate_base_city', value)
    numpyro.deterministic('base_rate_state_per_100k', 100000.0 * jnp.exp(log_rate_base_state))
    numpyro.deterministic('base_rate_city_per_100k', 100000.0 * jnp.exp(value))
    return value

def _annual_burden_features(data: dict, obs) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    C = int(data['n_cities'])
    Y = int(data['n_years'])
    T = int(data['weeks_per_year'])
    n_state = int(data.get('n_states', 1))
    state_idx = jnp.asarray(data.get('state_idx', jnp.zeros(C, dtype=jnp.int32))).astype(jnp.int32)
    mask = jnp.asarray(data.get('obs_mask', jnp.ones((C, Y, T)))).astype(bool)
    y = jnp.where(mask, jnp.asarray(obs), 0.0) if obs is not None else jnp.zeros_like(mask, dtype=jnp.float32)
    pop = jnp.maximum(jnp.asarray(data['population']).reshape(C, Y, T), 1.0)
    annual_cases = y.sum(axis=2)
    observed_weeks = jnp.maximum(mask.sum(axis=2), 1)
    annual_pop = jnp.maximum(jnp.where(mask, pop, 0.0).sum(axis=2) / observed_weeks, 1.0)
    attack = jnp.log1p(10000.0 * annual_cases / annual_pop)
    state_cases = jnp.zeros((n_state, Y), dtype=y.dtype).at[state_idx].add(annual_cases)
    state_pop_sum = jnp.zeros((n_state, Y), dtype=pop.dtype).at[state_idx].add(annual_pop)
    state_pop = jnp.maximum(state_pop_sum, 1.0)
    state_attack = jnp.log1p(10000.0 * state_cases / state_pop)
    omega = _bounded_unit('omega_raw', center=0.65, scale=0.7, low=0.3, high=0.95)
    mem = jnp.zeros(C)
    prev_attack = jnp.zeros(C)
    history_attack = data.get('history_attack')
    if history_attack is not None:
        hist = jnp.asarray(history_attack)
        for hi in range(int(hist.shape[1])):
            mem = omega * mem + hist[:, hi]
            prev_attack = hist[:, hi]
    prev_state_attack = jnp.zeros(n_state)
    state_history_attack = data.get('state_history_attack')
    if state_history_attack is not None:
        state_hist = jnp.asarray(state_history_attack)
        for hi in range(int(state_hist.shape[1])):
            prev_state_attack = state_hist[:, hi]
    memories = []
    momentums = []
    state_momentums = []
    for yi in range(Y):
        memories.append(mem)
        momentums.append(prev_attack)
        state_momentums.append(prev_state_attack[state_idx])
        mem = omega * mem + attack[:, yi]
        prev_attack = attack[:, yi]
        prev_state_attack = state_attack[:, yi]
    return (jnp.stack(memories, axis=1), jnp.stack(momentums, axis=1), jnp.stack(state_momentums, axis=1))

def _sample_city_pulse(data: dict) -> dict:
    mu_fast = _sample_state_city_param('mu_fast', data, nat_loc=20.0, nat_scale=6.0, state_scale_prior=3.0, city_scale_prior=2.0, spatial_scale_prior=2.0)
    lsig_fast = _sample_state_city_param('lsig_fast', data, nat_loc=math.log(6.0), nat_scale=0.35, state_scale_prior=0.25, city_scale_prior=0.2, spatial_scale_prior=0.2)
    log_rate_base = _sample_log_rate_base(data)
    sigma_fast = jnp.clip(jax.nn.softplus(lsig_fast), 5.0, 18.0)
    return {'mu_fast': jnp.mod(mu_fast, FAST_PERIOD), 'sigma_fast': sigma_fast, 'log_rate_base': log_rate_base}

def _sample_year_effect(Y: int, data: dict) -> jnp.ndarray:
    rho_year = _bounded_unit('rho_year_raw', center=0.35, scale=0.55, low=-0.1, high=0.8)
    sigma_year = numpyro.sample('sigma_year', dist.HalfNormal(0.1))
    with numpyro.plate('year', Y):
        year_noise = numpyro.sample('year_noise', dist.Normal(0.0, 1.0))
    effects = []
    prev = jnp.asarray(data.get('year_effect_prev', 0.0))
    for yi in range(Y):
        cur = rho_year * prev + sigma_year * year_noise[yi]
        effects.append(cur)
        prev = cur
    year_effect = jnp.stack(effects)
    if Y > 1 and 'year_effect_prev' not in data:
        year_effect = year_effect - year_effect.mean()
    numpyro.deterministic('year_effect', year_effect)
    return year_effect

def _expected_cases(data: dict, obs, pulse: dict) -> jnp.ndarray:
    C = int(data['n_cities'])
    Y = int(data['n_years'])
    T = int(data['weeks_per_year'])
    weeks = jnp.arange(T, dtype=jnp.float32)
    shape = fourier_wrapped_gaussian(weeks, pulse['mu_fast'], pulse['sigma_fast'], FAST_PERIOD, n_harmonics=20)
    shape = jnp.clip(shape, 0.0, None)
    shape = shape / jnp.maximum(shape.sum(axis=1, keepdims=True), EPS)
    state_profile = data.get('state_week_profile')
    if state_profile is not None:
        state_shape = jnp.asarray(state_profile)[data['state_idx']]
        state_shape = jnp.clip(state_shape, 0.0, None)
        state_shape = state_shape / jnp.maximum(state_shape.sum(axis=1, keepdims=True), EPS)
        n_state = int(data['n_states'])
        with numpyro.plate('state_profile_weight_plate', n_state):
            state_profile_weight_raw = numpyro.sample('state_profile_weight_raw', dist.Normal(_logit((0.9 - 0.6) / (0.99 - 0.6)), 0.45))
        state_weight_state = 0.6 + 0.39 * jax.nn.sigmoid(state_profile_weight_raw)
        numpyro.deterministic('state_profile_weight', state_weight_state)
        state_weight = state_weight_state[data['state_idx']][:, None]
        shape = (1.0 - state_weight) * shape + state_weight * state_shape
        shape = shape / jnp.maximum(shape.sum(axis=1, keepdims=True), EPS)
    year_effect = _sample_year_effect(Y, data)
    xi = _bounded_unit('xi_raw', center=0.45, scale=0.65, low=0.1, high=0.8)
    tau_state_amp_bias = _bounded_unit('tau_state_amp_bias_raw', center=0.2, scale=0.5, low=0.02, high=0.5)
    with numpyro.plate('state_amp_bias_plate', int(data['n_states'])):
        state_amp_bias_raw = numpyro.sample('state_amp_bias_raw', dist.Normal(0.0, 1.0))
    state_amp_bias = tau_state_amp_bias * state_amp_bias_raw
    state_amp_bias = state_amp_bias - state_amp_bias.mean()
    numpyro.deterministic('state_amp_bias', state_amp_bias)
    beta_momentum = _bounded_unit('beta_momentum_raw', center=0.03, scale=0.55, low=-0.05, high=0.12)
    beta_state_momentum = _bounded_unit('beta_state_momentum_raw', center=0.05, scale=0.55, low=-0.05, high=0.15)
    sigma_amp = numpyro.sample('sigma_amp', dist.HalfNormal(0.2))
    with numpyro.plate('city_amp', C):
        with numpyro.plate('year_amp', Y):
            eps_amp_raw = numpyro.sample('eps_amp_raw', dist.Normal(0.0, 1.0))
    eps_amp = sigma_amp * eps_amp_raw.T
    eps_amp = eps_amp - eps_amp.mean(axis=1, keepdims=True)
    numpyro.deterministic('eps_amp', eps_amp)
    immunity, momentum, state_momentum = _annual_burden_features(data, obs)
    numpyro.deterministic('state_momentum', state_momentum)
    pop = jnp.maximum(jnp.asarray(data['population']).reshape(C, Y, T), 1.0)
    log_rate = pulse['log_rate_base'][:, None] + year_effect[None, :] + state_amp_bias[data['state_idx']][:, None] + eps_amp + beta_momentum * momentum + beta_state_momentum * state_momentum - xi * immunity
    annual_mean = pop.mean(axis=2) * jnp.exp(log_rate)
    mu = annual_mean[:, :, None] * shape[:, None, :]
    return jnp.maximum(mu, EPS)

def model(data: dict, obs=None) -> None:
    C = int(data['n_cities'])
    Y = int(data['n_years'])
    T = int(data['weeks_per_year'])
    pulse = _sample_city_pulse(data)
    mu_obs = _expected_cases(data, obs, pulse)
    theta = jnp.asarray(THETA_FIXED, dtype=mu_obs.dtype)
    numpyro.deterministic('theta', theta)
    numpyro.deterministic('mu_obs', mu_obs)
    numpyro.deterministic('phi_mean', jnp.nan)
    numpyro.deterministic('rho_mean', jnp.nan)
    numpyro.deterministic('period_slow_years', jnp.nan)
    mask = jnp.asarray(data.get('obs_mask', jnp.ones((C, Y, T)))).astype(bool)
    numpyro.sample('Y', dist.NegativeBinomial2(mean=mu_obs, concentration=jnp.broadcast_to(theta, mu_obs.shape)).mask(mask).to_event(2), obs=obs)
