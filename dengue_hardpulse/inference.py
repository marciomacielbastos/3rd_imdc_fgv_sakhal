from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from .model import model
from .seasonal import build_neighbor_index
Array = Any
ModelFn = Callable[..., Any]
CITY_KEYS_1D = {'lat', 'lon', 'alt', 'koppen_idx', 'state_idx'}
CITY_KEYS_2D = {'ifdm', 'log_density', 'population', 'temp_lag', 'humid_lag', 'rain_lag', 'history_attack'}
CITY_KEYS_3D = {'obs_mask'}
UNCONSTRAINED_GLOBALS = frozenset({'omega_raw', 'rho_year_raw', 'beta_momentum_raw', 'beta_state_momentum_raw', 'xi_raw', 'tau_state_amp_bias_raw', 'mu_fast_nat', 'lsig_fast_nat', 'beta_log_pop_base', 'beta_log_density_base', 'log_rate_base_state_raw', 'state_profile_weight_raw'})
STATE_CITY_PREFIXES = ('mu_fast', 'lsig_fast')
CITY_ONLY_PREFIXES = ()
STATE_ONLY_PREFIXES = ()
HIER_PREFIXES_FULL = STATE_CITY_PREFIXES
HIER_PREFIXES_REDUCED = CITY_ONLY_PREFIXES
HIER_PREFIXES_ALL = HIER_PREFIXES_FULL + HIER_PREFIXES_REDUCED + STATE_ONLY_PREFIXES
CONSTRAINED_PARAMS = frozenset({f'tau_{prefix}_city' for prefix in STATE_CITY_PREFIXES + CITY_ONLY_PREFIXES} | {f'tau_{prefix}_spatial' for prefix in STATE_CITY_PREFIXES + CITY_ONLY_PREFIXES} | {f'tau_{prefix}_state' for prefix in HIER_PREFIXES_ALL} | {'sigma_year', 'sigma_amp', 'tau_log_rate_base_city', 'tau_log_rate_base_spatial'})
GLOBAL_FIX_SITES = frozenset({'omega_raw', 'mu_fast_nat', 'lsig_fast_nat'})
CITY_SHAPED_SITES = {f'u_{prefix}_city' for prefix in STATE_CITY_PREFIXES + CITY_ONLY_PREFIXES} | {'u_log_rate_base_city'}
REGION_SHAPED_SITES = {f'u_{prefix}_state' for prefix in HIER_PREFIXES_ALL} | {'log_rate_base_state_raw', 'state_amp_bias_raw', 'state_profile_weight_raw'}

def _rng(seed=0):
    return jax.random.PRNGKey(seed)

def _rank(v):
    from scipy.stats import rankdata
    v = np.asarray(v, dtype=float)
    return (rankdata(v) - 1.0) / max(v.size - 1, 1)

def _burden(data, obs):
    o = np.asarray(obs, dtype=float)
    if 'obs_mask' not in data:
        return o.sum(axis=(1, 2))
    m = np.asarray(data['obs_mask'], dtype=bool)
    return np.where(m, o, 0.0).sum(axis=(1, 2)) if m.shape == o.shape else o.sum(axis=(1, 2))

def _n(posterior):
    return int(jnp.asarray(next(iter(posterior.values()))).shape[0])

def _slice(posterior, idx, drop=None):
    drop = set() if drop is None else set(drop)
    return {k: jnp.asarray(v)[idx] for k, v in posterior.items() if k not in drop}

def run_svi(data, obs, n_steps=30000, lr=0.0001, lr_final=5e-06, warmup_steps=500, guide_type='normal', rank=20, print_every=1000, rng_key=None, init_params=None, model_fn=None, n_posterior_draws=200) -> dict:
    import optax
    rng_key = _rng(0) if rng_key is None else rng_key
    fitted_model = model if model_fn is None else model_fn
    guide = AutoLowRankMultivariateNormal(fitted_model, rank=rank) if guide_type == 'lowrank' else AutoNormal(fitted_model)
    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=lr, warmup_steps=warmup_steps, decay_steps=n_steps, end_value=lr_final)
    optimizer = numpyro.optim.optax_to_numpyro(optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=schedule)))
    svi = SVI(fitted_model, guide, optimizer, loss=Trace_ELBO())
    rng_key, ik = jax.random.split(rng_key)
    svi_state = svi.init(ik, data, obs=obs, init_params=init_params) if init_params is not None else svi.init(ik, data, obs=obs)

    @jax.jit
    def step(s):
        return svi.update(s, data, obs=obs)
    losses = []
    print(f'SVI: running {n_steps:,} steps …')
    for i in range(n_steps):
        svi_state, loss = step(svi_state)
        losses.append(float(loss))
        if print_every > 0 and (i + 1) % print_every == 0:
            elbo = -np.mean(losses[-print_every:])
            print(f'  step {i + 1:6d}/{n_steps}  ELBO={elbo:,.1f}')
    params = svi.get_params(svi_state)
    median_params = {k[:-len('_auto_loc')]: v for k, v in params.items() if k.endswith('_auto_loc')}
    rng_key, sk = jax.random.split(rng_key)
    collected: dict = {}
    for dk in jax.random.split(sk, int(n_posterior_draws)):
        for name, val in guide.sample_posterior(dk, params, sample_shape=()).items():
            collected.setdefault(name, []).append(val)
    guide_samples = {k: jnp.stack(v) for k, v in collected.items()}
    print('SVI done.\n')
    return {'guide': guide, 'losses': np.asarray(losses), 'params': params, 'median_params': median_params, 'guide_samples': guide_samples}

def run_all_city_svi(data_train: dict, Y_train: jnp.ndarray, nuts2_samples: dict, n_steps: int=10000, lr: float=5e-05, lr_final: float=2e-06, warmup_steps: int=1000, print_every: int=1000, rng_key: Any=None, init_params: dict | None=None, n_posterior_draws: int=200) -> dict:

    def _pm(name):
        v = nuts2_samples.get(name)
        return float(jnp.median(jnp.asarray(v))) if v is not None else None
    globals_to_fix = {}
    for name in GLOBAL_FIX_SITES:
        val = _pm(name)
        if val is not None:
            globals_to_fix[name] = val
    print(f'  Fixed globals from NUTS 2: {sorted(globals_to_fix.keys())}')
    model_fixed = condition(model, globals_to_fix)
    svi_init = None
    if init_params is not None:
        fixed = set(globals_to_fix)
        compatible = filter_init_params_for_subset(init_params, data_train)
        svi_init = {k: v for k, v in compatible.items() if k not in fixed}
    rng_key = _rng(5) if rng_key is None else rng_key
    return run_svi(data_train, Y_train, model_fn=model_fixed, n_steps=n_steps, lr=lr, lr_final=lr_final, warmup_steps=warmup_steps, print_every=print_every, rng_key=rng_key, init_params=svi_init, n_posterior_draws=n_posterior_draws)

def build_subset_data(data, obs, city_ids, n_neighbours: int=10):
    idx = jnp.asarray(city_ids)
    original_state_idx = np.asarray(data.get('state_idx'))[np.asarray(city_ids)] if 'state_idx' in data else None
    subset = {}
    for k, v in data.items():
        if k in CITY_KEYS_1D or k in CITY_KEYS_2D or k in CITY_KEYS_3D:
            subset[k] = v[idx]
        elif k == 'h_bar' and isinstance(v, Mapping):
            subset[k] = {n: arr[idx] for n, arr in v.items()}
        elif k in ('nbr_idx', 'nbr_weights', 'state_week_profile', 'state_history_attack'):
            continue
        else:
            subset[k] = v
    subset['n_cities'] = len(city_ids)
    if 'state_idx' in subset:
        unique_states, inverse_state = np.unique(original_state_idx, return_inverse=True)
        subset['state_idx'] = jnp.array(inverse_state)
        if 'state_week_profile' in data:
            subset['state_week_profile'] = jnp.asarray(data['state_week_profile'])[jnp.array(unique_states)]
        if 'state_history_attack' in data:
            subset['state_history_attack'] = jnp.asarray(data['state_history_attack'])[jnp.array(unique_states)]
    if 'koppen_idx' in subset:
        _, subset['koppen_idx'] = jnp.unique(subset['koppen_idx'], return_inverse=True)
    subset['n_states'] = int(subset['state_idx'].max() + 1) if 'state_idx' in subset else 1
    subset['n_koppen'] = int(subset['koppen_idx'].max() + 1) if 'koppen_idx' in subset else 1
    lat = np.asarray(subset['lat'])
    lon = np.asarray(subset['lon'])
    k_eff = min(n_neighbours, max(len(city_ids) - 1, 1))
    nbr_idx, nbr_weights = build_neighbor_index(lat, lon, n_neighbours=k_eff)
    subset['nbr_idx'] = jnp.array(nbr_idx)
    subset['nbr_weights'] = jnp.array(nbr_weights)
    return (subset, obs[idx])

def filter_init_params_for_subset(init_params, data):
    C = int(data['n_cities'])
    Y = int(data.get('n_years', 1))
    n_state = int(data.get('n_states', 1))
    n_koppen = int(data.get('n_koppen', 1))
    shape_for = {}
    for site in CITY_SHAPED_SITES:
        shape_for[site] = (C,)
    state_shaped_raw = {'log_rate_base_state_raw', 'state_amp_bias_raw', 'state_profile_weight_raw'}
    for site in REGION_SHAPED_SITES:
        shape_for[site] = (n_state,) if site.endswith('_state') or site in state_shaped_raw else (n_koppen,)
    shape_for['eps_amp_raw'] = (Y, C)
    shape_for['year_noise'] = (Y,)
    drop_always = {'eps_amp', 'year_effect', 'mu_obs', 'Y', 'log_rate_base_state', 'log_rate_base_state_per_100k', 'base_rate_state_per_100k', 'base_rate_city_per_100k', 'state_profile_weight'}
    filtered, dropped = ({}, [])
    for name, value in init_params.items():
        if name in drop_always:
            dropped.append(f'{name}: deterministic or observation site')
            continue
        if name in CONSTRAINED_PARAMS:
            dropped.append(f'{name}: constrained prior')
            continue
        arr = jnp.asarray(value)
        if name == 'log_theta':
            arr = jnp.maximum(arr, -2.0)
        if name in shape_for and arr.shape != shape_for[name]:
            dropped.append(f'{name}: got {arr.shape}, expected {shape_for[name]}')
            continue
        filtered[name] = arr
    if dropped:
        print(f'  filter_init_params: dropped {len(dropped)} params:')
        for d in dropped:
            print(f'    {d}')
    return filtered

def select_important_cities(data, obs, n_cities=50, w_burden=0.35, w_pop=0.25, w_incidence=0.25, w_geo=0.15, seed=0):
    del seed
    n_total = int(data['n_cities'])
    n_sel = min(n_cities, n_total)
    weights = np.array([w_burden, w_pop, w_incidence, w_geo], dtype=float)
    weights /= weights.sum()
    burden = _burden(data, obs)
    pop_mean = np.asarray(data['population'], dtype=float).mean(axis=1)
    incidence = burden / np.maximum(pop_mean, 1.0)
    state_idx = np.asarray(data.get('state_idx', np.zeros(n_total, int)))
    st_counts = np.bincount(state_idx, minlength=int(data.get('n_states', 1)))
    rarity = 1.0 / np.maximum(st_counts[state_idx], 1)
    score = weights[0] * _rank(burden) + weights[1] * _rank(pop_mean) + weights[2] * _rank(incidence) + weights[3] * _rank(rarity)
    selected = np.argsort(score)[::-1][:n_sel].tolist()
    geocodes = np.asarray(data.get('geocode_order', np.arange(n_total)), dtype=np.int64)
    df_candidates = [c for c in range(n_total) if int(geocodes[c]) // 100000 == 53]
    if df_candidates and n_sel > 0:
        df_city = max(df_candidates, key=lambda c: burden[c])
        if df_city not in selected:
            selected_set = set(selected)
            replaceable = [c for c in selected if c not in df_candidates]
            if replaceable:
                worst = min(replaceable, key=lambda c: score[c])
                selected.remove(worst)
            elif selected:
                selected.pop()
            selected.append(df_city)
            selected = list(dict.fromkeys(selected))[:n_sel]
            if len(selected) < n_sel:
                for c in np.argsort(score)[::-1]:
                    c = int(c)
                    if c not in selected_set and c not in selected:
                        selected.append(c)
                    if len(selected) >= n_sel:
                        break
            print(f'  Forced DF/Brasilia into NUTS subset: idx={df_city}, geocode={int(geocodes[df_city])}')
    covered = set(state_idx[selected].tolist())
    n_states = int(data.get('n_states', 1))
    for s in range(n_states):
        if s in covered:
            continue
        cands = [c for c in range(n_total) if state_idx[c] == s and c not in selected]
        if not cands:
            continue
        best = max(cands, key=lambda c: score[c])
        st_cnt = defaultdict(int)
        for c in selected:
            st_cnt[int(state_idx[c])] += 1
        replaceable = [c for c in selected if st_cnt[int(state_idx[c])] > 1]
        if not replaceable:
            continue
        worst = min(replaceable, key=lambda c: score[c])
        selected.remove(worst)
        selected.append(best)
        covered.add(s)
    selected = sorted(selected)
    geocodes = data.get('geocode_order', list(range(n_total)))
    n_st = len(set(state_idx[selected].tolist()))
    print(f'\n  Selected {len(selected)} cities  |  states: {n_st}/{n_states}')
    print(f"  {'Rank':<5}{'idx':<8}{'Geocode':<12}{'Burden':>10}{'Pop':>12}{'Incidence':>10}{'Score':>8}")
    print('  ' + '-' * 56)
    for rank, c in enumerate(sorted(selected, key=lambda c: -score[c]), 1):
        print(f'  {rank:<5}{c:<8}{str(geocodes[c]):<12}{burden[c]:>10,.0f}{pop_mean[c]:>12,.0f}{incidence[c]:>10.4f}{score[c]:>8.4f}')
    return selected

def select_representative_cities(data, obs, n_cities=80, seed=0):
    rng = np.random.default_rng(seed)
    n_total = int(data['n_cities'])
    n_sel = min(n_cities, n_total)
    log_pop = np.log(np.asarray(data['population'][:, 0]) + 1.0)
    pop_q = np.digitize(log_pop, np.percentile(log_pop, [25, 50, 75]))
    burden = _burden(data, obs)
    burd_q = np.digitize(burden, np.percentile(burden, [33, 66]))
    st = np.asarray(data.get('state_idx', np.zeros(n_total, int)))
    kp = np.asarray(data.get('koppen_idx', np.zeros(n_total, int)))
    buckets: dict = defaultdict(list)
    for c in range(n_total):
        buckets[int(st[c]), int(kp[c]), int(pop_q[c]), int(burd_q[c])].append(c)
    selected: list = []
    for cities in buckets.values():
        n = max(1, round(len(cities) / n_total * n_sel))
        selected.extend(rng.choice(cities, size=min(n, len(cities)), replace=False).tolist())
    rng.shuffle(selected)
    selected = selected[:n_sel] if len(selected) > n_sel else selected
    if len(selected) < n_sel:
        rest = [c for c in range(n_total) if c not in set(selected)]
        selected.extend(rng.choice(rest, size=n_sel - len(selected), replace=False).tolist())
    return sorted(selected)

def run_nuts(data, obs, init_params=None, n_warmup=500, n_samples=500, n_chains=1, target_accept=0.9, rng_key=None, dense_mass=False, model_fn=None) -> dict:
    rng_key = _rng(1) if rng_key is None else rng_key
    fitted_model = model if model_fn is None else model_fn
    if init_params is not None:
        init_params = filter_init_params_for_subset(init_params, data)
    strategy = numpyro.infer.init_to_value(values=init_params) if init_params is not None else numpyro.infer.init_to_median()
    nuts = NUTS(fitted_model, target_accept_prob=target_accept, dense_mass=dense_mass, init_strategy=strategy)
    mcmc = MCMC(nuts, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains, progress_bar=True)
    print(f'NUTS: {n_warmup} warmup + {n_samples} samples …')
    mcmc.run(rng_key, data, obs=obs, extra_fields=('accept_prob', 'diverging'))
    mcmc.print_summary(exclude_deterministic=False)
    return {'mcmc': mcmc, 'samples': mcmc.get_samples(), 'extra': mcmc.get_extra_fields()}

def posterior_predictive(data, posterior, n_draws=500, rng_key=None, model_fn=None):
    rng_key = _rng(2) if rng_key is None else rng_key
    fitted_model = model if model_fn is None else model_fn
    nd = min(n_draws, _n(posterior))
    pred = Predictive(fitted_model, posterior_samples=posterior, num_samples=nd)
    return pred(rng_key, data, obs=None)['Y']

def forecast_all_cities(data_test: dict, all_city_svi: dict, nuts2_samples: dict | None=None, fixed_globals: dict | None=None, n_draws: int=200, batch_size: int=500, rng_key: Any=None) -> np.ndarray:
    del nuts2_samples
    rng_key = _rng(8) if rng_key is None else rng_key
    C_full = int(data_test['n_cities'])
    Y_test = int(data_test['n_years'])
    T = int(data_test['weeks_per_year'])
    svi_samples = all_city_svi['guide_samples']
    n_avail = _n(svi_samples)
    nd = min(n_draws, n_avail)
    rng_key, ik = jax.random.split(rng_key)
    draw_idx = jax.random.randint(ik, (nd,), 0, n_avail)
    fixed_globals = {} if fixed_globals is None else dict(fixed_globals)
    all_preds = np.zeros((nd, C_full, Y_test, T), dtype=np.float32)
    all_mu = np.zeros((nd, C_full, Y_test, T), dtype=np.float32)
    n_batches = int(np.ceil(C_full / batch_size))
    print(f'  Forecasting {C_full:,} cities in {n_batches} batches ({batch_size} cities/batch) …')
    state_idx_full = np.asarray(data_test['state_idx'])
    koppen_idx_full = np.asarray(data_test['koppen_idx'])
    for b in range(n_batches):
        c_start = b * batch_size
        c_end = min(c_start + batch_size, C_full)
        city_batch = list(range(c_start, c_end))
        sub_test, _ = build_subset_data(data_test, jnp.zeros((C_full, Y_test, T)), city_batch)
        sub_test['state_idx'] = jnp.array(state_idx_full[city_batch])
        sub_test['koppen_idx'] = jnp.array(koppen_idx_full[city_batch])
        if 'state_week_profile' in data_test:
            sub_test['state_week_profile'] = jnp.asarray(data_test['state_week_profile'])
        if 'state_history_attack' in data_test:
            sub_test['state_history_attack'] = jnp.asarray(data_test['state_history_attack'])
        sub_test['n_states'] = int(data_test['n_states'])
        sub_test['n_koppen'] = int(data_test['n_koppen'])
        batch_post = {}
        for k, v in svi_samples.items():
            if k in {'Y', 'mu_obs', 'phi_mean', 'rho_mean', 'period_slow_years', 'state_profile_weight', 'state_momentum', 'log_rate_base_state', 'log_rate_base_state_per_100k', 'state_amp_bias', 'base_rate_state_per_100k', 'base_rate_city_per_100k', 'state_profile_weight'}:
                continue
            arr = jnp.asarray(v)[draw_idx]
            if k == 'year_effect':
                continue
            if k == 'year_noise' and arr.ndim >= 2 and (arr.shape[1] != Y_test):
                arr = jnp.repeat(arr[:, -1:], Y_test, axis=1)
            if k == 'eps_amp':
                continue
            if k == 'eps_amp_raw' and arr.ndim >= 3:
                if arr.shape[1] != Y_test:
                    arr = jnp.repeat(arr[:, -1:, :], Y_test, axis=1)
                if arr.shape[2] == C_full:
                    batch_post[k] = arr[:, :, c_start:c_end]
                continue
            if k in CITY_SHAPED_SITES and arr.ndim >= 2 and (arr.shape[1] == C_full):
                batch_post[k] = arr[:, c_start:c_end]
            elif k not in CITY_SHAPED_SITES:
                batch_post[k] = arr
        for k, val in fixed_globals.items():
            batch_post[k] = jnp.full((nd,), val)
        rng_key, pk = jax.random.split(rng_key)
        pred = Predictive(model, posterior_samples=batch_post)
        pout = pred(pk, sub_test, obs=None)
        all_preds[:, c_start:c_end, :, :] = np.array(pout['Y'])
        all_mu[:, c_start:c_end, :, :] = np.array(pout['mu_obs'])
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f'    batch {b + 1}/{n_batches} done ({c_end:,}/{C_full:,} cities)')
    return (all_preds, all_mu)

def predict_training_mu_all_cities(data_train: dict, Y_train: Array, all_city_svi: dict, fixed_globals: dict | None=None, n_draws: int=200, batch_size: int=500, rng_key: Any=None) -> np.ndarray:
    rng_key = _rng(9) if rng_key is None else rng_key
    C_full = int(data_train['n_cities'])
    Y = int(data_train['n_years'])
    T = int(data_train['weeks_per_year'])
    svi_samples = all_city_svi['guide_samples']
    n_avail = _n(svi_samples)
    nd = min(n_draws, n_avail)
    rng_key, ik = jax.random.split(rng_key)
    draw_idx = jax.random.randint(ik, (nd,), 0, n_avail)
    fixed_globals = {} if fixed_globals is None else dict(fixed_globals)
    all_mu = np.zeros((nd, C_full, Y, T), dtype=np.float32)
    n_batches = int(np.ceil(C_full / batch_size))
    print(f'  Predicting fitted training means in {n_batches} batches ({batch_size} cities/batch) ...')
    state_idx_full = np.asarray(data_train['state_idx'])
    koppen_idx_full = np.asarray(data_train['koppen_idx'])
    obs_np = np.asarray(Y_train)
    for b in range(n_batches):
        c_start = b * batch_size
        c_end = min(c_start + batch_size, C_full)
        city_batch = list(range(c_start, c_end))
        sub_train, sub_obs = build_subset_data(data_train, jnp.asarray(obs_np), city_batch)
        sub_train['state_idx'] = jnp.array(state_idx_full[city_batch])
        sub_train['koppen_idx'] = jnp.array(koppen_idx_full[city_batch])
        if 'state_week_profile' in data_train:
            sub_train['state_week_profile'] = jnp.asarray(data_train['state_week_profile'])
        if 'state_history_attack' in data_train:
            sub_train['state_history_attack'] = jnp.asarray(data_train['state_history_attack'])
        sub_train['n_states'] = int(data_train['n_states'])
        sub_train['n_koppen'] = int(data_train['n_koppen'])
        batch_post = {}
        for k, v in svi_samples.items():
            if k in {'Y', 'mu_obs', 'phi_mean', 'rho_mean', 'period_slow_years', 'state_momentum', 'log_rate_base_state', 'log_rate_base_state_per_100k', 'state_amp_bias', 'base_rate_state_per_100k', 'base_rate_city_per_100k', 'state_profile_weight'}:
                continue
            arr = jnp.asarray(v)[draw_idx]
            if k == 'year_effect':
                continue
            if k == 'eps_amp':
                continue
            if k == 'eps_amp_raw' and arr.ndim >= 3:
                if arr.shape[1] == Y and arr.shape[2] == C_full:
                    batch_post[k] = arr[:, :, c_start:c_end]
                continue
            if k in CITY_SHAPED_SITES and arr.ndim >= 2 and (arr.shape[1] == C_full):
                batch_post[k] = arr[:, c_start:c_end]
            elif k not in CITY_SHAPED_SITES:
                batch_post[k] = arr
        for k, val in fixed_globals.items():
            batch_post[k] = jnp.full((nd,), val)
        rng_key, pk = jax.random.split(rng_key)
        pred = Predictive(model, posterior_samples=batch_post)
        pout = pred(pk, sub_train, obs=sub_obs)
        all_mu[:, c_start:c_end, :, :] = np.array(pout['mu_obs'])
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f'    training batch {b + 1}/{n_batches} done ({c_end:,}/{C_full:,} cities)')
    return all_mu

def forecast_ensemble(data_test, posterior, n_draws=200, rng_key=None, model_fn=None):
    rng_key = _rng(3) if rng_key is None else rng_key
    fitted_model = model if model_fn is None else model_fn
    n_avail = _n(posterior)
    nd = min(n_draws, n_avail)
    rng_key, ik, pk = jax.random.split(rng_key, 3)
    idx = jax.random.randint(ik, (nd,), 0, n_avail)
    post = _slice(posterior, idx, drop=set())
    pred = Predictive(fitted_model, posterior_samples=post)
    pout = pred(pk, data_test, obs=None)
    return (pout['Y'], pout['mu_obs'])

def check_convergence(nuts_result, params=None):
    try:
        import arviz as az
    except ImportError:
        print('arviz not installed — pip install arviz')
        return
    idata = az.from_numpyro(nuts_result['mcmc'])
    print(az.summary(idata, var_names=params).to_string())
    div = int(nuts_result['extra'].get('diverging', jnp.array(0)).sum())
    print(f'\nDivergences: {div}')
    if div > 10:
        print('  WARNING: increase target_accept_prob or reparameterise.')
