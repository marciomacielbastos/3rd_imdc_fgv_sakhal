import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
jax.config.update('jax_enable_x64', False)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dengue_hardpulse.model import THETA_FIXED
from dengue_hardpulse.seasonal import build_neighbor_index
from dengue_hardpulse.inference import run_svi, run_nuts, run_all_city_svi, build_subset_data, select_important_cities, posterior_predictive, forecast_all_cities, predict_training_mu_all_cities, check_convergence, GLOBAL_FIX_SITES, CONSTRAINED_PARAMS
from dengue_hardpulse.plot_forecast import plot_forecast_vs_observed, plot_coverage_by_week, print_skill_metrics, evaluate_all_cities, print_state_summary, plot_top_cities
SEP = '=' * 70
N_NEIGHBOURS = 10
SEASON_START_WEEK = 41
OPTIONAL_DENGUE_CITY_GEOCODES = [2931350, 2933307, 2302503, 3119401, 3549805, 3541406, 1200401, 1200203, 1716109, 4113700, 4103701, 4104808, 5201405, 5102637, 5215231]

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('outputs/preprocessed_data_pulse'))
    p.add_argument('--out-dir', type=Path, default=Path('outputs/results_hardpulse'))
    p.add_argument('--n-cities', type=int, default=50, help='Cities for NUTS (Stage 3).')
    p.add_argument('--svi1-steps', type=int, default=10000)
    p.add_argument('--svi2-steps', type=int, default=20000, help='Deprecated; ignored by dengue_hardpulse.')
    p.add_argument('--svi-city-steps', type=int, default=10000, help='Stage 5: all-city SVI steps.')
    p.add_argument('--nuts-warmup', type=int, default=1000)
    p.add_argument('--nuts-samples', type=int, default=1000)
    p.add_argument('--nuts-chains', type=int, default=1)
    p.add_argument('--target-accept', type=float, default=0.9)
    p.add_argument('--n-forecast-draws', type=int, default=200)
    p.add_argument('--state-residual-calibration', action='store_true', help='Apply training-residual state-level amplitude calibration before saving/evaluation/export.')
    p.add_argument('--state-calibration-shrinkage', type=float, default=1.0 / 3.0, help='Shrinkage applied to log observed/predicted training state residuals.')
    p.add_argument('--state-calibration-prior-cases', type=float, default=500.0, help='Pseudo-cases added to observed and predicted state totals for stable calibration.')
    p.add_argument('--state-calibration-max-log', type=float, default=1.5, help='Maximum absolute log state correction; 1.50 allows roughly 0.22x to 4.48x.')
    p.add_argument('--state-calibration-recent-seasons', type=int, default=1, help='Number of most recent observed epidemic seasons used for state residual calibration.')
    p.add_argument('--stage5-only', action='store_true', help='Run only Stage 5 all-city SVI, Stage 6 forecast, and Stage 7 evaluation.')
    p.add_argument('--stage5-posterior-path', type=Path, default=None, help='Saved calibration posterior .npz used to fix globals for --stage5-only.')
    p.add_argument('--forecast-batch', type=int, default=500, help='Cities per batch in Stage 6 (memory control).')
    p.add_argument('--skip-nuts', action='store_true', help='Use SVI posteriors instead of NUTS (fast check).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init-params', type=Path, default=None, help='Optional .npz of posterior median sample-site values to warm-start SVI.')
    p.add_argument('--skip-plots', action='store_true', help='Skip PNG forecast plots; useful for repeated challenge validation runs.')
    p.add_argument('--challenge-sequential', action='store_true', help='Run train_1->target_1 through train_4->target_4 and export mandatory UF forecasts.')
    p.add_argument('--challenge-splits', default='1,2,3,4', help='Comma-separated validation split numbers for --challenge-sequential.')
    p.add_argument('--challenge-work-dir', type=Path, default=Path('outputs/challenge_hardpulse'), help='Root directory for per-split preprocessing, model outputs, and exports.')
    p.add_argument('--challenge-export-dir', type=Path, default=None, help='Directory for mandatory state forecast CSVs. Defaults under --challenge-work-dir.')
    p.add_argument('--challenge-export-city', action='store_true', help='Also export Optional Challenge 1 dengue city-level CSVs for the 15 selected cities.')
    p.add_argument('--challenge-city-export-dir', type=Path, default=None, help='Directory for optional city forecast CSVs. Defaults under --challenge-work-dir.')
    p.add_argument('--challenge-city-geocodes', default=','.join((str(g) for g in OPTIONAL_DENGUE_CITY_GEOCODES)), help='Comma-separated adm_2 geocodes for Optional Challenge 1 city exports.')
    p.add_argument('--challenge-dengue-path', type=Path, default=Path('data/dengue.csv.gz'), help='Dengue challenge file used to recover target start dates and UF codes.')
    p.add_argument('--challenge-preprocess-script', type=Path, default=Path('scripts/build_dengue_pulse_inputs.py'), help='Script used to build split-specific hard-pulse tensor bundles.')
    p.add_argument('--merge-small-cities', action='store_true', help='Merge small nearby municipalities during challenge preprocessing.')
    p.add_argument('--small-pop-threshold', type=float, default=20000.0)
    p.add_argument('--max-merge-distance-km', type=float, default=30.0)
    p.add_argument('--allow-merge-koppen-mismatch', action='store_true', help='Allow small-city merges within state even when Koppen classes differ.')
    p.add_argument('--challenge-example-path', type=Path, default=Path('data/example_prediction2.csv'), help='Example-format CSV whose dates define the full target_4 horizon.')
    p.add_argument('--challenge-skip-nuts', action='store_true', help='Skip Stage 3 NUTS inside --challenge-sequential and use SVI2 guide samples instead.')
    p.add_argument('--challenge-jax-platform', default='cpu', choices=('cpu', 'gpu', ''), help='JAX platform for per-split challenge child runs. CPU avoids GPU OOM during full-city SVI; use gpu only if there is enough VRAM.')
    return p.parse_args()

def _observed_year_weeks_from_mask(mask, year_order):
    time_mask = np.asarray(mask, dtype=bool).any(axis=0)
    observed = []
    for yi, year in enumerate(year_order):
        for wi, has_obs in enumerate(time_mask[yi]):
            if has_obs:
                observed.append({'season_year': int(year), 'season_week': int(wi + 1)})
    return observed

def _format_observed_span(observed_year_weeks):
    if not observed_year_weeks:
        return 'no observed weeks'
    first = observed_year_weeks[0]
    last = observed_year_weeks[-1]
    if 'season_year' in first:
        return f"season {first['season_year']}-SW{first['season_week']:02d} to season {last['season_year']}-SW{last['season_week']:02d}"
    return f"{first['year']}-W{first['week']:02d} to {last['year']}-W{last['week']:02d}"

def _annotate_observed_window(data):
    mask = np.asarray(data['obs_mask'], dtype=bool)
    observed = data.get('observed_year_weeks')
    if not observed:
        observed = _observed_year_weeks_from_mask(mask, data.get('year_order', []))
    data['observed_year_weeks'] = observed
    data['n_observed_weeks'] = len(observed)
    data['observed_span'] = data.get('observed_span') or _format_observed_span(observed)
    data['time_observed_mask'] = jnp.array(mask.any(axis=0), dtype=bool)

def load_bundle(data_dir: Path, name: str):
    arrays = np.load(data_dir / f'{name}_arrays.npz')
    with open(data_dir / f'{name}_metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    data = {**meta, 'ifdm': jnp.array(arrays['ifdm']), 'log_density': jnp.array(arrays['log_density']), 'population': jnp.array(arrays['population']), 'temp_lag': jnp.array(arrays['temp_lag']), 'humid_lag': jnp.array(arrays['humid_lag']), 'rain_lag': jnp.array(arrays['rain_lag']), 'lat': jnp.array(arrays['lat']), 'lon': jnp.array(arrays['lon']), 'alt': jnp.array(arrays['alt']), 'koppen_idx': jnp.array(arrays['koppen_idx']), 'state_idx': jnp.array(arrays['state_idx']), 'obs_mask': jnp.array(arrays['obs_mask'])}
    data['n_states'] = int(np.asarray(arrays['state_idx']).max() + 1)
    data['n_koppen'] = int(np.asarray(arrays['koppen_idx']).max() + 1)
    nbr_idx, nbr_weights = build_neighbor_index(np.asarray(arrays['lat']), np.asarray(arrays['lon']), n_neighbours=N_NEIGHBOURS)
    data['nbr_idx'] = jnp.array(nbr_idx)
    data['nbr_weights'] = jnp.array(nbr_weights)
    _annotate_observed_window(data)
    return (data, jnp.array(arrays['Y']), jnp.array(arrays['mask']))

def _circular_smooth_rows(x: np.ndarray, window: int=5, start_week: int=1) -> np.ndarray:
    if window <= 1:
        return x
    start_idx = int(start_week) - 1
    order = np.r_[np.arange(start_idx, x.shape[1]), np.arange(0, start_idx)]
    inv_order = np.argsort(order)
    season_x = x[:, order]
    pad = window // 2
    padded = np.concatenate([season_x[:, -pad:], season_x, season_x[:, :pad]], axis=1)
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.stack([np.convolve(row, kernel, mode='valid') for row in padded], axis=0)
    return smoothed[:, inv_order]

def _state_week_profile_from_history(data, obs, window: int=7) -> jnp.ndarray:
    T = int(data['weeks_per_year'])
    n_state = int(data['n_states'])
    state_idx = np.asarray(data['state_idx'], dtype=int)
    y = np.asarray(obs, dtype=float)
    mask = np.asarray(data['obs_mask'], dtype=bool)
    y = np.where(mask, y, 0.0)
    state_week = np.zeros((n_state, T), dtype=float)
    for s in range(n_state):
        city_mask = state_idx == s
        if city_mask.any():
            state_week[s] = y[city_mask].sum(axis=(0, 1))
    national = state_week.sum(axis=0)
    if national.sum() <= 0:
        national = np.ones(T, dtype=float)
    national = national / national.sum()
    state_week = _circular_smooth_rows(state_week + 0.001, window=window)
    for s in range(n_state):
        if state_week[s].sum() <= 0:
            state_week[s] = national
        else:
            state_week[s] = state_week[s] / state_week[s].sum()
    return jnp.array(state_week, dtype=jnp.float32)

def _attach_state_week_profile(data_train: dict, data_test: dict, Y_train) -> None:
    profile = _state_week_profile_from_history(data_train, Y_train)
    data_train['state_week_profile'] = profile
    data_test['state_week_profile'] = profile

def _annual_attack_from_history(data, obs) -> jnp.ndarray:
    C = int(data['n_cities'])
    Y = int(data['n_years'])
    T = int(data['weeks_per_year'])
    mask = jnp.asarray(data['obs_mask']).astype(bool)
    y = jnp.where(mask, jnp.asarray(obs), 0.0)
    pop = jnp.maximum(jnp.asarray(data['population']).reshape(C, Y, T), 1.0)
    annual_cases = y.sum(axis=2)
    observed_weeks = jnp.maximum(mask.sum(axis=2), 1)
    annual_pop = jnp.maximum(jnp.where(mask, pop, 0.0).sum(axis=2) / observed_weeks, 1.0)
    return jnp.log1p(10000.0 * annual_cases / annual_pop)

def _state_annual_attack_from_history(data, obs) -> jnp.ndarray:
    C = int(data['n_cities'])
    Y = int(data['n_years'])
    T = int(data['weeks_per_year'])
    n_state = int(data.get('n_states', 1))
    state_idx = np.asarray(data.get('state_idx', np.zeros(C, dtype=int)), dtype=int)
    mask = np.asarray(data['obs_mask'], dtype=bool)
    y = np.where(mask, np.asarray(obs, dtype=float), 0.0)
    pop = np.maximum(np.asarray(data['population'], dtype=float).reshape(C, Y, T), 1.0)
    annual_cases = y.sum(axis=2)
    observed_weeks = np.maximum(mask.sum(axis=2), 1)
    annual_pop = np.maximum(np.where(mask, pop, 0.0).sum(axis=2) / observed_weeks, 1.0)
    state_cases = np.zeros((n_state, Y), dtype=float)
    state_pop = np.zeros((n_state, Y), dtype=float)
    for s in range(n_state):
        city_mask = state_idx == s
        if city_mask.any():
            state_cases[s] = annual_cases[city_mask].sum(axis=0)
            state_pop[s] = annual_pop[city_mask].sum(axis=0)
    state_pop = np.maximum(state_pop, 1.0)
    return jnp.array(np.log1p(10000.0 * state_cases / state_pop), dtype=jnp.float32)

def _attach_forecast_history(data_test: dict, data_train: dict, Y_train) -> None:
    data_test['history_attack'] = _annual_attack_from_history(data_train, Y_train)
    data_test['state_history_attack'] = _state_annual_attack_from_history(data_train, Y_train)

def _bounded_rho_year(raw) -> jnp.ndarray:
    return -0.1 + 0.9 * jax.nn.sigmoid(jnp.asarray(raw))

def _year_effect_last_from_samples(samples: dict, fixed_globals: dict | None=None) -> float:
    fixed_globals = {} if fixed_globals is None else fixed_globals
    if 'year_effect' in samples:
        return float(jnp.median(jnp.asarray(samples['year_effect'])[..., -1]))
    if 'year_noise' not in samples or 'sigma_year' not in samples:
        return 0.0
    year_noise = jnp.asarray(samples['year_noise'])
    sigma_year = jnp.asarray(samples['sigma_year'])
    rho_raw = fixed_globals.get('rho_year_raw', samples.get('rho_year_raw'))
    if rho_raw is None:
        return 0.0
    rho_year = _bounded_rho_year(rho_raw)
    prev = jnp.zeros_like(sigma_year)
    for yi in range(int(year_noise.shape[-1])):
        cur = rho_year * prev + sigma_year * year_noise[..., yi]
        prev = cur
    return float(jnp.median(prev))

def print_bundle(data, name):
    yr = data.get('year_order', [])
    T = int(data['weeks_per_year'])
    grid_weeks = int(data['n_years']) * T
    observed_weeks = int(data.get('n_observed_weeks', grid_weeks))
    observed_span = data.get('observed_span', 'unknown')
    print(f"  {name}: {data['n_cities']:,} cities × {data['n_years']} seasons × {T}w grid = {grid_weeks} slots  ({(yr[0] if yr else '?')}–{(yr[-1] if yr else '?')})")
    print(f'    observed weeks: {observed_weeks} ({observed_span})')
    if observed_weeks < grid_weeks:
        print(f'    padded unobserved slots: {grid_weeks - observed_weeks}; metrics use obs_mask only')
    mask = np.array(data['obs_mask'])
    print(f"    states {data['n_states']}  koppen regions {data['n_koppen']}  obs_coverage {mask.mean() * 100:.1f}%")
UF_LABELS = {11: 'RO - Rondonia', 12: 'AC - Acre', 13: 'AM - Amazonas', 14: 'RR - Roraima', 15: 'PA - Para', 16: 'AP - Amapa', 17: 'TO - Tocantins', 21: 'MA - Maranhao', 22: 'PI - Piaui', 23: 'CE - Ceara', 24: 'RN - Rio Grande do Norte', 25: 'PB - Paraiba', 26: 'PE - Pernambuco', 27: 'AL - Alagoas', 28: 'SE - Sergipe', 29: 'BA - Bahia', 31: 'MG - Minas Gerais', 32: 'ES - Espirito Santo', 33: 'RJ - Rio de Janeiro', 35: 'SP - Sao Paulo', 41: 'PR - Parana', 42: 'SC - Santa Catarina', 43: 'RS - Rio Grande do Sul', 50: 'MS - Mato Grosso do Sul', 51: 'MT - Mato Grosso', 52: 'GO - Goias', 53: 'DF - Distrito Federal'}

def _uf_label(uf_code: int) -> str:
    return UF_LABELS.get(int(uf_code), f'UF {int(uf_code):02d}')

def _inv_sp(y):
    return float(math.log(math.expm1(max(float(y), 1e-06))))

def _pm(samples, name):
    v = samples.get(name)
    return float(jnp.median(jnp.asarray(v))) if v is not None else float('nan')

def sanity(samples, label=''):
    sp, sg = (jax.nn.softplus, jax.nn.sigmoid)
    mu_fast = _pm(samples, 'mu_fast_nat')
    omega_raw = _pm(samples, 'omega_raw')
    rho_year_raw = _pm(samples, 'rho_year_raw')
    omega = 0.3 + 0.65 * float(sg(jnp.array(omega_raw))) if not math.isnan(omega_raw) else float('nan')
    rho_year = -0.1 + 0.9 * float(sg(jnp.array(rho_year_raw))) if not math.isnan(rho_year_raw) else float('nan')
    checks = [('omega', omega, 0.3, 0.95), ('rho_year', rho_year, -0.1, 0.8), ('beta_momentum', _pm(samples, 'beta_momentum'), -0.05, 0.12), ('beta_state_momentum', _pm(samples, 'beta_state_momentum'), -0.05, 0.15), ('tau_state_amp_bias', _pm(samples, 'tau_state_amp_bias'), 0.02, 0.5), ('theta', float(THETA_FIXED), 0.3, 15.0), ('mu_fast_nat_mod', float(mu_fast % 52.1772) if not math.isnan(mu_fast) else float('nan'), 0.0, 52.1772), ('sigma_fast', float(jnp.clip(sp(jnp.array(_pm(samples, 'lsig_fast_nat'))), 5.0, 18.0)), 5.0, 18.0), ('annual state base / 100k', float(100000.0 * math.exp(_pm(samples, 'log_rate_base_state'))) if not math.isnan(_pm(samples, 'log_rate_base_state')) else float('nan'), 0.01, 200.0), ('annual city base / 100k', float(100000.0 * math.exp(_pm(samples, 'log_rate_base_city'))) if not math.isnan(_pm(samples, 'log_rate_base_city')) else float('nan'), 0.01, 300.0), ('xi (immunity)', _pm(samples, 'xi'), 0.1, 0.8)]
    ok = True
    print(f'\n  Sanity [{label}]:')
    for name, val, lo, hi in checks:
        if math.isnan(val):
            print(f'    - {name:<20} = n/a')
            continue
        flag = '✓' if lo <= val <= hi else '⚠'
        if not lo <= val <= hi:
            ok = False
        print(f'    {flag} {name:<20} = {val:.4f}')
    print(f"  {('All OK ✓' if ok else 'Some out of range ⚠')}")
    return ok

def elapsed(t0):
    return f'{(time.time() - t0) / 60:.1f} min'

def _load_npz_dict(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Required artifact not found: {path}')
    arrs = np.load(path)
    return {k: jnp.array(arrs[k]) for k in arrs.files}

def _load_init_params(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f'Warm-start parameter file not found: {path}')
    arrs = np.load(path)
    params = {k: jnp.array(arrs[k]) for k in arrs.files if k not in CONSTRAINED_PARAMS}
    print(f'  Loaded {len(params)} warm-start params from {path}')
    return params

def _save_init_params(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in params.items() if k not in CONSTRAINED_PARAMS})
    print(f'  Saved warm-start params: {path}')

def _merge_init(primary: dict, warm: dict, drop: set | None=None) -> dict:
    drop = set() if drop is None else set(drop)
    merged = {k: v for k, v in warm.items() if k not in drop and k not in CONSTRAINED_PARAMS}
    merged.update({k: v for k, v in primary.items() if k not in drop and k not in CONSTRAINED_PARAMS})
    return merged

def _parse_splits(value: str) -> list[int]:
    splits = [int(v.strip()) for v in value.split(',') if v.strip()]
    bad = [s for s in splits if s not in (1, 2, 3, 4)]
    if bad:
        raise ValueError(f'Only validation splits 1..4 are supported, got {bad}')
    return splits

def _parse_geocodes(value: str) -> list[int]:
    geocodes = [int(v.strip()) for v in value.split(',') if v.strip()]
    if not geocodes:
        raise ValueError('At least one city geocode is required')
    duplicates = sorted({g for g in geocodes if geocodes.count(g) > 1})
    if duplicates:
        raise ValueError(f'Duplicate city geocodes are not allowed: {duplicates}')
    return geocodes

def _target4_full_horizon(example_path: Path) -> pd.DataFrame:
    if example_path.exists():
        dates = pd.to_datetime(pd.read_csv(example_path, usecols=['date'])['date'])
    else:
        dates = pd.date_range('2025-10-05', periods=53, freq='7D')
    epiweeks = []
    year, week = (2025, 41)
    for _ in range(len(dates)):
        epiweeks.append(year * 100 + week)
        week += 1
        if year == 2025 and week > 53:
            year, week = (2026, 1)
        elif year != 2025 and week > 52:
            year, week = (year + 1, 1)
    return pd.DataFrame({'date': dates, 'epiweek': epiweeks})

def _target_dates_and_indices(split: int, dengue_path: Path, year_order: list[int], example_path: Path | None=None) -> tuple[pd.DatetimeIndex, list[tuple[int, int]]]:
    if split == 4 and example_path is not None:
        target = _target4_full_horizon(example_path)
    else:
        target_col = f'target_{split}'
        df = pd.read_csv(dengue_path, compression='gzip', usecols=['date', 'epiweek', target_col], low_memory=False)
        sub = df[df[target_col].fillna(False).astype(bool)][['date', 'epiweek']]
        if sub.empty:
            raise ValueError(f'No rows marked {target_col} in {dengue_path}')
        target = sub.drop_duplicates('epiweek').sort_values('epiweek').assign(date=lambda x: pd.to_datetime(x['date']))
    year_index = {int(y): i for i, y in enumerate(year_order)}
    idx = []
    missing = []
    season_start = SEASON_START_WEEK
    for epiweek in target['epiweek'].astype(int):
        year = epiweek // 100
        week = epiweek % 100
        if not 1 <= week <= 53:
            missing.append(int(epiweek))
            continue
        season_year = year if week >= season_start else year - 1
        week_for_tensor = min(week, 52)
        season_week = (week_for_tensor - season_start) % 52 + 1
        if season_year not in year_index:
            missing.append(int(epiweek))
            continue
        idx.append((year_index[season_year], season_week - 1))
    if missing:
        raise ValueError(f'Test tensor year_order={year_order} cannot cover target epiweeks {missing[:10]}')
    return (pd.DatetimeIndex(target['date']), idx)

def _predictive_forecast_path(out_dir: Path) -> Path:
    predictive = out_dir / 'forecast_predictive_all.npy'
    return predictive if predictive.exists() else out_dir / 'forecast_all.npy'

def _recent_calibration_mask(mask: np.ndarray, recent_seasons: int) -> tuple[np.ndarray, list[int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f'Expected mask [city, season, week], got shape {mask.shape}')
    observed_by_season = mask.any(axis=(0, 2))
    observed_idx = np.flatnonzero(observed_by_season)
    if observed_idx.size == 0:
        return (mask, [])
    n_recent = max(int(recent_seasons), 1)
    selected = observed_idx[-n_recent:].tolist()
    recent_mask = np.zeros_like(mask, dtype=bool)
    recent_mask[:, selected, :] = mask[:, selected, :]
    return (recent_mask, selected)

def compute_state_residual_calibration(train_mu_draws: np.ndarray, Y_train, mask_train, data_train: dict, shrinkage: float=0.5, prior_cases: float=500.0, max_abs_log: float=0.7, recent_seasons: int=1) -> tuple[np.ndarray, pd.DataFrame]:
    pred = np.median(np.asarray(train_mu_draws, dtype=float), axis=0)
    obs = np.asarray(Y_train, dtype=float)
    mask_all = np.asarray(mask_train, dtype=bool)
    mask, selected_seasons = _recent_calibration_mask(mask_all, recent_seasons)
    C = int(data_train['n_cities'])
    state_idx = np.asarray(data_train.get('state_idx', np.zeros(C, dtype=int)), dtype=int)
    geocodes = np.asarray(data_train.get('geocode_order', np.arange(C)), dtype=np.int64)
    uf_codes = geocodes // 100000 if geocodes.size == C else state_idx
    year_order = list(data_train.get('year_order', []))
    selected_labels = [str(year_order[i]) if i < len(year_order) else str(i) for i in selected_seasons]
    n_states = int(data_train.get('n_states', int(state_idx.max()) + 1))
    correction = np.zeros(n_states, dtype=np.float32)
    rows = []
    for state in range(n_states):
        city_mask = state_idx == state
        if not city_mask.any():
            rows.append({'state_idx': state, 'uf_code': state, 'n_cities': 0, 'calibration_season_indices': ','.join(map(str, selected_seasons)), 'calibration_season_labels': ','.join(selected_labels), 'calibration_obs': 0.0, 'calibration_pred': 0.0, 'calibration_ratio_pred_over_obs': np.nan, 'all_train_obs': 0.0, 'all_train_pred': 0.0, 'all_train_ratio_pred_over_obs': np.nan, 'raw_log_correction': 0.0, 'log_correction': 0.0, 'multiplier': 1.0})
            continue
        m = mask[city_mask]
        m_all = mask_all[city_mask]
        obs_total = float(obs[city_mask][m].sum())
        pred_total = float(pred[city_mask][m].sum())
        all_obs_total = float(obs[city_mask][m_all].sum())
        all_pred_total = float(pred[city_mask][m_all].sum())
        raw = math.log((obs_total + prior_cases) / (pred_total + prior_cases))
        cal = float(np.clip(shrinkage * raw, -max_abs_log, max_abs_log))
        correction[state] = cal
        uf_vals = uf_codes[city_mask]
        uf_code = int(pd.Series(uf_vals).mode().iloc[0]) if len(uf_vals) else int(state)
        rows.append({'state_idx': state, 'uf_code': uf_code, 'n_cities': int(city_mask.sum()), 'calibration_season_indices': ','.join(map(str, selected_seasons)), 'calibration_season_labels': ','.join(selected_labels), 'calibration_obs': obs_total, 'calibration_pred': pred_total, 'calibration_ratio_pred_over_obs': float(pred_total / max(obs_total, 1.0)), 'all_train_obs': all_obs_total, 'all_train_pred': all_pred_total, 'all_train_ratio_pred_over_obs': float(all_pred_total / max(all_obs_total, 1.0)), 'raw_log_correction': raw, 'log_correction': cal, 'multiplier': float(math.exp(cal))})
    df = pd.DataFrame(rows)
    return (correction, df)

def apply_state_residual_calibration(fore_y_all: np.ndarray, fore_all: np.ndarray, data_test: dict, correction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state_idx = np.asarray(data_test['state_idx'], dtype=int)
    factor = np.exp(np.asarray(correction, dtype=float)[state_idx]).astype(np.float32)
    factor = factor[None, :, None, None]
    return (fore_y_all * factor, fore_all * factor)

def maybe_apply_state_residual_calibration(args, data_train: dict, Y_train, mask_train, data_test: dict, svi_city: dict, globals_median: dict, fore_y_all: np.ndarray, fore_all: np.ndarray, rng_key) -> tuple[np.ndarray, np.ndarray]:
    if not args.state_residual_calibration:
        return (fore_y_all, fore_all)
    print(f'\n{SEP}\nState residual calibration (recent training season only)\n{SEP}')
    train_mu = predict_training_mu_all_cities(data_train=data_train, Y_train=Y_train, all_city_svi=svi_city, fixed_globals=globals_median, n_draws=args.n_forecast_draws, batch_size=args.forecast_batch, rng_key=rng_key)
    correction, cal_df = compute_state_residual_calibration(train_mu_draws=train_mu, Y_train=Y_train, mask_train=mask_train, data_train=data_train, shrinkage=args.state_calibration_shrinkage, prior_cases=args.state_calibration_prior_cases, max_abs_log=args.state_calibration_max_log, recent_seasons=args.state_calibration_recent_seasons)
    cal_path = args.out_dir / 'state_residual_calibration.csv'
    cal_df.to_csv(cal_path, index=False)
    print(f'  Saved -> {cal_path}')
    print(f"  State multiplier range: {cal_df['multiplier'].min():.3f} to {cal_df['multiplier'].max():.3f}; median {cal_df['multiplier'].median():.3f}")
    return apply_state_residual_calibration(fore_y_all, fore_all, data_test, correction)

def _official_prediction_frame(draws_52: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    q = np.quantile(np.maximum(draws_52, 0.0), [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975], axis=0)
    out = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'pred': q[4], 'lower_50': q[3], 'upper_50': q[5], 'lower_80': q[2], 'upper_80': q[6], 'lower_90': q[1], 'upper_90': q[7], 'lower_95': q[0], 'upper_95': q[8]})
    value_cols = [c for c in out.columns if c != 'date']
    out[value_cols] = out[value_cols].clip(lower=0.0)
    return out

def export_mandatory_state_forecasts(forecast_path: Path, data_dir: Path, output_dir: Path, split: int, dengue_path: Path, example_path: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = output_dir / f'target_{split}'
    target_dir.mkdir(parents=True, exist_ok=True)
    fore = np.load(forecast_path)
    metadata_path = data_dir / 'test_metadata.json'
    with open(metadata_path, encoding='utf-8') as f:
        meta = json.load(f)
    geocodes = np.asarray(meta['geocode_order'], dtype=np.int64)
    uf_codes = geocodes // 100000
    dates, target_idx = _target_dates_and_indices(split, dengue_path, meta['year_order'], example_path=example_path)
    flat = np.stack([fore[:, :, yi, wi] for yi, wi in target_idx], axis=-1)
    rows = []
    for uf_code in sorted(np.unique(uf_codes)):
        uf_code = int(uf_code)
        if uf_code == 32:
            continue
        mask = uf_codes == uf_code
        state_draws = flat[:, mask, :].sum(axis=1)
        pred = _official_prediction_frame(state_draws, dates)
        pred.to_csv(target_dir / f'adm1_{uf_code:02d}.csv', index=False)
        pred.to_csv(output_dir / f'target_{split}_adm1_{uf_code:02d}.csv', index=False)
        long = pred.copy()
        long.insert(0, 'target', split)
        long.insert(1, 'adm_level', 1)
        long.insert(2, 'adm_1', uf_code)
        rows.append(long)
    combined = pd.concat(rows, ignore_index=True)
    combined_path = output_dir / f'target_{split}_all_adm1.csv'
    combined.to_csv(combined_path, index=False)
    print(f"  Exported target_{split} forecasts: {target_dir} ({combined['adm_1'].nunique()} states, {len(dates)} weeks)")
    return combined

def plot_state_forecast_comparison(forecast_path: Path, data_dir: Path, output_path: Path, split: int, dengue_path: Path, example_path: Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fore = np.load(forecast_path)
    arrays = np.load(data_dir / 'test_arrays.npz')
    Y = np.asarray(arrays['Y'], dtype=float)
    mask_test = np.asarray(arrays['mask'], dtype=bool)
    with open(data_dir / 'test_metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    geocodes = np.asarray(meta['geocode_order'], dtype=np.int64)
    uf_codes = geocodes // 100000
    dates, target_idx = _target_dates_and_indices(split, dengue_path, meta['year_order'], example_path=example_path)
    fore_flat = np.stack([fore[:, :, yi, wi] for yi, wi in target_idx], axis=-1)
    obs_flat = np.stack([Y[:, yi, wi] for yi, wi in target_idx], axis=-1)
    mask_flat = np.stack([mask_test[:, yi, wi] for yi, wi in target_idx], axis=-1)
    states = [int(u) for u in sorted(np.unique(uf_codes)) if int(u) != 32]
    n_states = len(states)
    n_cols = 4
    n_rows = int(math.ceil(n_states / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows), sharex=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    x = np.arange(len(dates))
    for ax, uf_code in zip(axes, states):
        city_mask = uf_codes == uf_code
        draws = fore_flat[:, city_mask, :].sum(axis=1)
        q = np.quantile(np.maximum(draws, 0.0), [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975], axis=0)
        obs_mask = mask_flat[city_mask, :].any(axis=0)
        obs = np.where(obs_mask, np.where(mask_flat[city_mask, :], obs_flat[city_mask, :], 0.0).sum(axis=0), np.nan)
        ax.fill_between(x, q[0], q[6], color='#9ecae1', alpha=0.28, linewidth=0)
        ax.fill_between(x, q[1], q[5], color='#6baed6', alpha=0.32, linewidth=0)
        ax.fill_between(x, q[2], q[4], color='#3182bd', alpha=0.34, linewidth=0)
        ax.plot(x, q[3], color='#08519c', linewidth=1.6)
        if np.isfinite(obs).any():
            ax.plot(x, obs, color='#111111', linewidth=1.2, marker='o', markersize=2.2)
        ax.set_title(_uf_label(uf_code), fontsize=8.5)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes[n_states:]:
        ax.axis('off')
    tick_idx = np.linspace(0, max(len(dates) - 1, 0), min(4, len(dates)), dtype=int)
    tick_labels = [pd.Timestamp(dates[i]).strftime('%Y-%m-%d') for i in tick_idx]
    for ax in axes[:n_states]:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=30, ha='right')
    legend_handles = [Patch(facecolor='#9ecae1', alpha=0.28, label='95% interval'), Patch(facecolor='#6baed6', alpha=0.32, label='80% interval'), Patch(facecolor='#3182bd', alpha=0.34, label='50% interval'), Line2D([0], [0], color='#08519c', linewidth=1.6, label='Forecast median'), Line2D([0], [0], color='#111111', linewidth=1.2, marker='o', markersize=3, label='Observed')]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(legend_handles), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f'Dengue hard-pulse UF forecasts vs observed - target_{split}', fontsize=14, y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f'  Saved state-level forecast plot: {output_path}')
    return output_path

def export_optional_city_forecasts(forecast_path: Path, data_dir: Path, output_dir: Path, split: int, dengue_path: Path, example_path: Path, city_geocodes: list[int]) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = output_dir / f'target_{split}'
    target_dir.mkdir(parents=True, exist_ok=True)
    fore = np.load(forecast_path)
    metadata_path = data_dir / 'test_metadata.json'
    with open(metadata_path, encoding='utf-8') as f:
        meta = json.load(f)
    geocodes = np.asarray(meta['geocode_order'], dtype=np.int64)
    city_pos = {int(geocode): i for i, geocode in enumerate(geocodes)}
    merge_mapping_path = data_dir.parent / 'merge_mapping.csv'
    merge_lookup = {}
    if merge_mapping_path.exists():
        mapping = pd.read_csv(merge_mapping_path)
        required = {'original_geocode', 'model_geocode', 'disagg_population_share'}
        missing_cols = required - set(mapping.columns)
        if missing_cols:
            raise ValueError(f'Merge mapping {merge_mapping_path} is missing columns: {sorted(missing_cols)}')
        for row in mapping.itertuples(index=False):
            merge_lookup[int(row.original_geocode)] = (int(row.model_geocode), float(row.disagg_population_share))
    unresolved = []
    city_draw_specs = {}
    for geocode in city_geocodes:
        geocode = int(geocode)
        if geocode in merge_lookup:
            model_geocode, share = merge_lookup[geocode]
            if model_geocode in city_pos:
                mode = 'direct' if model_geocode == geocode and abs(share - 1.0) < 1e-09 else 'disaggregated'
                city_draw_specs[geocode] = (model_geocode, share, mode)
                continue
        if geocode in city_pos:
            city_draw_specs[geocode] = (geocode, 1.0, 'direct')
            continue
        unresolved.append(geocode)
    if unresolved:
        raise ValueError(f'Optional city geocodes missing from {metadata_path} and merge mapping {merge_mapping_path}: {unresolved}')
    dates, target_idx = _target_dates_and_indices(split, dengue_path, meta['year_order'], example_path=example_path)
    flat = np.stack([fore[:, :, yi, wi] for yi, wi in target_idx], axis=-1)
    rows = []
    n_disaggregated = 0
    for geocode in city_geocodes:
        model_geocode, share, mode = city_draw_specs[int(geocode)]
        draws = flat[:, city_pos[model_geocode], :] * share
        if mode == 'disaggregated':
            n_disaggregated += 1
        pred = _official_prediction_frame(draws, dates)
        pred.to_csv(target_dir / f'adm2_{geocode}.csv', index=False)
        pred.to_csv(output_dir / f'target_{split}_adm2_{geocode}.csv', index=False)
        long = pred.copy()
        long.insert(0, 'target', split)
        long.insert(1, 'adm_level', 2)
        long.insert(2, 'adm_2', geocode)
        rows.append(long)
    combined = pd.concat(rows, ignore_index=True)
    combined_path = output_dir / f'target_{split}_all_adm2.csv'
    combined.to_csv(combined_path, index=False)
    n_cities = combined['adm_2'].nunique()
    suffix = f', {n_disaggregated} disaggregated from merged units' if n_disaggregated else ''
    print(f'  Exported target_{split} city forecasts: {target_dir} ({n_cities} cities, {len(dates)} weeks{suffix})')
    return combined

def run_challenge_sequential(args) -> None:
    splits = _parse_splits(args.challenge_splits)
    export_dir = args.challenge_export_dir or args.challenge_work_dir / 'mandatory_state_forecasts'
    export_dir.mkdir(parents=True, exist_ok=True)
    city_geocodes = _parse_geocodes(args.challenge_city_geocodes)
    city_export_dir = args.challenge_city_export_dir or args.challenge_work_dir / 'optional_city_forecasts'
    if args.challenge_export_city:
        city_export_dir.mkdir(parents=True, exist_ok=True)
    previous_init = args.init_params
    stage5_posterior_path = None
    combined = []
    combined_city = []
    script = Path(__file__).resolve()
    for split in splits:
        print(f'\n{SEP}\nChallenge validation split {split}: train_{split} -> target_{split}\n{SEP}')
        split_root = args.challenge_work_dir / f'validation_{split}'
        data_dir = split_root / 'preprocessed'
        out_dir = split_root / 'model'
        master_path = split_root / 'data_pulse.parquet'
        pop_path = split_root / 'population_weekly.parquet'
        split_root.mkdir(parents=True, exist_ok=True)
        prep_cmd = [sys.executable, str(args.challenge_preprocess_script), '--dengue-path', str(args.challenge_dengue_path), '--split', str(split), '--master-output', str(master_path), '--population-weekly-output', str(pop_path), '--tensor-output-dir', str(data_dir)]
        if args.merge_small_cities:
            prep_cmd.extend(['--merge-small-cities', '--small-pop-threshold', str(args.small_pop_threshold), '--max-merge-distance-km', str(args.max_merge_distance_km), '--merge-mapping-output', str(split_root / 'merge_mapping.csv')])
            if args.allow_merge_koppen_mismatch:
                prep_cmd.append('--allow-merge-koppen-mismatch')
        if split > 1:
            prep_cmd.append('--include-previous-target')
        print('  Preprocessing:', ' '.join(prep_cmd))
        subprocess.run(prep_cmd, check=True)
        run_cmd = [sys.executable, str(script), '--data-dir', str(data_dir), '--out-dir', str(out_dir), '--n-cities', str(args.n_cities), '--svi1-steps', str(args.svi1_steps), '--svi2-steps', str(args.svi2_steps), '--svi-city-steps', str(args.svi_city_steps), '--nuts-warmup', str(args.nuts_warmup), '--nuts-samples', str(args.nuts_samples), '--nuts-chains', str(args.nuts_chains), '--target-accept', str(args.target_accept), '--n-forecast-draws', str(args.n_forecast_draws), '--forecast-batch', str(args.forecast_batch), '--seed', str(args.seed + split), '--skip-plots']
        if args.state_residual_calibration:
            run_cmd.extend(['--state-residual-calibration', '--state-calibration-shrinkage', str(args.state_calibration_shrinkage), '--state-calibration-prior-cases', str(args.state_calibration_prior_cases), '--state-calibration-max-log', str(args.state_calibration_max_log), '--state-calibration-recent-seasons', str(args.state_calibration_recent_seasons)])
        update_only = split > 1
        if update_only:
            if stage5_posterior_path is None:
                raise RuntimeError('Stage 5-only update requires split 1 calibration posterior')
            if previous_init is None:
                raise RuntimeError('Stage 5-only update requires previous split median params')
            run_cmd.extend(['--stage5-only', '--stage5-posterior-path', str(stage5_posterior_path)])
        elif args.challenge_skip_nuts:
            run_cmd.append('--skip-nuts')
        if previous_init is not None:
            run_cmd.extend(['--init-params', str(previous_init)])
        child_env = os.environ.copy()
        if args.challenge_jax_platform:
            child_env['JAX_PLATFORM_NAME'] = args.challenge_jax_platform
        child_env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        child_env.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.70')
        child_env.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
        print('  Fitting/forecasting:', ' '.join(run_cmd))
        print(f"  Child JAX platform: {child_env.get('JAX_PLATFORM_NAME', 'default')}")
        subprocess.run(run_cmd, check=True, env=child_env)
        if split == 1:
            posterior_name = 'svi1_guide_samples.npz' if args.challenge_skip_nuts else 'nuts_samples.npz'
            stage5_posterior_path = out_dir / posterior_name
        previous_init = out_dir / 'all_city_median_params.npz'
        export_forecast_path = _predictive_forecast_path(out_dir)
        if export_forecast_path.name == 'forecast_predictive_all.npy':
            print('  Exporting/plotting posterior predictive count draws.')
        else:
            print('  Exporting/plotting latent mean draws; predictive count draws not found.')
        combined.append(export_mandatory_state_forecasts(forecast_path=export_forecast_path, data_dir=data_dir, output_dir=export_dir, split=split, dengue_path=args.challenge_dengue_path, example_path=args.challenge_example_path))
        plot_state_forecast_comparison(forecast_path=export_forecast_path, data_dir=data_dir, output_path=out_dir / f'state_forecast_target_{split}.png', split=split, dengue_path=args.challenge_dengue_path, example_path=args.challenge_example_path)
        if args.challenge_export_city:
            combined_city.append(export_optional_city_forecasts(forecast_path=export_forecast_path, data_dir=data_dir, output_dir=city_export_dir, split=split, dengue_path=args.challenge_dengue_path, example_path=args.challenge_example_path, city_geocodes=city_geocodes))
    all_path = export_dir / 'mandatory_dengue_state_all_validations.csv'
    pd.concat(combined, ignore_index=True).to_csv(all_path, index=False)
    manifest = {'disease': 'A90', 'case_definition': 'probable', 'adm_level': 1, 'adm_0': 'BRA', 'excluded_adm_1': [32], 'splits': splits, 'horizon': 'Targets 1-3 use marked target_i dates; target_4 uses the full date horizon from data/example_prediction2.csv.', 'epiweek_53_handling': 'The epidemic-season tensor has 52 slots; EW53 folds into season week 12 before EW01 maps to season week 13.', 'update_scheme': 'For split i>1, training rows are train_i plus target_{i-1} rows. Split 1 runs stages 1-7 with Stage 3 NUTS. Splits 2-4 run only Stage 5 all-city SVI, Stage 6 forecast, and Stage 7 export, warm-started from split i-1 all-city median params.', 'state_residual_calibration': bool(args.state_residual_calibration), 'state_residual_calibration_details': 'When enabled, each split computes bounded state amplitude multipliers from recent training epidemic-season observed/predicted residual totals and applies them before validation exports.', 'prediction_columns': ['date', 'pred', 'lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_90', 'upper_90', 'lower_95', 'upper_95']}
    with open(export_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    if args.challenge_export_city:
        city_all_path = city_export_dir / 'optional_dengue_city_all_validations.csv'
        pd.concat(combined_city, ignore_index=True).to_csv(city_all_path, index=False)
        city_manifest = {'disease': 'A90', 'case_definition': 'probable', 'adm_level': 2, 'adm_0': 'BRA', 'adm_2': city_geocodes, 'splits': splits, 'horizon': manifest['horizon'], 'epiweek_53_handling': manifest['epiweek_53_handling'], 'update_scheme': manifest['update_scheme'], 'prediction_columns': manifest['prediction_columns']}
        with open(city_export_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(city_manifest, f, indent=2)
        print(f'Optional city combined CSV: {city_all_path}')
    print(f'\n{SEP}\nChallenge exports complete: {export_dir}\nCombined CSV: {all_path}\n{SEP}')

def main():
    args = parse_args()
    if args.challenge_sequential:
        run_challenge_sequential(args)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)
    t_total = time.time()
    warm_start_params = _load_init_params(args.init_params)
    print(f'Devices: {jax.devices()}')
    _t = jnp.ones((1000, 1000))
    print(f'GPU: {float((_t @ _t).mean()):.0f}  (expect 1000)\n')
    print(f'{SEP}\nLoading data\n{SEP}')
    data_train, Y_train, mask_train = load_bundle(args.data_dir, 'train')
    data_test, Y_test, mask_test = load_bundle(args.data_dir, 'test')
    print_bundle(data_train, 'train')
    print_bundle(data_test, 'test')
    _attach_forecast_history(data_test, data_train, Y_train)
    _attach_state_week_profile(data_train, data_test, Y_train)
    C = int(data_train['n_cities'])
    if args.stage5_only:
        if args.init_params is None:
            raise ValueError('--stage5-only requires --init-params from the previous split')
        posterior_path = args.stage5_posterior_path or args.out_dir / 'nuts_samples.npz'
        print(f'\n{SEP}\nStage 5-only update: loading fixed globals and previous medians\n{SEP}')
        posterior1 = _load_npz_dict(posterior_path)
        globals_median = {}
        for name in GLOBAL_FIX_SITES:
            val = posterior1.get(name)
            if val is not None:
                globals_median[name] = float(jnp.median(jnp.asarray(val)))
        init_params = _load_init_params(args.init_params)
        print(f'\n{SEP}\nStage 5: All-city SVI update (5570 cities)\n{SEP}')
        print('  National-level params fixed from split 1 NUTS medians.')
        print('  Warm-start: previous split all-city medians.')
        rng, k = jax.random.split(rng)
        t0 = time.time()
        svi_city = run_all_city_svi(data_train=data_train, Y_train=Y_train, nuts2_samples=posterior1, n_steps=args.svi_city_steps, lr=5e-05, lr_final=2e-06, warmup_steps=min(1000, args.svi_city_steps // 10), print_every=max(500, args.svi_city_steps // 20), rng_key=k, init_params=init_params, n_posterior_draws=args.n_forecast_draws)
        print(f'  Stage 4 complete ({elapsed(t0)})')
        _save_init_params(args.out_dir / 'all_city_median_params.npz', svi_city['median_params'])
        print(f"\n{SEP}\nStage 5: Forecast all {C:,} cities ({data_test.get('n_observed_weeks', 0)} observed test weeks)\n{SEP}")
        rng, k = jax.random.split(rng)
        t0 = time.time()
        data_test['year_effect_prev'] = _year_effect_last_from_samples(svi_city['guide_samples'], fixed_globals=globals_median)
        fore_y_all, fore_all = forecast_all_cities(data_test=data_test, all_city_svi=svi_city, fixed_globals=globals_median, n_draws=args.n_forecast_draws, batch_size=args.forecast_batch, rng_key=k)
        print(f'  Stage 5 complete ({elapsed(t0)})')
        print(f'  Forecast shape: {fore_all.shape}')
        rng, k = jax.random.split(rng)
        fore_y_all, fore_all = maybe_apply_state_residual_calibration(args=args, data_train=data_train, Y_train=Y_train, mask_train=mask_train, data_test=data_test, svi_city=svi_city, globals_median=globals_median, fore_y_all=fore_y_all, fore_all=fore_all, rng_key=k)
        np.save(args.out_dir / 'forecast_all.npy', fore_all.astype(np.float32))
        np.save(args.out_dir / 'forecast_predictive_all.npy', fore_y_all.astype(np.float32))
        np.save(args.out_dir / 'Y_test.npy', np.array(Y_test).astype(np.float32))
        np.save(args.out_dir / 'mask_test.npy', np.array(mask_test).astype(bool))
        print(f'\n{SEP}\nStage 6: Evaluate all cities\n{SEP}')
        print_skill_metrics(fore_all, np.array(Y_test), np.array(mask_test))
        metrics_df = evaluate_all_cities(fore=fore_all, Y_test=np.array(Y_test), mask_test=np.array(mask_test), data_test=data_test, save_path=args.out_dir / 'metrics_per_city.csv')
        print_state_summary(metrics_df, data_test)
        print(f'\n{SEP}')
        print(f'Pipeline complete.  Total time: {elapsed(t_total)}')
        print(f'All outputs in: {args.out_dir}/')
        print(SEP)
        return
    print(f'\n{SEP}\nStage 1: SVI warm-up\n{SEP}')
    bio_fixed = {}
    rng, k = jax.random.split(rng)
    t0 = time.time()
    svi1 = run_svi(data_train, Y_train, model_fn=None, n_steps=args.svi1_steps, lr=0.0001, lr_final=5e-06, warmup_steps=min(1000, args.svi1_steps // 10), print_every=max(500, args.svi1_steps // 20), rng_key=k, init_params=_merge_init({}, warm_start_params, drop=set(bio_fixed)) or None, n_posterior_draws=20)
    print(f"  Stage 1 complete ({elapsed(t0)})  ELBO: {-svi1['losses'][0]:,.0f} → {-svi1['losses'][-1]:,.0f}")
    np.save(args.out_dir / 'svi1_losses.npy', svi1['losses'])
    sanity(svi1['guide_samples'], 'SVI Stage 1')
    _save_init_params(args.out_dir / 'svi1_median_params.npz', svi1['median_params'])
    print(f'\n  Selecting {args.n_cities} cities for NUTS …')
    city_ids = select_important_cities(data_train, Y_train, n_cities=args.n_cities)
    sub_data, sub_obs = build_subset_data(data_train, Y_train, city_ids)
    print(f'\n{SEP}\nStage 2: NUTS — calibrated posterior on subset\n{SEP}')
    if not args.skip_nuts:
        rng, k = jax.random.split(rng)
        t0 = time.time()
        nuts1 = run_nuts(sub_data, sub_obs, init_params=svi1['median_params'], n_warmup=args.nuts_warmup, n_samples=args.nuts_samples, n_chains=args.nuts_chains, target_accept=args.target_accept, rng_key=k)
        print(f'  Stage 2 complete ({elapsed(t0)})')
        check_convergence(nuts1, params=['omega', 'rho_year', 'theta', 'mu_fast_nat', 'lsig_fast_nat', 'log_rate_base_state', 'log_rate_base_city', 'beta_log_pop_base', 'beta_log_density_base', 'beta_momentum', 'beta_state_momentum', 'xi', 'sigma_year', 'sigma_amp', 'tau_state_amp_bias'])
        sanity(nuts1['samples'], 'NUTS Stage 2')
        posterior1 = nuts1['samples']
        np.savez(args.out_dir / 'nuts_samples.npz', **{kk: np.array(v) for kk, v in posterior1.items()})
    else:
        print('  --skip-nuts: using SVI1 guide samples')
        posterior1 = svi1['guide_samples']
        np.savez(args.out_dir / 'svi1_guide_samples.npz', **{kk: np.array(v) for kk, v in posterior1.items()})
    print(f'\n{SEP}\nStage 3: Extract global parameters\n{SEP}')
    globals_median = {}
    for name in GLOBAL_FIX_SITES:
        val = posterior1.get(name)
        if val is not None:
            globals_median[name] = float(jnp.median(jnp.asarray(val)))
    print('  Global parameter medians (model/constrained space where applicable):')
    for name, val in sorted(globals_median.items()):
        print(f'    {name:<20} = {val:+.4f}')
    if 'period_slow_global' in globals_median:
        print(f"    -> slow-clock period fixed at {globals_median['period_slow_global'] / 52.1772:.2f} years")
    with open(args.out_dir / 'globals_median.json', 'w', encoding='utf-8') as f:
        json.dump(globals_median, f, indent=2)
    print(f'\n{SEP}\nStage 4: All-city SVI (5570 cities)\n{SEP}')
    print('  National-level params fixed at NUTS medians.')
    print('  Free: every city/state/koppen deviation + tau_* smoothing hyperpriors.')
    rng, k = jax.random.split(rng)
    t0 = time.time()
    svi_city = run_all_city_svi(data_train=data_train, Y_train=Y_train, nuts2_samples=posterior1, n_steps=args.svi_city_steps, lr=5e-05, lr_final=2e-06, warmup_steps=min(1000, args.svi_city_steps // 10), print_every=max(500, args.svi_city_steps // 20), rng_key=k, init_params=svi1['median_params'], n_posterior_draws=args.n_forecast_draws)
    print(f'  Stage 4 complete ({elapsed(t0)})')
    _save_init_params(args.out_dir / 'all_city_median_params.npz', svi_city['median_params'])
    print(f'\n{SEP}\nInterim: Posterior predictive check (training subset)\n{SEP}')
    rng, k = jax.random.split(rng)
    ppc = posterior_predictive(sub_data, posterior1, n_draws=min(200, len(next(iter(posterior1.values())))), rng_key=k)
    ppc_np = np.array(ppc)
    obs_np = np.array(sub_obs)
    m_np = np.array(sub_data['obs_mask'])
    lo, hi = np.percentile(ppc_np, [5, 95], axis=0)
    cov = float(((obs_np >= lo) & (obs_np <= hi))[m_np].mean())
    print(f'  90% PPC coverage on subset: {cov * 100:.1f}%  (ideal ≈ 90%)')
    print(f"\n{SEP}\nStage 5: Forecast all {C:,} cities ({data_test.get('n_observed_weeks', 0)} observed test weeks)\n{SEP}")
    rng, k = jax.random.split(rng)
    t0 = time.time()
    data_test['year_effect_prev'] = _year_effect_last_from_samples(svi_city['guide_samples'], fixed_globals=globals_median)
    fore_y_all, fore_all = forecast_all_cities(data_test=data_test, all_city_svi=svi_city, fixed_globals=globals_median, n_draws=args.n_forecast_draws, batch_size=args.forecast_batch, rng_key=k)
    print(f'  Stage 5 complete ({elapsed(t0)})')
    print(f'  Forecast shape: {fore_all.shape}')
    rng, k = jax.random.split(rng)
    fore_y_all, fore_all = maybe_apply_state_residual_calibration(args=args, data_train=data_train, Y_train=Y_train, mask_train=mask_train, data_test=data_test, svi_city=svi_city, globals_median=globals_median, fore_y_all=fore_y_all, fore_all=fore_all, rng_key=k)
    np.save(args.out_dir / 'forecast_all.npy', fore_all.astype(np.float32))
    np.save(args.out_dir / 'forecast_predictive_all.npy', fore_y_all.astype(np.float32))
    np.save(args.out_dir / 'Y_test.npy', np.array(Y_test).astype(np.float32))
    np.save(args.out_dir / 'mask_test.npy', np.array(mask_test).astype(bool))
    print(f'\n{SEP}\nStage 6: Evaluate all cities\n{SEP}')
    print_skill_metrics(fore_all, np.array(Y_test), np.array(mask_test))
    metrics_df = evaluate_all_cities(fore=fore_all, Y_test=np.array(Y_test), mask_test=np.array(mask_test), data_test=data_test, save_path=args.out_dir / 'metrics_per_city.csv')
    print_state_summary(metrics_df, data_test)
    if not args.skip_plots:
        plot_top_cities(fore=fore_all, Y_test=np.array(Y_test), mask_test=np.array(mask_test), data_test=data_test, metrics_df=metrics_df, n_top=20, n_cols=4, save_path=str(args.out_dir / 'top20_cities.png'))
        plot_coverage_by_week(fore=fore_all, test_obs=np.array(Y_test), test_mask=np.array(mask_test), test_years=data_test.get('year_order', []), save_path=str(args.out_dir / 'coverage_by_week_all.png'))
        sub_test, sub_test_obs = build_subset_data(data_test, Y_test, city_ids)
        fore_sub = fore_all[:, city_ids, :, :]
        geocodes = data_train.get('geocode_order', list(range(C)))
        plot_forecast_vs_observed(fore=fore_sub, train_obs=np.array(sub_obs), test_obs=np.array(sub_test_obs), test_data=sub_test, train_data=sub_data, city_ids=city_ids, geocode_order=geocodes, n_cols=2, last_n_train_years=4, save_path=str(args.out_dir / 'forecast_nuts_cities.png'))
        plot_coverage_by_week(fore=fore_sub, test_obs=np.array(sub_test_obs), test_mask=np.array(sub_test['obs_mask']), test_years=data_test.get('year_order', []), save_path=str(args.out_dir / 'coverage_nuts_cities.png'))
    print(f'\n{SEP}')
    print(f'Pipeline complete.  Total time: {elapsed(t_total)}')
    print(f'All outputs in: {args.out_dir}/')
    print('\n  Files:')
    for fpath in sorted(args.out_dir.iterdir()):
        sz = fpath.stat().st_size / 1024
        print(f'    {fpath.name:<40} {sz:8.1f} KB')
    print(SEP)
if __name__ == '__main__':
    main()
