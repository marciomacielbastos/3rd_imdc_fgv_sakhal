import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pandas as pd
LAG_WEEKS = 8
WEEKS_PER_YEAR = 52
EPIDEMIC_SEASON_START = 41

def add_epidemic_season_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    week = df['week'].astype(int)
    year = df['year'].astype(int)
    df['season_year'] = np.where(week >= EPIDEMIC_SEASON_START, year, year - 1).astype('int64')
    capped_week = week.clip(upper=WEEKS_PER_YEAR)
    df['season_week'] = ((capped_week - EPIDEMIC_SEASON_START) % WEEKS_PER_YEAR + 1).astype('int64')
    return df

def add_log_density(df):
    area_km2 = df['constructed_area_m2'] / 1000000.0
    density = df['population'] / area_km2.clip(lower=0.001)
    df = df.copy()
    df['log_density'] = np.log(density.clip(lower=0.001))
    return df

def reindex_full_grid(df):
    df = df[df['week'] <= WEEKS_PER_YEAR].copy()
    all_yw = df[['year', 'week']].drop_duplicates()
    grid = pd.DataFrame({'geocode': df['geocode'].unique()}).merge(all_yw, how='cross')
    out = grid.merge(df, on=['geocode', 'year', 'week'], how='left')
    out = out.sort_values(['geocode', 'year', 'week']).reset_index(drop=True)
    out['train'] = out['train'].notna() & out['train'].eq(True)
    out['test'] = out['test'].notna() & out['test'].eq(True)
    return add_epidemic_season_columns(out)

def add_lagged_covariates(df, lag=LAG_WEEKS):
    df = df.copy().sort_values(['geocode', 'year', 'week'])
    for raw, lagged in [('temp_med', 'temp_lag'), ('humid_med', 'humid_lag'), ('precip_med', 'rain_lag')]:
        df[lagged] = df.groupby('geocode')[raw].transform(lambda x: x.shift(lag))
        df[lagged] = df[lagged].fillna(df.groupby('geocode')[raw].transform('mean'))
    return df

def encode_categoricals(df):
    df = df.copy()
    uf_cats = sorted(df['uf'].dropna().unique())
    koppen_cats = sorted(df['koppen'].dropna().unique())
    df['state_idx'] = df['uf'].map({v: i for i, v in enumerate(uf_cats)})
    df['koppen_idx'] = df['koppen'].map({v: i for i, v in enumerate(koppen_cats)})
    print(f'  States (S): {len(uf_cats)}   Köppen (K): {len(koppen_cats)}')
    return (df, {'uf': uf_cats, 'koppen': koppen_cats})

def compute_h_bar(df_train, geocode_order, cols=('temp_lag', 'humid_lag', 'rain_lag')):
    C = len(geocode_order)
    geo_index = {g: i for i, g in enumerate(geocode_order)}
    h_bar = {col: np.full((C, WEEKS_PER_YEAR), np.nan, dtype=np.float32) for col in cols}
    sub = df_train[df_train['geocode'].isin(geocode_order)]
    for col in cols:
        pivot = sub.groupby(['geocode', 'season_week'])[col].mean().unstack('season_week').reindex(columns=range(1, WEEKS_PER_YEAR + 1))
        pivot = pivot.ffill(axis=1).bfill(axis=1)
        global_mean = pivot.stack().mean()
        pivot = pivot.fillna(global_mean)
        for geocode, row in pivot.iterrows():
            if geocode in geo_index:
                h_bar[col][geo_index[geocode]] = row.values.astype(np.float32)
    return h_bar

@dataclass
class SplitArrays:
    Y_obs: np.ndarray
    obs_mask: np.ndarray
    ifdm: np.ndarray
    log_density: np.ndarray
    population: np.ndarray
    temp_lag: np.ndarray
    humid_lag: np.ndarray
    rain_lag: np.ndarray

def _get_years_in_split(df_split: pd.DataFrame) -> list[int]:
    years = sorted(df_split['season_year'].unique())
    if len(years) == 0:
        raise ValueError('df_split has no epidemic seasons. Cannot build split tensor.')
    return years

def _observed_year_weeks_from_mask(obs_mask: np.ndarray, years_in_split: list[int]) -> list[dict]:
    time_mask = np.asarray(obs_mask, dtype=bool).any(axis=0)
    observed = []
    for yi, year in enumerate(years_in_split):
        for wi, has_obs in enumerate(time_mask[yi]):
            if has_obs:
                observed.append({'season_year': int(year), 'season_week': int(wi + 1)})
    return observed

def _format_observed_span(observed_year_weeks: list[dict]) -> str:
    if not observed_year_weeks:
        return 'no observed weeks'
    first = observed_year_weeks[0]
    last = observed_year_weeks[-1]
    return f"season {first['season_year']}-SW{first['season_week']:02d} to season {last['season_year']}-SW{last['season_week']:02d}"

def _report_split_coverage(df_split: pd.DataFrame, years_in_split: list[int], is_test: bool, weeks_per_year: int) -> None:
    split_name = 'Test' if is_test else 'Train'
    print(f'  {split_name}: {len(years_in_split)} year(s) touched  ({years_in_split[0]}–{years_in_split[-1]})')
    obs_weeks = df_split.groupby(['geocode', 'season_year']).size().unstack('season_year', fill_value=0)
    obs_coverage = obs_weeks.values
    if obs_coverage.size == 0 or obs_coverage.max() == 0:
        min_w = 0
        max_w = 0
    else:
        min_w = obs_coverage[obs_coverage > 0].min()
        max_w = obs_coverage.max()
    partial_msg = '  (partial seasons present)' if min_w < weeks_per_year else ''
    print(f'  Observed weeks per city-season: min={min_w}  max={max_w}' + partial_msg)

def _tile_climatology(h_bar: dict, key: str, n_years: int) -> np.ndarray:
    base = np.asarray(h_bar[key], dtype=np.float32)
    return np.tile(base[:, np.newaxis, :], (1, n_years, 1))

def _allocate_split_arrays(n_cities: int, n_years: int, weeks_per_year: int, h_bar: dict) -> SplitArrays:
    shape3 = (n_cities, n_years, weeks_per_year)
    return SplitArrays(Y_obs=np.zeros(shape3, dtype=np.int32), obs_mask=np.zeros(shape3, dtype=bool), ifdm=np.zeros(shape3, dtype=np.float32), log_density=np.zeros(shape3, dtype=np.float32), population=np.zeros(shape3, dtype=np.float32), temp_lag=_tile_climatology(h_bar, 'temp_lag', n_years), humid_lag=_tile_climatology(h_bar, 'humid_lag', n_years), rain_lag=_tile_climatology(h_bar, 'rain_lag', n_years))

def _fill_observed_rows(arrays, df_split, geocode_order, years_in_split, is_test, weeks_per_year):
    geo_index = {g: i for i, g in enumerate(geocode_order)}
    year_index = {y: j for j, y in enumerate(years_in_split)}
    sub = df_split[df_split['geocode'].isin(geo_index)].copy()
    if sub.empty:
        return
    ci = sub['geocode'].map(geo_index).values
    yi = sub['season_year'].map(year_index).values
    wi = sub['season_week'].values.astype(int) - 1
    valid = (wi >= 0) & (wi < weeks_per_year)
    ci, yi, wi = (ci[valid], yi[valid], wi[valid])
    sub = sub.iloc[valid]
    arrays.Y_obs[ci, yi, wi] = sub['casos'].values
    arrays.obs_mask[ci, yi, wi] = True
    arrays.ifdm[ci, yi, wi] = sub['ifdm'].values
    arrays.log_density[ci, yi, wi] = sub['log_density'].values
    arrays.population[ci, yi, wi] = sub['population'].values
    if not is_test:
        arrays.temp_lag[ci, yi, wi] = sub['temp_lag'].values
        arrays.humid_lag[ci, yi, wi] = sub['humid_lag'].values
        arrays.rain_lag[ci, yi, wi] = sub['rain_lag'].values

def _fill_context_covariate_rows(arrays: SplitArrays, df_context: pd.DataFrame, geocode_order: list, years_in_split: list[int], weeks_per_year: int) -> None:
    geo_index = {g: i for i, g in enumerate(geocode_order)}
    year_index = {y: j for j, y in enumerate(years_in_split)}
    sub = df_context[df_context['geocode'].isin(geo_index) & df_context['season_year'].isin(year_index)].copy()
    if sub.empty:
        return
    ci = sub['geocode'].map(geo_index).values
    yi = sub['season_year'].map(year_index).values
    wi = sub['season_week'].values.astype(int) - 1
    valid = (wi >= 0) & (wi < weeks_per_year)
    ci, yi, wi = (ci[valid], yi[valid], wi[valid])
    sub = sub.iloc[valid]
    arrays.ifdm[ci, yi, wi] = sub['ifdm'].values
    arrays.log_density[ci, yi, wi] = sub['log_density'].values
    arrays.population[ci, yi, wi] = sub['population'].values
    arrays.temp_lag[ci, yi, wi] = sub['temp_lag'].values
    arrays.humid_lag[ci, yi, wi] = sub['humid_lag'].values
    arrays.rain_lag[ci, yi, wi] = sub['rain_lag'].values

def _forward_backward_fill_1d(values: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    filled = values.copy()
    last = None
    for wi, value in enumerate(filled):
        if observed_mask[wi]:
            last = value
        elif last is not None:
            filled[wi] = last
    last = None
    for wi, _ in reversed(tuple(enumerate(filled))):
        if observed_mask[wi]:
            last = filled[wi]
        elif last is not None:
            filled[wi] = last
    return filled

def _fill_missing_city_year_values(arr_3d, obs_mask, df_full, geocode_order, column):
    city_means = df_full.groupby('geocode')[column].mean().reindex(geocode_order).fillna(0.0).values
    for ci, _ in enumerate(geocode_order):
        for yi in range(arr_3d.shape[1]):
            if obs_mask[ci, yi, :].any():
                arr_3d[ci, yi, :] = _forward_backward_fill_1d(arr_3d[ci, yi, :], obs_mask[ci, yi, :])
            else:
                arr_3d[ci, yi, :] = city_means[ci]

def _fill_missing_static_covariates(arrays: SplitArrays, df_full: pd.DataFrame, geocode_order: list) -> None:
    covariates = [(arrays.ifdm, 'ifdm'), (arrays.log_density, 'log_density'), (arrays.population, 'population')]
    for arr_3d, column in covariates:
        _fill_missing_city_year_values(arr_3d=arr_3d, obs_mask=arrays.obs_mask, df_full=df_full, geocode_order=geocode_order, column=column)

def _build_static_city_arrays(df_full, geocode_order):
    static = df_full.dropna(subset=['lat', 'lon', 'koppen_idx', 'state_idx']).drop_duplicates('geocode').set_index('geocode').reindex(geocode_order)
    missing = static[static['koppen_idx'].isna()].index.tolist()
    if missing:
        print(f'  ⚠  {len(missing)} cities missing static fields — filling with mode')
        static['koppen_idx'] = static['koppen_idx'].fillna(static['koppen_idx'].mode().iloc[0])
        static['state_idx'] = static['state_idx'].fillna(static['state_idx'].mode().iloc[0])
    return {'lat': static['lat'].values.astype(np.float32), 'lon': static['lon'].values.astype(np.float32), 'alt': static['alt'].fillna(0.0).values.astype(np.float32), 'koppen_idx': static['koppen_idx'].values.astype(np.int32), 'state_idx': static['state_idx'].values.astype(np.int32)}

def _flatten_city_year_week_array(arr: np.ndarray, n_cities: int, n_years: int, weeks_per_year: int) -> np.ndarray:
    return arr.reshape(n_cities, n_years * weeks_per_year)

def _build_data_dict(arrays: SplitArrays, static: dict, h_bar: dict, geocode_order: list, years_in_split: list[int], encoders: dict, is_test: bool, weeks_per_year: int) -> dict:
    n_cities = len(geocode_order)
    n_years = len(years_in_split)

    def flat(arr: np.ndarray) -> np.ndarray:
        return _flatten_city_year_week_array(arr=arr, n_cities=n_cities, n_years=n_years, weeks_per_year=weeks_per_year)
    state_idx = static['state_idx']
    koppen_idx = static['koppen_idx']
    return {'n_cities': n_cities, 'n_states': int(state_idx.max() + 1), 'n_koppen': int(koppen_idx.max() + 1), 'n_years': n_years, 'weeks_per_year': weeks_per_year, 'ifdm': jnp.array(flat(arrays.ifdm)), 'log_density': jnp.array(flat(arrays.log_density)), 'population': jnp.array(flat(arrays.population)), 'temp_lag': jnp.array(flat(arrays.temp_lag)), 'humid_lag': jnp.array(flat(arrays.humid_lag)), 'rain_lag': jnp.array(flat(arrays.rain_lag)), 'lat': jnp.array(static['lat']), 'lon': jnp.array(static['lon']), 'alt': jnp.array(static['alt']), 'koppen_idx': jnp.array(koppen_idx), 'state_idx': jnp.array(state_idx), 'include_vc_noise': is_test, 'h_bar': {k: jnp.array(v) for k, v in h_bar.items()}, 'geocode_order': geocode_order, 'year_order': years_in_split, 'season_year_order': years_in_split, 'calendar': 'epidemic_season_EW41_52slot', 'epidemic_season_start': EPIDEMIC_SEASON_START, 'obs_mask': jnp.array(arrays.obs_mask), 'encoders': encoders}

def build_split_tensor(df_split: pd.DataFrame, df_full: pd.DataFrame, geocode_order: list, h_bar: dict, is_test: bool, encoders: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    n_cities = len(geocode_order)
    weeks_per_year = WEEKS_PER_YEAR
    years_in_split = _get_years_in_split(df_split)
    n_years = len(years_in_split)
    _report_split_coverage(df_split=df_split, years_in_split=years_in_split, is_test=is_test, weeks_per_year=weeks_per_year)
    arrays = _allocate_split_arrays(n_cities=n_cities, n_years=n_years, weeks_per_year=weeks_per_year, h_bar=h_bar)
    _fill_observed_rows(arrays=arrays, df_split=df_split, geocode_order=geocode_order, years_in_split=years_in_split, is_test=is_test, weeks_per_year=weeks_per_year)
    if is_test:
        _fill_context_covariate_rows(arrays=arrays, df_context=df_full, geocode_order=geocode_order, years_in_split=years_in_split, weeks_per_year=weeks_per_year)
    _fill_missing_static_covariates(arrays=arrays, df_full=df_full, geocode_order=geocode_order)
    static = _build_static_city_arrays(df_full=df_full, geocode_order=geocode_order)
    data_dict = _build_data_dict(arrays=arrays, static=static, h_bar=h_bar, geocode_order=geocode_order, years_in_split=years_in_split, encoders=encoders, is_test=is_test, weeks_per_year=weeks_per_year)
    observed_year_weeks = _observed_year_weeks_from_mask(arrays.obs_mask, years_in_split)
    data_dict['observed_year_weeks'] = observed_year_weeks
    data_dict['n_observed_weeks'] = len(observed_year_weeks)
    data_dict['observed_span'] = _format_observed_span(observed_year_weeks)
    data_dict['context_covariates'] = 'real_source_rows_where_available' if is_test else 'observed_rows'
    data_dict['time_observed_mask'] = jnp.array(arrays.obs_mask.any(axis=0), dtype=bool)
    return (data_dict, arrays.Y_obs, arrays.obs_mask)
ARRAY_KEYS = ['ifdm', 'log_density', 'population', 'temp_lag', 'humid_lag', 'rain_lag', 'lat', 'lon', 'alt', 'koppen_idx', 'state_idx', 'obs_mask']

def _json_safe(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return x

def save_data_bundle(save_dir: Path, name: str, data: dict, Y: np.ndarray, mask: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    arrays_to_save = {key: np.asarray(data[key]) for key in ARRAY_KEYS}
    arrays_to_save['Y'] = np.asarray(Y)
    arrays_to_save['mask'] = np.asarray(mask)
    for k, v in data['h_bar'].items():
        arrays_to_save[f'h_bar__{k}'] = np.asarray(v)
    np.savez_compressed(save_dir / f'{name}_arrays.npz', **arrays_to_save)
    metadata = {'n_cities': data['n_cities'], 'n_states': data['n_states'], 'n_koppen': data['n_koppen'], 'n_years': data['n_years'], 'weeks_per_year': data['weeks_per_year'], 'include_vc_noise': data['include_vc_noise'], 'geocode_order': data['geocode_order'], 'year_order': data['year_order'], 'season_year_order': data.get('season_year_order', data['year_order']), 'calendar': data.get('calendar', 'epidemic_season_EW41_52slot'), 'epidemic_season_start': data.get('epidemic_season_start', EPIDEMIC_SEASON_START), 'encoders': data['encoders'], 'observed_year_weeks': data.get('observed_year_weeks', []), 'observed_season_weeks': data.get('observed_year_weeks', []), 'n_observed_weeks': data.get('n_observed_weeks', 0), 'observed_span': data.get('observed_span', 'no observed weeks'), 'context_covariates': data.get('context_covariates', 'unknown')}
    with open(save_dir / f'{name}_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(_json_safe(metadata), f, indent=2)

def prepare_model_inputs(df, lag=LAG_WEEKS, save_dir=None):
    print('═' * 55 + '\n  Step 1 — log_density\n' + '═' * 55)
    df = add_log_density(df)
    print('\n' + '═' * 55 + '\n  Step 2 — full grid reindex\n' + '═' * 55)
    df = reindex_full_grid(df)
    print('\n' + '═' * 55 + f'\n  Step 3 — lagged covariates           (lag={lag}w, full timeline)\n' + '═' * 55)
    df = add_lagged_covariates(df, lag=lag)
    print('\n' + '═' * 55 + '\n  Step 4 — encode categoricals\n' + '═' * 55)
    df, encoders = encode_categoricals(df)
    df_train = df[df['train']].copy()
    df_test = df[df['test']].copy()
    print('\n' + '═' * 55 + '\n  Split summary\n' + '═' * 55)
    print(f"  Train rows : {len(df_train):,}  ({df_train['geocode'].nunique():,} cities, {df_train['year'].nunique()} years)")
    print(f"  Test  rows : {len(df_test):,}  ({df_test['geocode'].nunique():,} cities, {df_test['year'].nunique()} year(s))")
    train_cities = set(df_train['geocode'].unique())
    test_cities = set(df_test['geocode'].unique())
    common = sorted(train_cities & test_cities)
    only_train = train_cities - test_cities
    only_test = test_cities - train_cities
    if only_train:
        print(f'  ⚠  {len(only_train)} cities only in train (no test rows) — excluded')
    if only_test:
        print(f'  ⚠  {len(only_test)} cities only in test (no train rows) — excluded')
    geocode_order = common
    C = len(geocode_order)
    print(f'  ✓  {C:,} cities in both splits')
    df_train = df_train[df_train['geocode'].isin(geocode_order)]
    df_test = df_test[df_test['geocode'].isin(geocode_order)]
    print('\n' + '═' * 55 + '\n  Step 5 — climatological weather (from train)\n' + '═' * 55)
    h_bar = compute_h_bar(df_train, geocode_order)
    for col, arr in h_bar.items():
        print(f'  {col:<12} range [{arr.min():.1f}, {arr.max():.1f}]')
    print('\n' + '═' * 55 + '\n  Step 6 — training tensors\n' + '═' * 55)
    data_train, Y_train_np, mask_train_np = build_split_tensor(df_train, df, geocode_order, h_bar, is_test=False, encoders=encoders)
    print('\n' + '═' * 55 + '\n  Step 7 — test tensors\n' + '═' * 55)
    data_test, Y_test_np, mask_test_np = build_split_tensor(df_test, df, geocode_order, h_bar, is_test=True, encoders=encoders)
    Y_train = jnp.array(Y_train_np)
    Y_test = jnp.array(Y_test_np)
    mask_train = jnp.array(mask_train_np)
    mask_test = jnp.array(mask_test_np)
    print('\n' + '═' * 55 + '\n  Final checks\n' + '═' * 55)
    for name, d in [('train', data_train), ('test', data_test)]:
        for col in ['temp_lag', 'humid_lag', 'rain_lag', 'ifdm', 'log_density']:
            assert not jnp.isnan(d[col]).any(), f'NaN in {name}/{col}'
        grid_weeks = d['n_years'] * d['weeks_per_year']
        observed_weeks = int(d.get('n_observed_weeks', grid_weeks))
        observed_span = d.get('observed_span', 'unknown')
        print(f"  {name}: grid [{d['n_cities']:,}, {d['n_years']}, {d['weeks_per_year']}] = {grid_weeks} slots; observed weeks={observed_weeks} ({observed_span})  NaN-free ✓")
    print(f'  obs_mask coverage — train: {float(mask_train.mean()) * 100:.1f}%  test: {float(mask_test.mean()) * 100:.1f}%')
    if save_dir:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        save_data_bundle(save_dir=p, name='train', data=data_train, Y=Y_train_np, mask=mask_train_np)
        save_data_bundle(save_dir=p, name='test', data=data_test, Y=Y_test_np, mask=mask_test_np)
        print(f"\n  Saved full preprocessing bundle to '{save_dir}/'")
    print('\n  ✓ All tensors ready.\n')
    return (data_train, data_test, Y_train, Y_test, mask_train, mask_test)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess joined dengue/weather/ifdm data into model-ready tensors.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-path', type=Path, default=Path('outputs/data.parquet'), help='Path to the master-joined parquet file.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/preprocessed_data'), help='Directory where tensor .npy files will be written.')
    parser.add_argument('--lag-weeks', type=int, default=LAG_WEEKS, help='Number of weeks to lag weather covariates.')
    return parser.parse_args()

def load_input(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        print(f'ERROR: input file not found: {input_path}', file=sys.stderr)
        sys.exit(1)
    print(f'Loading {input_path} …')
    df = pd.read_parquet(input_path)
    print(f'  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns')
    required = {'geocode', 'year', 'week', 'casos', 'train', 'test', 'uf', 'lat', 'lon', 'constructed_area_m2', 'population', 'temp_med', 'humid_med', 'precip_med', 'koppen', 'alt', 'ifdm'}
    missing = required - set(df.columns)
    if missing:
        print(f'ERROR: missing required columns: {sorted(missing)}', file=sys.stderr)
        sys.exit(1)
    return df
if __name__ == '__main__':
    args = parse_args()
    input_df = load_input(args.input_path)
    prepare_model_inputs(input_df, lag=args.lag_weeks, save_dir=args.output_dir)
    print('Done.')
