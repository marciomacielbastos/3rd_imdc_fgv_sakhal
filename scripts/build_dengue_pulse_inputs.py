from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from merge_small_city_units import aggregate_master, build_city_table, build_groups, validate_groups
WEEKS_PER_YEAR = 52
CLIMATE_PLACEHOLDER_COLUMNS = {'temp_min': 0.0, 'temp_med': 0.0, 'temp_max': 0.0, 'precip_min': 0.0, 'precip_med': 0.0, 'precip_max': 0.0, 'humid_min': 0.0, 'humid_med': 0.0, 'humid_max': 0.0, 'rainy_days': 0.0}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build master parquet and tensor bundles for dengue_pulse.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dengue-path', type=Path, default=Path('data/dengue.csv.gz'))
    parser.add_argument('--population-path', type=Path, default=Path('data/datasus_population_2001_2025.csv.gz'))
    parser.add_argument('--geo-path', type=Path, default=Path('outputs/geospatial_results/built_stats_municipios.parquet'), help='Static geospatial output from 1_compute_built_area_and_centroids_by_uf.py.')
    parser.add_argument('--ifdm-weekly-path', type=Path, default=Path('outputs/ifdm_results/ifdm_weekly_2002_2025.parquet'), help='Supplemental weekly IFDM table. Rebuilt separately when raw IFDM changes.')
    parser.add_argument('--koppen-static-path', type=Path, default=Path('outputs/weather_results/weather_geo_koppen.parquet'), help='Any table with geocode, koppen, alt. Weather columns are ignored.')
    parser.add_argument('--split', type=int, default=4, choices=(1, 2, 3, 4), help='Use train_<split>/target_<split> from dengue.csv.gz.')
    parser.add_argument('--all-splits', action='store_true', help='Build split-specific preprocessing bundles for validation splits 1..4 under --challenge-work-dir. This avoids overwriting the single-split default output paths.')
    parser.add_argument('--challenge-work-dir', type=Path, default=Path('outputs/challenge_pulse'), help='Root directory used by --all-splits for validation_i outputs.')
    parser.add_argument('--master-output', type=Path, default=Path('outputs/data_pulse.parquet'), help='Joined long table consumed by 7_prepare_tensors.py.')
    parser.add_argument('--population-weekly-output', type=Path, default=Path('outputs/population_results/datasus_population_weekly_pulse.parquet'))
    parser.add_argument('--tensor-output-dir', type=Path, default=Path('outputs/preprocessed_data_pulse'))
    parser.add_argument('--prepare-script', type=Path, default=Path('scripts/7_prepare_tensors.py'), help='Existing tensor builder. Called after master parquet is written.')
    parser.add_argument('--include-prior-targets', action='store_true', help='For split i, add target_1..target_{i-1} rows to the training mask. This is cumulative and retained for backwards compatibility.')
    parser.add_argument('--include-previous-target', action='store_true', help='For split i>1, add only target_{i-1} rows to the training mask. Use this for one-step sequential validation updates.')
    parser.add_argument('--skip-tensors', action='store_true', help='Only write the joined master parquet and weekly population table.')
    parser.add_argument('--merge-small-cities', action='store_true', help='Merge small nearby municipalities into modeling units before tensor construction.')
    parser.add_argument('--small-pop-threshold', type=float, default=20000.0)
    parser.add_argument('--max-merge-distance-km', type=float, default=30.0)
    parser.add_argument('--allow-merge-koppen-mismatch', action='store_true', help='Allow small-city merges within state even when Koppen classes differ.')
    parser.add_argument('--merge-mapping-output', type=Path, default=None, help='CSV mapping original geocode to merged modeling unit. Defaults beside --master-output.')
    return parser.parse_args()

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f'{label} not found: {path}')

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^\\w_]', '', regex=True)
    return df

def load_dengue(path: Path, split: int, include_prior_targets: bool=False, include_previous_target: bool=False) -> pd.DataFrame:
    require_file(path, 'Dengue file')
    train_col = f'train_{split}'
    target_col = f'target_{split}'
    if include_prior_targets and include_previous_target:
        raise ValueError('Use only one of --include-prior-targets or --include-previous-target')
    if include_previous_target and split == 1:
        prior_target_cols = []
    elif include_previous_target:
        prior_target_cols = [f'target_{split - 1}']
    else:
        prior_target_cols = [f'target_{i}' for i in range(1, split)] if include_prior_targets else []
    print(f'Loading dengue cases: {path}  (split={split})')
    df = pd.read_csv(path, compression='gzip', low_memory=False)
    if 'disease' in df.columns:
        df = df[df['disease'].astype(str).str.lower().eq('dengue')].copy()
    required = {'date', 'epiweek', 'geocode', 'casos', train_col, target_col, *prior_target_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Dengue file is missing columns: {sorted(missing)}')
    cols = ['date', 'epiweek', 'geocode', 'casos', train_col, target_col] + prior_target_cols
    out = df[cols].copy()
    out = out.rename(columns={train_col: 'train', target_col: 'test'})
    if prior_target_cols:
        prior_mask = out[prior_target_cols].fillna(False).astype(bool).any(axis=1)
        out['train'] = out['train'].fillna(False).astype(bool) | prior_mask
        out = out.drop(columns=prior_target_cols)
        label = 'previous-target' if include_previous_target else 'prior-target'
        print(f'  sequential update: added {int(prior_mask.sum()):,} {label} rows to train')
    out['date'] = pd.to_datetime(out['date'])
    out['epiweek'] = out['epiweek'].astype('int64')
    out['year'] = (out['epiweek'] // 100).astype('int64')
    out['week'] = (out['epiweek'] % 100).astype('int64')
    out = out[out['week'].between(1, WEEKS_PER_YEAR)].copy()
    out['geocode'] = out['geocode'].astype('int64')
    out['casos'] = out['casos'].fillna(0).clip(lower=0).round().astype('int64')
    out['train'] = out['train'].fillna(False).astype(bool)
    out['test'] = out['test'].fillna(False).astype(bool)
    print(f"  dengue rows={len(out):,}; cities={out['geocode'].nunique():,}; years={out['year'].min()}-{out['year'].max()}")
    print(f"  train rows={int(out['train'].sum()):,}; target rows={int(out['test'].sum()):,}")
    return out

def load_geo(path: Path) -> pd.DataFrame:
    require_file(path, 'Geospatial parquet')
    print(f'Loading static geospatial data: {path}')
    geo = pd.read_parquet(path)
    geo = geo[['city_idx', 'SIGLA_UF', 'built_centroid_lat', 'built_centroid_lon', 'constructed_area_m2']].drop_duplicates('city_idx')
    geo = geo.rename(columns={'city_idx': 'geocode', 'SIGLA_UF': 'uf', 'built_centroid_lat': 'lat', 'built_centroid_lon': 'lon'})
    geo['geocode'] = geo['geocode'].astype('int64')
    print(f'  geo cities={len(geo):,}')
    return geo

def load_static_koppen(path: Path, geo: pd.DataFrame) -> pd.DataFrame:
    require_file(path, 'Koppen/altitude static table')
    print(f'Loading static Koppen/altitude data: {path}')
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    df = standardize_columns(df)
    rename = {'ibgecode': 'geocode', 'ibge_code': 'geocode', 'altitude': 'alt'}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if 'geocode' not in df.columns or 'koppen' not in df.columns:
        raise ValueError(f'{path} must contain geocode and koppen columns')
    if 'alt' not in df.columns:
        df['alt'] = np.nan
    out = df[['geocode', 'koppen', 'alt']].dropna(subset=['geocode', 'koppen']).copy()
    out['geocode'] = out['geocode'].astype('int64')
    out = out.drop_duplicates('geocode')
    full = geo[['geocode', 'lat', 'lon']].merge(out, on='geocode', how='left')
    missing = full['koppen'].isna()
    if missing.any():
        from sklearn.neighbors import BallTree
        ref = full[~missing].dropna(subset=['lat', 'lon'])
        targets = full[missing].dropna(subset=['lat', 'lon'])
        if ref.empty or targets.empty:
            raise ValueError('Cannot impute missing Koppen values; no coordinate reference.')
        tree = BallTree(np.radians(ref[['lat', 'lon']].to_numpy()), metric='haversine')
        _, idx = tree.query(np.radians(targets[['lat', 'lon']].to_numpy()), k=1)
        donor = ref.iloc[idx[:, 0]][['koppen', 'alt']].reset_index(drop=True)
        full.loc[targets.index, 'koppen'] = donor['koppen'].to_numpy()
        full.loc[targets.index, 'alt'] = donor['alt'].to_numpy()
        print(f'  imputed Koppen/alt for {len(targets):,} cities by nearest centroid')
    full['alt'] = pd.to_numeric(full['alt'], errors='coerce').fillna(0.0)
    return full[['geocode', 'koppen', 'alt']]

def extend_annual_population(pop: pd.DataFrame, max_year: int) -> pd.DataFrame:
    pop = pop.sort_values(['geocode', 'year']).copy()
    rows = [pop]
    current_max = int(pop['year'].max())
    if current_max >= max_year:
        return pop
    print(f'  extending annual population from {current_max} to {max_year}')
    base = pop.copy()
    for year in range(current_max + 1, max_year + 1):
        latest = base.sort_values(['geocode', 'year']).groupby('geocode').tail(1).copy()
        hist = base.copy()
        hist['log_pop'] = np.log(hist['population'].clip(lower=1.0))
        hist['growth'] = hist.groupby('geocode')['log_pop'].diff()
        recent = hist.groupby('geocode').tail(3)
        growth = recent.groupby(recent['geocode'])['growth'].mean()
        fallback = float(hist['growth'].median()) if hist['growth'].notna().any() else 0.0
        lo, hi = hist['growth'].quantile([0.005, 0.995]).fillna(0.0)
        latest_growth = latest['geocode'].map(growth).fillna(fallback).clip(lower=lo, upper=hi)
        latest['population'] = np.exp(np.log(latest['population'].clip(lower=1.0)) + latest_growth)
        latest['population'] = latest['population'].round().clip(lower=1).astype('int64')
        latest['year'] = year
        rows.append(latest[['geocode', 'year', 'population']])
        base = pd.concat([base, rows[-1]], ignore_index=True)
    return pd.concat(rows, ignore_index=True).drop_duplicates(['geocode', 'year'], keep='last')

def build_weekly_population(path: Path, output_path: Path, max_year: int) -> pd.DataFrame:
    require_file(path, 'Population file')
    print(f'Loading annual population: {path}')
    pop = pd.read_csv(path, compression='gzip', low_memory=False)
    pop = standardize_columns(pop)
    required = {'geocode', 'year', 'population'}
    missing = required - set(pop.columns)
    if missing:
        raise ValueError(f'Population file is missing columns: {sorted(missing)}')
    pop = pop[['geocode', 'year', 'population']].copy()
    pop['geocode'] = pop['geocode'].astype('int64')
    pop['year'] = pop['year'].astype('int64')
    pop['population'] = pd.to_numeric(pop['population'], errors='coerce').clip(lower=1)
    pop = pop.dropna(subset=['population']).drop_duplicates(['geocode', 'year'], keep='last')
    pop['population'] = pop['population'].round().astype('int64')
    pop = extend_annual_population(pop, max_year=max_year)
    annual = pop.sort_values(['geocode', 'year']).copy()
    annual['log_pop'] = np.log(annual['population'].clip(lower=1.0))
    annual['log_pop_prev'] = annual.groupby('geocode')['log_pop'].shift(1)
    interp = annual.dropna(subset=['log_pop_prev']).copy()
    weeks = pd.DataFrame({'week': np.arange(1, WEEKS_PER_YEAR + 1, dtype=int)})
    weekly = interp.merge(weeks, how='cross')
    weekly['alpha'] = weekly['week'] / float(WEEKS_PER_YEAR)
    weekly['population'] = np.exp((1.0 - weekly['alpha']) * weekly['log_pop_prev'] + weekly['alpha'] * weekly['log_pop'])
    weekly['population'] = weekly['population'].round().clip(lower=1).astype('int64')
    weekly = weekly[['geocode', 'year', 'week', 'population']].sort_values(['geocode', 'year', 'week'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(output_path, index=False)
    print(f'  weekly population rows={len(weekly):,}; saved {output_path}')
    return weekly

def load_ifdm_weekly(path: Path, required_years: list[int], geocodes: pd.Series) -> pd.DataFrame:
    require_file(path, 'Weekly IFDM table. Build/provide this supplemental file before running pulse preprocessing')
    print(f'Loading weekly IFDM supplemental table: {path}')
    df = pd.read_parquet(path)
    df = standardize_columns(df)
    required = {'geocode', 'year', 'week', 'ifdm'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'IFDM table is missing columns: {sorted(missing)}')
    df = df[['geocode', 'year', 'week', 'ifdm']].copy()
    df['geocode'] = df['geocode'].astype('int64')
    df['year'] = df['year'].astype('int64')
    df['week'] = df['week'].astype('int64')
    df['ifdm'] = pd.to_numeric(df['ifdm'], errors='coerce')
    min_year, max_year = (min(required_years), max(required_years))
    df = df[df['year'].between(min_year, min(max_year, int(df['year'].max())))].copy()
    have_years = set(df['year'].unique())
    missing_years = [y for y in required_years if y not in have_years]
    if missing_years:
        print(f'  extending IFDM by carry-forward for years: {missing_years}')
        last = df.sort_values(['geocode', 'year', 'week']).groupby('geocode').tail(1)
        grid = pd.MultiIndex.from_product([geocodes.astype('int64').unique(), missing_years, range(1, WEEKS_PER_YEAR + 1)], names=['geocode', 'year', 'week']).to_frame(index=False)
        last_map = last.set_index('geocode')['ifdm']
        fallback = float(df['ifdm'].median()) if df['ifdm'].notna().any() else 0.5
        grid['ifdm'] = grid['geocode'].map(last_map).fillna(fallback)
        df = pd.concat([df, grid], ignore_index=True)
    return df[df['year'].isin(required_years) & df['week'].between(1, WEEKS_PER_YEAR)]

def add_climate_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, value in CLIMATE_PLACEHOLDER_COLUMNS.items():
        df[col] = value
    return df

def build_master(args: argparse.Namespace) -> pd.DataFrame:
    dengue = load_dengue(args.dengue_path, split=args.split, include_prior_targets=args.include_prior_targets, include_previous_target=args.include_previous_target)
    required_years = sorted(dengue['year'].unique())
    max_year = max(required_years)
    geo = load_geo(args.geo_path)
    koppen = load_static_koppen(args.koppen_static_path, geo=geo)
    weekly_pop = build_weekly_population(args.population_path, output_path=args.population_weekly_output, max_year=max_year)
    ifdm = load_ifdm_weekly(args.ifdm_weekly_path, required_years, dengue['geocode'])
    print('Merging dengue + static geo ...')
    df = dengue.merge(geo, on='geocode', how='left', validate='many_to_one')
    df = df.merge(koppen, on='geocode', how='left', validate='many_to_one')
    print('Merging weekly population ...')
    df = df.merge(weekly_pop, on=['geocode', 'year', 'week'], how='left', validate='many_to_one')
    print('Merging weekly IFDM ...')
    df = df.merge(ifdm, on=['geocode', 'year', 'week'], how='left', validate='many_to_one')
    df = add_climate_placeholders(df)
    for col in ['population', 'ifdm', 'constructed_area_m2', 'lat', 'lon', 'alt']:
        missing = int(df[col].isna().sum())
        if missing:
            print(f'  filling {missing:,} residual NaNs in {col}')
            if col in {'population', 'ifdm'}:
                city_mean = df.groupby('geocode')[col].transform('mean')
                df[col] = df[col].fillna(city_mean)
            df[col] = df[col].fillna(df[col].median())
    if df['koppen'].isna().any():
        df['koppen'] = df['koppen'].fillna(df['koppen'].mode().iloc[0])
    if df['uf'].isna().any():
        raise ValueError('Some dengue geocodes are missing UF/geospatial rows; rebuild geospatial inputs.')
    df = df.sort_values(['geocode', 'year', 'week']).reset_index(drop=True)
    df = apply_small_city_merge(df, args)
    df = df.sort_values(['geocode', 'year', 'week']).reset_index(drop=True)
    args.master_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.master_output, index=False)
    print(f'Saved pulse master table: {args.master_output}  rows={len(df):,} cols={df.shape[1]}')
    print(f"  train rows={int(df['train'].sum()):,}; target rows={int(df['test'].sum()):,}; cities={df['geocode'].nunique():,}")
    return df

def apply_small_city_merge(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.merge_small_cities:
        return df
    mapping_path = args.merge_mapping_output or args.master_output.with_name('merge_mapping.csv')
    print('Merging small nearby municipalities into modeling units ...')
    city = build_city_table(df, 'population')
    mapping, links = build_groups(city, small_pop_threshold=args.small_pop_threshold, max_distance_km=args.max_merge_distance_km, allow_koppen_mismatch=args.allow_merge_koppen_mismatch)
    validate_groups(mapping, allow_koppen_mismatch=args.allow_merge_koppen_mismatch)
    merged = aggregate_master(df, mapping, population_column='population')
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_out = mapping.rename(columns={'geocode': 'original_geocode'}).copy()
    if not links.empty:
        mapping_out = mapping_out.merge(links.rename(columns={'source_geocode': 'original_geocode'}), on='original_geocode', how='left')
    mapping_out.to_csv(mapping_path, index=False)
    n_groups = mapping['model_geocode'].nunique()
    n_cities = mapping['geocode'].nunique()
    n_small = int(mapping['is_small'].sum())
    n_small_merged = int((mapping['is_small'] & mapping['is_merged']).sum())
    print(f'  cities {n_cities:,} -> modeling units {n_groups:,}; small merged {n_small_merged:,}/{n_small:,}')
    print(f'  merge mapping saved: {mapping_path}')
    return merged

def run_tensor_builder(args: argparse.Namespace) -> None:
    require_file(args.prepare_script, 'Tensor preparation script')
    cmd = [sys.executable, str(args.prepare_script), '--input-path', str(args.master_output), '--output-dir', str(args.tensor_output_dir), '--lag-weeks', '0']
    print('Running tensor builder:')
    print('  ' + ' '.join(cmd))
    env = os.environ.copy()
    env.setdefault('JAX_PLATFORM_NAME', 'cpu')
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    subprocess.run(cmd, check=True, env=env)

def build_one_split(args: argparse.Namespace) -> None:
    build_master(args)
    if not args.skip_tensors:
        run_tensor_builder(args)

def split_args(args: argparse.Namespace, split: int) -> argparse.Namespace:
    split_root = args.challenge_work_dir / f'validation_{split}'
    return argparse.Namespace(**{**vars(args), 'split': split, 'master_output': split_root / 'data_pulse.parquet', 'population_weekly_output': split_root / 'population_weekly.parquet', 'tensor_output_dir': split_root / 'preprocessed', 'merge_mapping_output': split_root / 'merge_mapping.csv' if args.merge_mapping_output is None else args.merge_mapping_output})

def build_all_splits(args: argparse.Namespace) -> None:
    print(f'Building all validation splits under {args.challenge_work_dir}')
    for split in (1, 2, 3, 4):
        print('\n' + '=' * 72)
        print(f'Validation split {split}: train_{split} -> target_{split}')
        print('=' * 72)
        build_one_split(split_args(args, split))

def main() -> None:
    args = parse_args()
    if args.all_splits:
        build_all_splits(args)
        print('Done. Split-specific outputs are under:')
        print(f'  {args.challenge_work_dir}/validation_*/preprocessed')
        return
    build_one_split(args)
    print('Done. Use:')
    print(f'  python dengue_pulse/run_model.py --data-dir {args.tensor_output_dir}')
if __name__ == '__main__':
    main()
