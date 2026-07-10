from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
EARTH_RADIUS_KM = 6371.0088

class UnionFind:

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = (self.find(a), self.find(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = (rb, ra)
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Merge small nearby municipalities into modeling units.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-path', type=Path, required=True, help='Input master parquet/csv.')
    p.add_argument('--output-path', type=Path, required=True, help='Merged master parquet/csv.')
    p.add_argument('--mapping-output', type=Path, required=True, help='CSV mapping original geocode -> model geocode/group.')
    p.add_argument('--small-pop-threshold', type=float, default=20000.0)
    p.add_argument('--max-distance-km', type=float, default=30.0)
    p.add_argument('--allow-koppen-mismatch', action='store_true', help='Allow merges within state even when Koppen classes differ.')
    p.add_argument('--population-column', default='population', help='Population column used to classify small cities and compute weights.')
    return p.parse_args()

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in {'.csv', '.gz'} or path.name.endswith('.csv.gz'):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f'Unsupported input extension: {path}')

def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.parquet':
        df.to_parquet(path, index=False)
    elif path.suffix in {'.csv', '.gz'} or path.name.endswith('.csv.gz'):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f'Unsupported output extension: {path}')

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def mode_or_first(s: pd.Series):
    mode = s.dropna().mode()
    return mode.iloc[0] if not mode.empty else s.dropna().iloc[0] if s.dropna().size else np.nan

def build_city_table(df: pd.DataFrame, population_column: str) -> pd.DataFrame:
    required = {'geocode', 'uf', 'koppen', 'lat', 'lon', population_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Input is missing required columns: {sorted(missing)}')
    city = df.groupby('geocode', as_index=False).agg(uf=('uf', mode_or_first), koppen=('koppen', mode_or_first), lat=('lat', 'median'), lon=('lon', 'median'), ref_population=(population_column, 'median')).dropna(subset=['lat', 'lon', 'uf', 'koppen', 'ref_population'])
    city['geocode'] = city['geocode'].astype('int64')
    city['ref_population'] = pd.to_numeric(city['ref_population'], errors='coerce').clip(lower=1.0)
    return city.sort_values('geocode').reset_index(drop=True)

def compatible(a, b, allow_koppen_mismatch: bool) -> bool:
    if a['uf'] != b['uf']:
        return False
    if not allow_koppen_mismatch and a['koppen'] != b['koppen']:
        return False
    return True

def build_groups(city: pd.DataFrame, small_pop_threshold: float, max_distance_km: float, allow_koppen_mismatch: bool):
    geocodes = city['geocode'].to_numpy(dtype=np.int64)
    uf = city['uf'].astype(str).to_numpy()
    koppen = city['koppen'].astype(str).to_numpy()
    pop = city['ref_population'].to_numpy(dtype=float)
    lat = city['lat'].to_numpy(dtype=float)
    lon = city['lon'].to_numpy(dtype=float)
    small = pop <= small_pop_threshold
    uf_groups: dict[str, list[int]] = {}
    for i, val in enumerate(uf):
        uf_groups.setdefault(val, []).append(i)
    ufinder = UnionFind(geocodes.tolist())
    links = []
    order = sorted(range(len(city)), key=lambda i: (not small[i], pop[i], geocodes[i]))
    for i in order:
        if not small[i]:
            continue
        candidates = []
        for j in uf_groups[uf[i]]:
            if i == j:
                continue
            if not allow_koppen_mismatch and koppen[i] != koppen[j]:
                continue
            dist = float(haversine_km(lat[i], lon[i], lat[j], lon[j]))
            if dist <= max_distance_km:
                candidates.append((not small[j], dist, pop[j], geocodes[j], j))
        if not candidates:
            continue
        _, dist, _, target_geocode, j = min(candidates)
        ufinder.union(int(geocodes[i]), int(target_geocode))
        links.append({'source_geocode': int(geocodes[i]), 'target_geocode': int(target_geocode), 'distance_km': dist, 'target_is_small': bool(small[j])})
    city = city.copy()
    city['raw_group'] = [ufinder.find(int(g)) for g in geocodes]
    anchors = {}
    for raw_group, sub in city.groupby('raw_group'):
        anchor = sub.sort_values(['ref_population', 'geocode'], ascending=[False, True]).iloc[0]
        anchors[raw_group] = int(anchor['geocode'])
    city['model_geocode'] = city['raw_group'].map(anchors).astype('int64')
    group_pop = city.groupby('model_geocode')['ref_population'].sum().rename('group_ref_population')
    city = city.merge(group_pop, on='model_geocode', how='left')
    city['disagg_population_share'] = city['ref_population'] / city['group_ref_population'].clip(lower=1.0)
    city['is_small'] = city['ref_population'] <= small_pop_threshold
    city['is_merged'] = city.groupby('model_geocode')['geocode'].transform('size') > 1
    links_df = pd.DataFrame(links)
    return (city, links_df)

def aggregate_master(df: pd.DataFrame, mapping: pd.DataFrame, population_column: str) -> pd.DataFrame:
    work = df.merge(mapping[['geocode', 'model_geocode']], on='geocode', how='left', validate='many_to_one')
    if work['model_geocode'].isna().any():
        missing = work.loc[work['model_geocode'].isna(), 'geocode'].unique()[:10]
        raise ValueError(f'Some geocodes have no merge mapping: {missing}')
    work['model_geocode'] = work['model_geocode'].astype('int64')
    key_cols = ['model_geocode', 'year', 'week']
    if 'date' in work.columns:
        key_cols.append('date')
    if 'epiweek' in work.columns:
        key_cols.append('epiweek')
    sum_cols = [c for c in ['casos', population_column, 'constructed_area_m2'] if c in work.columns]
    bool_cols = [c for c in ['train', 'test'] if c in work.columns]
    preserve_cols = [c for c in ['uf', 'koppen'] if c in work.columns]
    weighted_cols = [c for c in work.select_dtypes(include=[np.number]).columns if c not in set(key_cols + sum_cols + bool_cols + ['geocode', 'model_geocode'])]
    weight = work[population_column].astype(float).clip(lower=1.0) if population_column in work.columns else 1.0
    for col in weighted_cols:
        work[f'__w_{col}'] = work[col].astype(float) * weight
    agg = {'geocode': ('model_geocode', 'first')}
    for col in sum_cols:
        agg[col] = (col, 'sum')
    for col in bool_cols:
        agg[col] = (col, 'first')
    for col in preserve_cols:
        agg[col] = (col, 'first')
    for col in weighted_cols:
        agg[f'__w_{col}'] = (f'__w_{col}', 'sum')
    if population_column in work.columns:
        agg['__weight_sum'] = (population_column, 'sum')
    out = work.groupby(key_cols, as_index=False).agg(**agg)
    if population_column in work.columns:
        denom = out['__weight_sum'].astype(float).clip(lower=1.0)
        for col in weighted_cols:
            out[col] = out.pop(f'__w_{col}') / denom
        out = out.drop(columns=['__weight_sum'])
    out = out.drop(columns=['model_geocode'])
    out['geocode'] = out['geocode'].astype('int64')
    ordered = [c for c in df.columns if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    out = out[ordered + extras].sort_values(['geocode', 'year', 'week']).reset_index(drop=True)
    return out

def validate_groups(mapping: pd.DataFrame, allow_koppen_mismatch: bool) -> None:
    bad_state = []
    bad_koppen = []
    for model_geocode, sub in mapping.groupby('model_geocode'):
        if sub['uf'].nunique(dropna=False) > 1:
            bad_state.append(model_geocode)
        if not allow_koppen_mismatch and sub['koppen'].nunique(dropna=False) > 1:
            bad_koppen.append(model_geocode)
    if bad_state:
        raise ValueError(f'Merge groups crossing states: {bad_state[:10]}')
    if bad_koppen:
        raise ValueError(f'Merge groups crossing Koppen classes: {bad_koppen[:10]}')

def main() -> None:
    args = parse_args()
    df = read_table(args.input_path)
    city = build_city_table(df, args.population_column)
    mapping, links = build_groups(city, small_pop_threshold=args.small_pop_threshold, max_distance_km=args.max_distance_km, allow_koppen_mismatch=args.allow_koppen_mismatch)
    validate_groups(mapping, allow_koppen_mismatch=args.allow_koppen_mismatch)
    merged = aggregate_master(df, mapping, population_column=args.population_column)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    write_table(merged, args.output_path)
    mapping_out = mapping.rename(columns={'geocode': 'original_geocode'}).copy()
    if not links.empty:
        mapping_out = mapping_out.merge(links.rename(columns={'source_geocode': 'original_geocode'}), on='original_geocode', how='left')
    mapping_out.to_csv(args.mapping_output, index=False)
    n_groups = mapping['model_geocode'].nunique()
    n_cities = mapping['geocode'].nunique()
    n_merged_cities = int(mapping['is_merged'].sum())
    n_small = int(mapping['is_small'].sum())
    n_small_merged = int((mapping['is_small'] & mapping['is_merged']).sum())
    print('Small-city merge complete')
    print(f'  input rows          : {len(df):,}')
    print(f'  output rows         : {len(merged):,}')
    print(f'  original cities     : {n_cities:,}')
    print(f'  modeling units      : {n_groups:,}')
    print(f'  small cities        : {n_small:,}')
    print(f'  small cities merged : {n_small_merged:,}')
    print(f'  all merged cities   : {n_merged_cities:,}')
    print(f'  output              : {args.output_path}')
    print(f'  mapping             : {args.mapping_output}')
if __name__ == '__main__':
    main()
