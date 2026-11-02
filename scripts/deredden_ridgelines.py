"""Remove the foreground-dust offset from the HUGS RGB ridgelines.

For every cluster with a Harris (2010) E(B-V), compute the extinction A_X of each
HST filter by synthetic photometry (red-giant model spectrum x Milky Way
extinction law x filter curve), then

    colour_0(X-Y)   = colour_obs(X-Y) - (A_X - A_Y)
    M_0(band)       = M_obs(band)     - A_band

Output keeps the schema of `data/GC Ridgelines.zip` (low_conf is carried over), in
`data/GC_Ridgelines_dereddened/` and `data/GC Ridgelines dereddened.zip`, plus
`data/GC_extinction_per_cluster.csv` (E(B-V) and A_X per cluster).

Requires: numpy, pandas, astropy, dust_extinction  (pip install astropy dust_extinction)
Inputs  : data/extinction/filters/*.dat       SVO filter curves (wavelength [A], transmission)
          data/extinction/spectra/*.fits      Castelli & Kurucz 2004, [M/H]=-1.5, logg=2.0
          data/GC_reddening_harris.csv        E(B-V) per cluster
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from dust_extinction.parameter_averages import F99, G23

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FILTERS = {'F275W': 'HST_WFC3_UVIS1.F275W', 'F336W': 'HST_WFC3_UVIS1.F336W', 'F438W': 'HST_WFC3_UVIS1.F438W',
           'F606W': 'HST_ACS_WFC.F606W', 'F814W': 'HST_ACS_WFC.F814W'}
BANDS = list(FILTERS)


def load_filters():
    return {b: np.loadtxt(DATA / 'extinction' / 'filters' / f'{n}.dat') for b, n in FILTERS.items()}


def load_spectrum(teff):
    h = fits.open(DATA / 'extinction' / 'spectra' / f'ckm15_{teff}.fits')[1].data
    return h['WAVELENGTH'].astype(float), h['g20'].astype(float)


def extinction_mag(law, ebv, spectra, filters, band):
    """A_band [mag] for a given E(B-V), averaged over the model spectra (photon counting)."""
    w, t = filters[band][:, 0], filters[band][:, 1]
    out = []
    for wl, fl in spectra:
        f = np.interp(w, wl, fl)
        trans = law.extinguish(w * u.AA, Ebv=ebv)
        out.append(-2.5 * np.log10(np.trapezoid(f * t * trans * w, w) / np.trapezoid(f * t * w, w)))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--law', choices=['F99', 'G23'], default='F99')
    ap.add_argument('--rv', type=float, default=3.1)
    ap.add_argument('--teff', type=int, nargs='+', default=[4500, 5000], help='model-spectrum Teff grid to average over (K)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    law = {'F99': F99, 'G23': G23}[args.law](Rv=args.rv)
    filters = load_filters()
    spectra = [load_spectrum(t) for t in args.teff]
    ebv = pd.read_csv(DATA / 'GC_reddening_harris.csv').set_index('NGC')['EBV']

    A = {int(n): {b: extinction_mag(law, e, spectra, filters, b) for b in BANDS} for n, e in ebv.items()}
    per_cluster = pd.DataFrame(A).T.rename(columns=lambda b: f'A_{b}')
    per_cluster.insert(0, 'EBV', ebv)
    per_cluster.index.name = 'NGC'
    per_cluster.round(4).to_csv(DATA / f'GC_extinction_per_cluster{args.tag}.csv')

    out_dir = DATA / f'GC_Ridgelines_dereddened{args.tag}'
    out_dir.mkdir(exist_ok=True)
    zip_path = DATA / f'GC Ridgelines dereddened{args.tag}.zip'
    n_files, n_skipped = 0, []
    with zipfile.ZipFile(DATA / 'GC Ridgelines.zip') as zin, zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            m = re.search(r'NGC(\d+)_(F\d+W)_ridge\.csv$', name)
            if not m:
                continue
            ngc, band = int(m.group(1)), m.group(2)
            if ngc not in A:
                n_skipped.append(ngc)
                continue
            d = pd.read_csv(io.BytesIO(zin.read(name)))
            for c in [c for c in d.columns if re.fullmatch(r'F\d+W-F\d+W', c)]:
                x, y = c.split('-')
                d[c] = d[c] - (A[ngc][x] - A[ngc][y])
            d[f'{band}_abs'] = d[f'{band}_abs'] - A[ngc][band]
            fname = Path(name).name
            d.round(4).to_csv(out_dir / fname, index=False)
            zout.writestr(f'GC Ridgelines dereddened{args.tag}/{fname}', d.round(4).to_csv(index=False))
            n_files += 1
    print(f'law={args.law} Rv={args.rv} Teff={args.teff}: wrote {n_files} ridge files to {out_dir.name}/ and {zip_path.name}')
    if n_skipped:
        print('skipped (no E(B-V)):', sorted(set(n_skipped)))
    print(per_cluster.describe().loc[['min', 'max']].round(3).to_string())


if __name__ == '__main__':
    main()
