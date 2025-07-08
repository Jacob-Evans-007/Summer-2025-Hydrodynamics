import numpy as np
import glob, h5py
from scipy import interpolate
from numba import njit

class Wiersma_Cooling:
    """
    Unitless Wiersma+09 cooling function.
    Inputs:
      - T: temperature in K
      - nH: hydrogen number density in cm^-3
    Output:
      - Lambda(T, nH): erg cm^3 / s
    """
    def __init__(self, Z2Zsun, z, dataDir='cooling/'):
        fns = np.array(glob.glob(dataDir + 'CoolingTables/z_?.???.hdf5'))
        zs = np.array([float(fn[-10:-5]) for fn in fns])
        fn = fns[zs.argsort()][searchsortedclosest(sorted(zs), z)]

        with h5py.File(fn, 'r') as f:
            He2Habundance = 10**-1.07 * (0.71553 + 0.28447 * Z2Zsun)
            X = (1 - 0.014 * Z2Zsun) / (1. + 4. * He2Habundance)
            Y = 4. * He2Habundance * X
            iHe = searchsortedclosest(f['Metal_free']['Helium_mass_fraction_bins'][...], Y)

            H_He_Cooling = f['Metal_free']['Net_Cooling'][iHe, ...]
            Tbins = f['Metal_free']['Temperature_bins'][...]
            nHbins = f['Metal_free']['Hydrogen_density_bins'][...]
            Metal_Cooling = f['Total_Metals']['Net_cooling'][...] * Z2Zsun

            Cooling_Table = Metal_Cooling + H_He_Cooling

        # Store grids and table in log space for interpolation
        self.logT_grid = np.log10(Tbins)
        self.logn_grid = np.log10(nHbins)
        self.logLambda_table = np.log10(Cooling_Table)

        # For fallback (non-numba) smooth interpolation if needed
        self.f_Cooling = interpolate.RegularGridInterpolator(
            (self.logT_grid, self.logn_grid),
            Cooling_Table,
            bounds_error=False, fill_value=None
        )

        # Compute derivatives for gradient interpolation (optional)
        Xg, Yg = np.meshgrid(Tbins, nHbins, indexing='ij')
        log_vals = np.log(self.f_Cooling((np.log10(Xg), np.log10(Yg))))
        dlogT = np.diff(np.log10(Tbins))[0]
        dlogn = np.diff(np.log10(nHbins))[0]
        dlnLambda_dlnrhoArr, dlnLambda_dlnTArr = np.gradient(log_vals, dlogn, dlogT)

        self.dlnLambda_dlnT_interp = interpolate.RegularGridInterpolator(
            (self.logT_grid, self.logn_grid),
            dlnLambda_dlnTArr,
            bounds_error=False, fill_value=None
        )
        self.dlnLambda_dlnrho_interp = interpolate.RegularGridInterpolator(
            (self.logT_grid, self.logn_grid),
            dlnLambda_dlnrhoArr,
            bounds_error=False, fill_value=None
        )

    @staticmethod
    @njit
    def fast_LAMBDA_numba(T, nH, logT_grid, logn_grid, logLambda_table):
        logT = np.log10(T)
        logn = np.log10(nH)

        i = np.searchsorted(logT_grid, logT) - 1
        j = np.searchsorted(logn_grid, logn) - 1

        i = max(0, min(i, len(logT_grid) - 2))
        j = max(0, min(j, len(logn_grid) - 2))

        x1 = logT_grid[i]
        x2 = logT_grid[i + 1]
        y1 = logn_grid[j]
        y2 = logn_grid[j + 1]

        Q11 = logLambda_table[i, j]
        Q12 = logLambda_table[i, j + 1]
        Q21 = logLambda_table[i + 1, j]
        Q22 = logLambda_table[i + 1, j + 1]

        denom = (x2 - x1) * (y2 - y1)
        if denom == 0:
            return 10 ** Q11

        f = (
            Q11 * (x2 - logT) * (y2 - logn) +
            Q21 * (logT - x1) * (y2 - logn) +
            Q12 * (x2 - logT) * (logn - y1) +
            Q22 * (logT - x1) * (logn - y1)
        ) / denom

        return 10 ** f

    def fast_LAMBDA(self, T, nH):
        return self.fast_LAMBDA_numba(T, nH, self.logT_grid, self.logn_grid, self.logLambda_table)

    def LAMBDA(self, T, nH):
        # fallback to scipy interpolator if you want
        return self.f_Cooling((np.log10(T), np.log10(nH)))

    def tcool(self, T, nH):
        k_B = 1.380649e-16  # erg/K
        Lambda_val = self.fast_LAMBDA(T, nH)  # use fast here for speed
        return (3.5 * k_B * T) / (nH * Lambda_val)

    def f_dlnLambda_dlnT(self, T, nH):
        return self.dlnLambda_dlnT_interp((np.log10(T), np.log10(nH)))

    def f_dlnLambda_dlnrho(self, T, nH):
        return self.dlnLambda_dlnrho_interp((np.log10(T), np.log10(nH)))


def searchsortedclosest(arr, val):
    ind = np.searchsorted(arr, val)
    ind = np.clip(ind, 1, len(arr) - 1)
    left = arr[ind - 1]
    right = arr[ind]
    return ind - 1 if abs(val - left) < abs(val - right) else ind