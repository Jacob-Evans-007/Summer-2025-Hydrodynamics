import numpy as np
import glob, h5py
from scipy import interpolate

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
        
        f = h5py.File(fn, 'r')
        
        He2Habundance = 10**-1.07 * (0.71553 + 0.28447 * Z2Zsun)
        X = (1 - 0.014 * Z2Zsun) / (1. + 4. * He2Habundance)
        Y = 4. * He2Habundance * X
        iHe = searchsortedclosest(f['Metal_free']['Helium_mass_fraction_bins'][...], Y)
        
        H_He_Cooling  = f['Metal_free']['Net_Cooling'][iHe, ...]
        Tbins         = f['Metal_free']['Temperature_bins'][...]
        nHbins        = f['Metal_free']['Hydrogen_density_bins'][...]
        Metal_Cooling = f['Total_Metals']['Net_cooling'][...] * Z2Zsun
        
        # Final cooling table
        Cooling_Table = Metal_Cooling + H_He_Cooling

        # Store interpolator
        self.f_Cooling = interpolate.RegularGridInterpolator(
            (np.log10(Tbins), np.log10(nHbins)),
            Cooling_Table,
            bounds_error=False, fill_value=None
        )

        # Store derivatives
        Xg, Yg = np.meshgrid(Tbins, nHbins, indexing='ij')
        log_vals = np.log(self.f_Cooling((np.log10(Xg), np.log10(Yg))))
        dlogT = np.diff(np.log10(Tbins))[0]
        dlogn = np.diff(np.log10(nHbins))[0]
        dlnLambda_dlnrhoArr, dlnLambda_dlnTArr = np.gradient(log_vals, dlogn, dlogT)

        self.dlnLambda_dlnT_interp = interpolate.RegularGridInterpolator(
            (np.log10(Tbins), np.log10(nHbins)),
            dlnLambda_dlnTArr,
            bounds_error=False, fill_value=None
        )
        self.dlnLambda_dlnrho_interp = interpolate.RegularGridInterpolator(
            (np.log10(Tbins), np.log10(nHbins)),
            dlnLambda_dlnrhoArr,
            bounds_error=False, fill_value=None
        )

    def LAMBDA(self, T, nH):
        """Cooling function in erg cm^3 / s"""
        return self.f_Cooling((np.log10(T), np.log10(nH)))

    def tcool(self, T, nH):
        """Cooling time in seconds (uses 3.5 k_B T / (nH * Lambda))"""
        k_B = 1.380649e-16  # erg/K
        Lambda_val = self.LAMBDA(T, nH)
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
    



def searchsortedclosest(arr, val):
    if arr[0]<arr[1]:
        ind = np.searchsorted(arr,val)
        ind = minarray(ind, len(arr)-1)
        return maxarray(ind - (val - arr[maxarray(ind-1,0)] < arr[ind] - val),0)        
    else:
        ind = np.searchsorted(-arr,-val)
        ind = minarray(ind, len(arr)-1)
        return maxarray(ind - (-val + arr[maxarray(ind-1,0)] < -arr[ind] + val),0)        
def maxarray(arr, v):
    return arr + (arr<v)*(v-arr)
def minarray(arr, v):
    return arr + (arr>v)*(v-arr)