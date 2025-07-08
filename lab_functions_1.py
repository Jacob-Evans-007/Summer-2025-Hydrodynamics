import numpy as np
from scipy.constants import G, proton_mass
from scipy.constants import k as k_B  # Boltzmann constant in J/K
from scipy.interpolate import interp1d
from scipy.special import erf
import WiersmaCopy as Cool
from astropy.cosmology import FlatLambdaCDM
from numba import njit

cosmo = FlatLambdaCDM(H0=70, Om0=0.30)

solarm = 1.988 * 10 ** 33  # solar mass (g)
meandens = 9.21 * 10 ** (-30)  # 'mean universal density'
kpc_to_cm = 3.0857e21
gamma = 5 / 3
mu = 0.6
mp_g = proton_mass * 1e3
k_Bcgs = k_B * 1e7
ktc = kpc_to_cm
G = 6.6743*10**-8
m_p = proton_mass*1e3


def Ax(n):
    return np.log(1 + n) - (n / (1 + n))


def findR(z, totmass):
    rho_m = cosmo.Om(z) * cosmo.critical_density(z)
    rho_m = rho_m.value
    x = cosmo.Om(z) - 1
    delc = 18 * np.pi**2 + 82 * x - 39 * x**2
    delm = delc / cosmo.Om(z)
    Rvir = (3 * totmass * solarm / (4 * np.pi * delm * rho_m))**(1/3)
    return Rvir



vcgrab_cache = {}

def vcgrab(r, z, stelmass, totmass):
    key = (r, z, stelmass, totmass)

    if key in vcgrab_cache:
        return vcgrab_cache[key]
    
    R = findR(z, totmass)
    rs = R / 5

    Rhalf = 3 * ktc * (totmass / 1e12)**(1/3)
    vc1 = (G * stelmass * solarm) / (r + Rhalf)

    a = 17 * ktc / (1 + np.sqrt(2))
    vc12 = G * stelmass * solarm * r / (r + a)**2

    c = R / rs
    x = r / rs
    vc2 = (G * (totmass * solarm) * c * Ax(x)) / (R * x * Ax(c))

    result = np.sqrt(vc12 + vc2)
    vcgrab_cache[key] = result
    return result


def Tc(r, z, stelmass, totmass):
    const = (0.6 * m_p / (gamma * k_Bcgs))
    vc2 = vcgrab(r, z, stelmass, totmass)**2
    return const * vc2

@njit
def expcor(a):
    return (1.31 * np.exp(a * -.63))/(0.3)

# Global cache for Wiersma cooling object
_Wiersma_cache = {}

x = np.arange(4.20, 8.20, 0.04)
y = np.array([-21.6114, -21.4833, -21.5129, -21.5974, -21.6878, -21.7659, -21.8092, -21.8230, -21.8059, -21.7621, -21.6941, -21.6111, 
    -21.5286, -21.4387, -21.3589, -21.2816, -21.2168, -21.1700, -21.1423, -21.1331, -21.1525, -21.1820, -21.2077, -21.2093, -21.2043, 
    -21.1937, -21.1832, -21.1811, -21.1799, -21.1883, -21.2263, -21.3118, -21.4700, -21.6521, -21.7926, -21.8728, -21.9090, -21.9290, 
    -21.9539, -22.0008, -22.0678, -22.1209, -22.1521, -22.1698, -22.1804, -22.1977, -22.2178, -22.2383, -22.2459, -22.2557, -22.2736, 
    -22.3075, -22.3657, -22.4391, -22.5207, -22.5909, -22.6490, -22.6878, -22.7148, -22.7308, -22.7361, -22.7379, -22.7283, -22.7216, 
    -22.7102, -22.7023, -22.6962, -22.6921, -22.6959, -22.6994, -22.7050, -22.7170, -22.7249, -22.7378, -22.7480, -22.7629, -22.7710, 
    -22.7697, -22.7655, -22.7605, -22.7565, -22.7461, -22.7323, -22.7176, -22.7039, -22.6873, -22.6700, -22.6613, -22.6436, -22.6251,
    -22.6071, -22.5914, -22.5727, -22.5542, -22.5360, -22.5172, -22.5014, -22.4828, -22.4642, -22.4455])
f = interp1d(x, y, kind='cubic', fill_value='extrapolate')

def Lambdacalc(T, r, assumedZ, n):
    adjT = np.log10(T)
    adjLamb = f(adjT)
    Lambda = 10 ** adjLamb

    # Radius-based metallicity correction (if needed)
    logr = np.log10(r / ktc)
    clogr = expcor(logr)

    if assumedZ == "a":
        return Lambda * clogr

    elif assumedZ == "W":
        Z2Zsun = 0.3
        z = 0.6
        key = (Z2Zsun, z)
        if key not in _Wiersma_cache:
            _Wiersma_cache[key] = Cool.Wiersma_Cooling(Z2Zsun, z)
        cooling = _Wiersma_cache[key]
        return cooling.fast_LAMBDA(T, n)

    else:
        return Lambda * assumedZ

# def dLdTfunc(T):
#     Lambda = Lambdacalc(T)
#     logT = np.log(T)
#     logL = np.log(Lambda)

#     Lambda2 = Lambdacalc(T * 1.01)
#     logLadj = np.log(Lambda2)
#     logTadj = np.log(T * 1.01)
#     return (logLadj - logL) / (logTadj - logT)


def dVcdrfunc(r, z, stelmass, totmass):
    Vc = vcgrab(r, z, stelmass, totmass)
    logvc = np.log(Vc)
    logr = np.log(r)

    logvc2 = np.log(vcgrab(r + 3.0857e15, z, stelmass, totmass))
    logr2 = np.log(r + 3.0857e15)
    return (logvc2 - logvc) / (logr2 - logr)


def rhocalc(v, tftc, T, r, assumedZ):
    gamma = 5/3
    nset = 10**np.arange(-7, 10, 0.01)
    rhoset = nset*mp_g
    Pset = nset * k_Bcgs * T
    if assumedZ == "W":
        tcoolset = (Pset/(5/2) / (nset**2 * Lambdacalc(T, r, "a", 1)))
    else:
        tcoolset = (Pset/(5/2) / (nset**2 * Lambdacalc(T, r, assumedZ, 1)))
    vset = (r / (tcoolset * tftc))

    lv = np.log10(v)
    lvs = np.log10(vset)
    arr = lvs - lv
    inds = ((np.sign(arr[1:]) * np.sign(arr[:-1])) < 0).nonzero()[0]
    if len(inds)!=1: return False
    ind = inds[0]
    log_rhoset = np.log10(rhoset)
    if vset[ind]<vset[ind+1]: 
        good_lrho = np.interp(lv, lvs[ind:ind+2], log_rhoset[ind:ind+2]) 
    else:
        good_lrho = np.interp(lv, lvs[ind+1:ind-1:-1], log_rhoset[ind+1:ind-1:-1]) 
    return 10.**good_lrho

#Potentially put on HPCC when speed dies.