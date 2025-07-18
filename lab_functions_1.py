import numpy as np
from scipy.constants import G, proton_mass
from scipy.constants import k as k_B  # Boltzmann constant in J/K
from scipy.interpolate import interp1d
from scipy.special import erf
import WiersmaCopy as Cool
from astropy.cosmology import FlatLambdaCDM
from numba import njit
import Lambda_Tables as LT

cosmo = FlatLambdaCDM(H0=70, Om0=0.315, Ob0=0.0457)

solarm = 1.988 * 10 ** 33  # solar mass (g)
meandens = 9.21 * 10 ** (-30)  # 'mean universal density'
kpc_to_cm = 3.0857e21
gamma = 5 / 3
mu = 0.593
mp_g = proton_mass * 1e3
k_Bcgs = k_B * 1e7
ktc = kpc_to_cm
G = 6.6743*10**-8
m_p = proton_mass*1e3

@njit
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

a = 17 * ktc / (1.8153)
vcgrab_cache = {}


@njit
def vc1vc2compute(r, rs, R, stelmass, totmass):
    vc1 = G * stelmass * solarm * r / (r + a)**2

    c = R / rs
    x = r / rs
    vc2 = (G * (totmass * solarm) * c * Ax(x)) / (R * x * Ax(c))
    return np.sqrt(vc1 + vc2)

def vcgrab(r, z, stelmass, totmass):
    key = (r, z, stelmass, totmass)
    if key in vcgrab_cache:
        return vcgrab_cache[key]
    
    R = findR(z, totmass)
    rs = R / 4.3

    result = vc1vc2compute(r, rs, R, stelmass, totmass)
    vcgrab_cache[key] = result
    return result


def Tc(r, z, stelmass, totmass):
    const = (0.6 * m_p / (gamma * k_Bcgs))
    vc2 = vcgrab(r, z, stelmass, totmass)**2
    return const * vc2

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

@njit
def expcor(r):
    clogr = np.exp(-0.007 * r)
    return clogr

def Lambdacalc(logT, r, assumedZ, n):
    adjLamb = np.interp(logT, x, y)
    Lambda = 10 ** adjLamb

    if logT > 8.16:
        Lambda = ((10**logT) ** 0.5) * 2.982e-27

    # Radius-based metallicity correction (if needed)
    r = r / ktc
    clogr = expcor(r)

    if assumedZ == -1:
        return Lambda * clogr

    elif assumedZ == -2:
        T = 10**logT
        Z2Zsun = 0.3
        z = 0.597
        key = (Z2Zsun, z)
        if key not in _Wiersma_cache:
            _Wiersma_cache[key] = Cool.Wiersma_Cooling(Z2Zsun, z)
        cooling = _Wiersma_cache[key]
        return cooling.fast_LAMBDA(T, n)

    else:
        return Lambda * (assumedZ/.3)


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
    if assumedZ == -2:
        tcoolset = (Pset/(5/2) / (nset**2 * Lambdacalc(np.log10(T), r, 0.25, 1)))
    else:
        tcoolset = (Pset/(5/2) / (nset**2 * Lambdacalc(np.log10(T), r, assumedZ, 1)))
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

def Kcalc(r, z, totmass):
    R = findR(z, totmass)
    Ob0 = cosmo.Ob0  # baryon density today
    E_z = cosmo.efunc(z)  # E(z) = H(z)/H0

    Ob_z = Ob0 * (1 + z)**3 / E_z**2
    fb = Ob_z / cosmo.Om(z)
    K200 = (G * totmass * solarm / 2 * R) * (200 * cosmo.critical_density(z) * fb / (1.14 * mp_g))**(-2/3)
    K = 1.47 * (r/R)**1.22 * K200
    return K.value

def dlnTdlnrcalc(R_sonic, x, z, T_sonic_point, Lambdatype): 
    if Lambdatype == -2:
        dlnL1 = np.log(Lambdacalc(np.log10(T_sonic_point), R_sonic, 0.25, 1))
        dlnL2 = np.log(Lambdacalc(np.log10(T_sonic_point*1.01), R_sonic, 0.25, 1))
    else:
        dlnL1 = np.log(Lambdacalc(np.log10(T_sonic_point), R_sonic, Lambdatype, 1))
        dlnL2 = np.log(Lambdacalc(np.log10(T_sonic_point*1.01), R_sonic, Lambdatype, 1))
    
    dlnT1 = np.log(T_sonic_point)
    dlnT2 = np.log(T_sonic_point*1.01)
    dlnLambda_dlnT = (dlnL2 - dlnL1)/(dlnT2 - dlnT1)


    dlnvc_dlnR = dVcdrfunc(R_sonic, z, 3e12, 2e15)
    
    #solve quadratic equation    
    b = 29/6.*x - 17/6. + 1/3.*(1.-x)*(dlnLambda_dlnT)
    c = 2/3.*x*dlnvc_dlnR + 5*x**2 - 13/3.*x + 2/3.
    if b**2-4*c >= 0:
        return [(-b +j * (b**2-4*c)**0.5)/2. for j in (-1,1)]
    else:
        return None, None

def Hernqdens(r, z, stelmass):
    a = 9.3648*ktc
    rho = (stelmass * solarm / 2 * np.pi)*(a / (r * (r + a)**3))
    return rho