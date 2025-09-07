import numpy as np
from scipy.constants import G, proton_mass
from scipy.constants import k as k_B  # Boltzmann constant in J/K
from scipy.interpolate import interp1d
from scipy.special import erf
import WiersmaCopy as Cool
from astropy.cosmology import FlatLambdaCDM
from numba import njit
import Lambda_Tables as LT
from scipy.interpolate import RegularGridInterpolator

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
    vc3 = (G * 2e10 * solarm)/r
    return np.sqrt(vc1 + vc2 + vc3)

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

y2 = np.array([-21.6106,-21.4817,-21.5094,-21.5894,-21.6708,-21.7330,-21.7528,-21.7361,-21.6873,-21.6136, -21.5229, 
    -21.4248,-21.3330,-21.2392,-21.1599,-21.0871,-21.0286,-20.9858,-20.9582,-20.9451, -20.9602,-20.9861,-21.0079,-21.0052,
    -20.9968,-20.9834,-20.9707,-20.9670,-20.9647,-20.9722,-21.0099,-21.0958,-21.2556,-21.4404,-21.5834,-21.6649,-21.7010,
    -21.7207,-21.7454,-21.7931,-21.8615,-21.9160,-21.9476,-21.9656,-21.9760,-21.9938,-22.0144,-22.0355,-22.0433,-22.0536,
    -22.0725,-22.1082,-22.1698,-22.2486,-22.3375,-22.4159,-22.4820,-22.5278,-22.5612,-22.5811,-22.5905,-22.5940,-22.5857,
    -22.5797,-22.5694,-22.5612,-22.5573,-22.5566,-22.5635,-22.5718,-22.5831,-22.6010,-22.6169,-22.6391,-22.6596,-22.6840,
    -22.7021,-22.7090,-22.7121,-22.7129,-22.7135,-22.7065,-22.6959,-22.6840,-22.6722,-22.6574,-22.6418,-22.6337,-22.6173,
    -22.5998,-22.5828,-22.5679,-22.5500,-22.5324,-22.5152,-22.4972,-22.4822,-22.4644,-22.4467,-22.4287])

y3 = np.array([-21.6087,-21.4779,-21.5009,-21.5702,-21.6311,-21.6603,-21.6373,-21.5738,-21.4838,-21.3771, -21.2642,-21.1525,
    -21.0529,-20.9557,-20.8769,-20.8078,-20.7546,-20.7154,-20.6877,-20.6713,-20.6828,-20.7056,-20.7242,-20.7180,-20.7069,-20.6912,
    -20.6768,-20.6720,-20.6687,-20.6755,-20.7130,-20.7993,-20.9603,-21.1472,-21.2921,-21.3746,-21.4107,-21.4301,-21.4547,-21.5029,
    -21.5726,-21.6281,-21.6600,-21.6782,-21.6885,-21.7067,-21.7277,-21.7494,-21.7574,-21.7680,-21.7877,-21.8248,-21.8893,-21.9727,
    -22.0680,-22.1536,-22.2272,-22.2797,-22.3195,-22.3434,-22.3570,-22.3622,-22.3553,-22.3501,-22.3409,-22.3324,-22.3309,-22.3340,
    -22.3443,-22.3580,-22.3758,-22.4007,-22.4264,-22.4606,-22.4953,-22.5331,-22.5666,-22.5865,-22.6018,-22.6128,-22.6215,-22.6209,
    -22.6164,-22.6101,-22.6019,-22.5907,-22.5782,-22.5715,-22.5577,-22.5421,-22.5275,-22.5143,-22.4980,-22.4822,-22.4671,-22.4508,
    -22.4375,-22.4215,-22.4056,-22.3893])

logLambda_grid = np.vstack([y, y2, y3])
Z_values = np.array([0.3, 0.5, 1.0])

interpolator = RegularGridInterpolator((Z_values, x), logLambda_grid, bounds_error=False, fill_value=None)


@njit
def expcor(a):
    return (1.31 * np.exp(a * -.63))


def Lambdacalc(logT, r, assumedZ, n):
    if assumedZ == -2:
        T = 10**logT
        Z2Zsun = 0.3
        z = 0.597
        key = (Z2Zsun, z)
        if key not in _Wiersma_cache:
            _Wiersma_cache[key] = Cool.Wiersma_Cooling(Z2Zsun, z)
        cooling = _Wiersma_cache[key]
        return cooling.fast_LAMBDA(T, n)
    
    elif assumedZ == -1:
        r = r / ktc
        clogr = expcor(np.log10(r))
        return Lambdacalc(logT, r, clogr, 1)
    
    else:
        # Interpolate log Lambda and return Lambda
        logL = interpolator([(assumedZ, logT)])[0]
        return 10**logL



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