import random
import statistics
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import numpy as np
#from scipy.integrate import odeint
#from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.constants import k as k_B  # Boltzmann constant in J/K
from scipy.constants import G, proton_mass
from functools import partial
import lab_functions_1 as lf
from numba import njit
from scipy.integrate import cumulative_trapezoid

mp_g = proton_mass*1e3
k_Bcgs = k_B*1e7
mu = 0.6
gamma = 5/3
ktc = 3.0857e21
etkv = 6.2415*10**8
kevtk = 1.16*10**7
z = 0.597
Mdot1 = 6.30391e25

radii_grid = np.linspace(1*ktc, 20000*ktc, 75000)
vc_grid = np.array([lf.vcgrab(r, z, 3e12, 2e15) for r in radii_grid])
vc_interp = interp1d(radii_grid, vc_grid, kind='cubic', fill_value='extrapolate')


r_grid = np.geomspace(ktc*0.01, 2200000 * ktc, 100000)  # finer grid = better accuracy
vc2_over_r = np.array([(lf.vcgrab(r, z, 3e12, 2e15 )**2 / r) for r in r_grid])

phi_cumint = cumulative_trapezoid(vc2_over_r, r_grid, initial=0.0)

phi_values = -(phi_cumint[-1] - phi_cumint)

phi_interp = interp1d(r_grid, phi_values, kind='cubic', fill_value="extrapolate")

def phi(r):
    """Interpolated gravitational potential at any r."""
    return phi_interp(r)

def dlnTdlnrcalc(R_sonic, x, T_sonic_point, Lambdatype, pr=True): 
    if Lambdatype == "W":
        dlnL1 = np.log(lf.Lambdacalc(np.log10(T_sonic_point), R_sonic, "a", 1))
        dlnL2 = np.log(lf.Lambdacalc(np.log10(T_sonic_point*1.01), R_sonic, "a", 1))
    else:
        dlnL1 = np.log(lf.Lambdacalc(np.log10(T_sonic_point), R_sonic, Lambdatype, 1))
        dlnL2 = np.log(lf.Lambdacalc(np.log10(T_sonic_point*1.01), R_sonic, Lambdatype, 1))
    
    dlnT1 = np.log(T_sonic_point)
    dlnT2 = np.log(T_sonic_point*1.01)
    dlnLambda_dlnT = (dlnL2 - dlnL1)/(dlnT2 - dlnT1)


    dlnvc_dlnR = lf.dVcdrfunc(R_sonic, z, 3e12, 2e15)
    
    #solve quadratic equation    
    b = 29/6.*x - 17/6. + 1/3.*(1.-x)*(dlnLambda_dlnT)
    c = 2/3.*x*dlnvc_dlnR + 5*x**2 - 13/3.*x + 2/3.
    if b**2-4*c >= 0:
        return [(-b +j * (b**2-4*c)**0.5)/2. for j in (-1,1)]
    else:
        return None, None
    
@njit
def compute_dvdr_dTdr(v, T, r, Mdot, vc2, Lambda):
    cs2 = (gamma * k_Bcgs * T) / (mu * mp_g)
    tflow = r / abs(v)
    mach2 = v**2 / cs2
    rho = Mdot / (4 * np.pi * r**2 * v)
    n = rho / (mu * mp_g)
    tcool = (3 * k_Bcgs * T) / (2 * Lambda * n)

    dlnvdlnr = (2 - (vc2 / cs2) - (tflow / (gamma * tcool))) / (mach2 - 1.0)
    dlnTdlnr = (tflow / tcool) - (2 / 3) * (2 + dlnvdlnr)

    dvdr = (v / r) * dlnvdlnr
    dTdr = (T / r) * dlnTdlnr

    return dvdr, dTdr

def TheODE(r, C, Mdot, Lambdatype, recorder=None):
    v, T = C
    vc2 = vc_interp(r)**2

    # n is computed inside, so use dummy rho/n for Lambda
    rho = Mdot / (4 * np.pi * r**2 * v)
    n = rho / (mu * mp_g)
    Lambda = lf.Lambdacalc(np.log10(T), r, Lambdatype, n)

    dvdr, dTdr = compute_dvdr_dTdr(v, T, r, Mdot, vc2, Lambda)

    if recorder is not None:
        recorder["ra2"].append(r)
        recorder["varray"].append(v)
        recorder["Tarray"].append(T)
        recorder["rhoarray"].append(rho)
        cs2 = (gamma * k_Bcgs * T) / (mu * mp_g)  
        Bern = (v**2 / 2) + (cs2 * 3 / 2) + phi(r)
        recorder["Bern"].append(Bern)
        mach = np.sqrt(v**2 / cs2)
        recorder["Mach"].append(mach)
    return [dvdr, dTdr]

    # CLASSES (THE BANE OF MY EXISTENCE)
class IntegrationResult:
    def __init__(self, res, stop_reason, xval=None, R0=None, v0=None, T0=None, Mdot=None):
        self.res = res
        self._stop_reason = stop_reason
        self.xval = xval
        self.R0 = R0
        self.v0 = v0
        self.T0 = T0
    
    def stopReason(self):
        return self._stop_reason
    
    def Rs(self):
        return self.res.t
    
    def __getitem__(self, key):
        return self.res[key]


# EVENTS LIST
def event_unbound(r, C, Mdot, Lambdatype, _):
    v, T = C
    cs2 = (gamma * k_Bcgs * T) / (mu * mp_g)
    phi_r = phi(r)
    bern = 0.5 * v**2 + 1.5 * cs2 + phi_r
    return bern
event_unbound.terminal = True
event_unbound.direction = 1

def event_lowT(r, C, Mdot, Lambdatype, _):
    T = C[1]
    return T - (10**4.2)
event_lowT.terminal = True
event_lowT.direction = -1 

def event_sonic_point(r, C, Mdot, Lambdatype, _):
    v, T = C
    cs2 = (gamma * k_Bcgs * T) / (mu * mp_g)
    mach = v / np.sqrt(cs2)
    return mach - 1.0
event_sonic_point.terminal = True
event_sonic_point.direction = -1

def event_max_R(r, C, Mdot, Lambdatype, _):
    v, T = C
    return r - (20000*ktc)
event_max_R.terminal = True
event_max_R.direction = 1

def event_overstepdlnv(r, C, Mdot, Lambdatype, _):
    v, T = C

    vc2 = vc_interp(r)**2
    cs2 = (gamma * k_Bcgs * T) / (mu * mp_g)
    
    tflow = r / np.abs(v)
    mach = v / np.sqrt(cs2)

    rho = Mdot / (4 * np.pi * r**2 * v)
    n = rho / (mu * mp_g)
    Lambda = lf.Lambdacalc(np.log10(T), r, Lambdatype, n)
    
    tcool = (3 * k_Bcgs * T) / (2 * Lambda * n)

    dlnvdlnr = (2 - (vc2 / cs2) - (tflow/ (gamma*tcool))) / (mach**2 - 1.0)
    return np.abs(dlnvdlnr) - 50
event_overstepdlnv.terminal = True
event_overstepdlnv.direction = 1 

my_event_list = [
    event_sonic_point,
    event_unbound,
    event_lowT,
    event_max_R,
    event_overstepdlnv
]


event_names = ['sonic point', 'unbound', 'lowT', 'max R reached', 'overstepdlnv' ]

# SHOOTING METHOD
def sonic_point_shooting(Rsonic, Lambdatype, Rmax=20000*ktc, tol=1e-8, epsilon=1e-5, dlnMdlnRInit=-1, x_high=0.99, x_low=0.01, return_all_results=False):
    results = {}
    dlnMdlnRold = dlnMdlnRInit
    
    # x = v_c / 2*c_s is the iterative variable
    while x_high - x_low > tol:
        #INITIAL GUESSES
        x = 0.5 * (x_high + x_low)
        cs2_sonic = vc_interp(Rsonic)**2 / (2 * x)
        v_sonic = cs2_sonic**0.5
        T_sonic = mu * mp_g * cs2_sonic / (gamma * k_Bcgs)
        tflow_to_tcool = (10/3) * (1 - x)
        
        rho_sonic = lf.rhocalc(v_sonic, tflow_to_tcool, T_sonic, Rsonic, Lambdatype)
        if rho_sonic == False:
            x_high = x
            continue
        Mdot = 4 * np.pi * Rsonic**2 * rho_sonic * v_sonic
        
        dlnTdlnR1, dlnTdlnR2 = dlnTdlnrcalc(Rsonic, x, T_sonic, Lambdatype, pr=True)
        if dlnTdlnR1 is None:
            x_high = x
            continue
        
        dlnMdlnR1, dlnMdlnR2 = [3 - 5*x - 2*dlnTdlnR for dlnTdlnR in (dlnTdlnR1, dlnTdlnR2)]
        if abs(dlnMdlnR1 - dlnMdlnRold) < abs(dlnMdlnR2 - dlnMdlnRold):
            dlnTdlnR = dlnTdlnR1
        else:
            dlnTdlnR = dlnTdlnR2
        
        dlnMdlnR = 3 - 5*x - 2*dlnTdlnR
        
        dlnvdlnR = -1.5 * dlnTdlnR + 3 - 5 * x
        
        T0 = T_sonic * (1 + epsilon * dlnTdlnR)
        v0 = v_sonic * (1 + epsilon * dlnvdlnR)
        R0 = Rsonic * (1 + epsilon)

        # Early checks
        cs2_0 = (gamma * k_Bcgs * T0) / (mu * mp_g)
        mach0 = v0 / np.sqrt(cs2_0)
        if mach0 > 1.0:
            x_high = x
            continue

        phi0 = phi(R0)
        bern = 0.5 * v0**2 + 1.5 * cs2_0 + phi0
        
        if bern > 0:
            x_low = x
            continue
        res_raw = solve_ivp(TheODE, [R0, Rmax], [v0, T0], args=(Mdot, Lambdatype, None), method='LSODA', 
            atol=1e-5, rtol=1e-5, events=my_event_list, dense_output=True)
        
        if res_raw.status < 0:
            stop_reason = 'integration failure'
        elif any(len(evt) > 0 for evt in res_raw.t_events):
            for idx, t_evt in enumerate(res_raw.t_events):
                if len(t_evt) > 0:
                    stop_reason = event_names[idx]
                    break
        else:
            if res_raw.t[-1] >= Rmax:
                stop_reason = 'max R reached'
            else:
                stop_reason = 'unknown'
        
        res = IntegrationResult(res_raw, stop_reason, xval=x, R0=R0, v0=v0, T0=T0)
        
        if res.stopReason() in ('sonic point', 'lowT', 'overstepdlnv'):
            x_high = x
            continue
        elif res.stopReason() == 'unbound':
            x_low = x
            continue
        elif res.stopReason() == 'max R reached':
            dlnMdlnRold = dlnMdlnR
            results[x] = res
            break
        else:
            break
    
    if return_all_results:
        return results
    if len(results) == 0:
        return None
    else:
        return results[x]

def find_converged_x(Rsonic, Lambdatype):
    result = sonic_point_shooting(Rsonic, Lambdatype)
    if result is None:
        return None
    return result 


def find_mdot(Rsonic, Lambdatype, result=None):
    if result is None:
        result = sonic_point_shooting(Rsonic, Lambdatype)
        if result is None:
            return np.nan
    x = result.xval
    cs2_sonic = vc_interp(Rsonic)**2 / (2 * x)
    v_sonic = cs2_sonic**0.5
    T_sonic = mu * mp_g * cs2_sonic / (gamma * k_Bcgs)
    tflow_to_tcool = (10 / 3) * (1 - x)
    rho_sonic = lf.rhocalc(v_sonic, tflow_to_tcool, T_sonic, Rsonic, Lambdatype)
    if rho_sonic is False:
        return np.nan
    Mdot = 4 * np.pi * Rsonic**2 * rho_sonic * v_sonic
    return Mdot

def postprocess(b, Mdot, Lambdatype):
    b_result = sonic_point_shooting(b, Lambdatype)
    x = b_result.xval
    R0 = b_result.R0
    v0 = b_result.v0
    T0 = b_result.T0
    Rmax = 20000 * ktc

    recorder = {"ra2": [], "varray": [], "Tarray": [], "rhoarray": [], "Bern": [], "Mach": []}

    res = solve_ivp(TheODE, [R0, Rmax], [v0, T0], args=(Mdot, Lambdatype, recorder), method='LSODA', max_step=Rmax / 100,
        atol=1e-5, rtol=1e-5, dense_output=True)

    return x, R0, v0, T0, recorder

def BrentLooper(Mdot, Rsonlow, Rsonhigh, Lambdatype, tol=(2e-6 * Mdot1)):
    target = Mdot
    a, b = Rsonlow, Rsonhigh
    fa = find_mdot(a, Lambdatype) - target
    fb = find_mdot(b, Lambdatype) - target

    if fa * fb >= 0:
        raise ValueError("Not bounded correctly!")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = e = b - a
    mflag = True

    while abs(b - a) > tol:
        if abs(fb) < tol:
            return postprocess(b, Mdot, Lambdatype)
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        if a < b:
            cond1 = not ((3 * a + b) / 4 < s < b)
        else:
            cond1 = not (b < s < (3 * a + b) / 4)

        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = find_mdot(s, Lambdatype) - target
        d, c = c, b
        fd, fc = fc, fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    return postprocess(b, Mdot, Lambdatype)

x_values = np.array([1069.01920, 586.29466, 355.40590, 230.31321, 159.55064, 126.31336, 108.70059, 88.97651, 76.56988, 62.67603, 47.19685, 40.61586, 
    21.90684, 18.23348, 14.67799, 11.62030, 8.18547])
y_values = np.array([2413.28546, 1405.48459, 656.73673, 360.18178, 201.53377, 146.29093, 119.74596, 90.47357, 72.58879, 54.84417, 38.24800, 31.30777, 
    15.84893, 13.23546, 8.86801, 3.60182, 2.27259])

y = np.log10(np.array([3324.598, 1522.680, 1553.475, 1297.309, 725.888, 594.173, 374.898, 346.043, 214.010, 193.623, 155.348, 137.762, 127.159, 112.765,
    96.075, 85.199, 78.642, 69.739, 57.085, 51.647, 39.811, 35.304, 33.246, 30.079, 16.830, 14.925, 14.055, 12.716, 9.607, 8.185,
    4.228, 3.008, 2.563, 1.975]))

y_pairs = y.reshape(-1, 2)

# Compute symmetric error for each pair
errors = np.abs(y_pairs[:, 0] - y_pairs[:, 1]) / 2
stdvi = errors**2

y_kvalues = np.log10(np.array([2413.28546, 1405.48459, 656.73673, 360.18178, 201.53377, 146.29093, 119.74596, 90.47357, 72.58879, 54.84417, 38.24800, 31.30777, 
    15.84893, 13.23546, 8.86801, 3.60182, 2.27259]))

def ChiSquaredCalc(Mdot, Lambdatype):
    x, R0, v0, T0, recorder = BrentLooper(Mdot*Mdot1, 0.5*ktc, 20*ktc, Lambdatype)
    rA = np.array(recorder["ra2"])
    TA = np.array(recorder["Tarray"])
    rhoA = np.array(recorder["rhoarray"])
    KA = (rhoA / (mu*mp_g))**(-2/3) * k_Bcgs * TA * etkv
    Kvals = []
    for j in range(len(x_values)):
        idx = np.argmin(np.abs(rA - x_values[j]*ktc))
        Kvals.append(np.log10(KA[idx]))
    sqdiff = (Kvals - y_kvalues)**2
    return sqdiff / stdvi

from joblib import Parallel, delayed

import os


def HeatMapCreator(Mdotlow, Mdothigh, zlow, zhigh, gridsize, outpath="heatmap_output.png"):
    M_vals = np.linspace(Mdotlow, Mdothigh, gridsize)
    Z_vals = np.linspace(zlow, zhigh, gridsize)

    chi2_grid = np.full((gridsize, gridsize), np.nan)

    for i, n in enumerate(M_vals):
        for j, Z in enumerate(Z_vals):
            try:
                chi2 = ChiSquaredCalc(n, Z).sum() / 16
            except Exception as e:
                print(f"Failure at i={i}, j={j}, Mdot={n}, Z={Z}: {e}")
                chi2 = np.nan
            chi2_grid[i, j] = chi2

    # Basic plot (non-interactive safe)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(chi2_grid, origin='lower', cmap='plasma',
                   extent=[zlow, zhigh, Mdotlow, Mdothigh], aspect='auto')
    fig.colorbar(im, ax=ax)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    mi, mj = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    best_n, best_Z = M_vals[mi], Z_vals[mj]
    best_chi2 = chi2_grid[mi, mj]

    print(f"Best fit → Mdot: {best_n:.2f}, Z: {best_Z:.3f}, Reduced chi2: {best_chi2:.4f}")
    print(f"Figure saved to: {os.path.abspath(outpath)}")

    return best_n, best_Z, best_chi2, outpath