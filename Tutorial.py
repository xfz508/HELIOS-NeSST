import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.formatter.limits'] = (-2,2)
import NeSST as nst

# N.B. functions that take D/T fractions have default values for these arguments of 0.5
frac_D = 0.5
frac_T = 0.5

Tion = 5.0e3 # eV

DTmean,DTstddev,DTvar = nst.DTprimspecmoments(Tion)
DDmean,DDstddev,DDvar = nst.DDprimspecmoments(Tion)

# Sum of primary yields is unity
Y_DT,Y_DD,Y_TT = nst.yields_normalised(Tion,frac_D=frac_D,frac_T=frac_T)
# Alternatively, scale from DT yield
# Y_DT = 1.0   # DT neutron yield, all reactions scaled by this value
# Y_DD = nst.yield_from_dt_yield_ratio('dd',Y_DT,Tion,frac_D=frac_D,frac_T=frac_T)
# Y_TT = nst.yield_from_dt_yield_ratio('tt',Y_DT,Tion,frac_D=frac_D,frac_T=frac_T)

print(Y_DT,Y_DD,Y_TT)

# DD and TT grid
E_pspec = np.linspace(1.0e6,10.0e6,300)
# DT grid
E_DTspec = np.linspace(12.5e6,15.5e6,200)
E_pspec = np.append(E_pspec,E_DTspec)

dNdE_DT = Y_DT*nst.QBrysk(E_pspec,DTmean,DTvar) # Brysk shape i.e. Gaussian
dNdE_DD = Y_DD*nst.QBrysk(E_pspec,DDmean,DDvar) # Brysk shape i.e. Gaussian
dNdE_TT = Y_TT*nst.dNdE_TT(E_pspec,Tion)

I_prim = dNdE_DT+dNdE_DD+dNdE_TT

# Scattered spectra and primary spectra can be on separate grids
E_sspec = np.linspace(1.0e6,16.0e6,400)

# Neutron-averaged areal density in kg/m2
rhoL = 1.50
# Need convert this into single scattering amplitude A_1S
A_1s = nst.rhoR_2_A1s(rhoL)

# Swap back to separate primary and scattered spectra grids
# Calculate d2sig/dEdmu on the given energy grids
nst.init_DT_scatter(E_sspec,E_pspec)

# Calculate the various scattering components for uniform areal density
dNdE_scatter,components = nst.DT_sym_scatter_spec(I_prim,frac_D=frac_D,frac_T=frac_T)

def rhoL_func(cos,p):
    return np.polynomial.legendre.legval(cos,p)

coeff = 0.2*(np.random.rand(12)-0.5)
coeff[0] = 1.0

#fig = plt.figure(dpi=200)

cos = np.linspace(-1,1,200)

#plt.plot(cos,rhoL_func(cos,coeff))
#plt.xlabel("Scattering Cosine")
#plt.ylabel("Normalised Areal Density")



# Use the defined areal density asymmetry function to find scattered spectrum
asym_dNdE_scatter, components = nst.DT_asym_scatter_spec(I_prim,lambda x: rhoL_func(x,coeff),frac_D=frac_D,frac_T=frac_T)

# Appropriate scaling by areal density
asym_dNdE_scatter *= A_1s

# Swap back to separate primary and scattered spectra grids
# Calculate d2sig/dEdmu on the given energy grids
nst.init_DT_scatter(E_sspec,E_pspec)

trans = nst.DT_transmission(rhoL,E_pspec,lambda x: rhoL_func(x,coeff))

print(nst.available_materials)

mat_C12 = nst.init_mat_scatter(E_sspec,E_pspec,'C12')
scatter_C12_total = nst.mat_scatter_spec(mat_C12,I_prim,lambda x : np.ones_like(x))

rhoR = 1.0 # kg/m^2
A_1S = mat_C12.rhoR_2_A1s(rhoR)

dNdE_scatter = A_1S*scatter_C12_total

# Energy grid for nT edge, zoom in on region of interest
E_nT = np.linspace(3.0e6,4.5e6,200)

# Ion velocity grid
varr = np.linspace(-800e3,800e3,100)
nst.init_DT_scatter(E_nT,E_DTspec)
nst.init_DT_ionkin_scatter(varr,nT=True,nD=False)

# DT spectrum on DT spec grid
I_DT = nst.QBrysk(E_DTspec,DTmean,DTvar)
nst.calc_DT_ionkin_primspec_rhoL_integral(I_DT,nT=True,nD=False)

# Mean and std. dev. of ion velocity distribution
vbar = 80.0e3 # m/s
dv   = 250.0e3 # m/s

dNdE,components                   = nst.DT_scatter_spec_w_ionkin(I_DT,vbar,dv,rhoL_func= lambda x : np.ones_like(x),frac_D=frac_D,frac_T=frac_T)
dNdE_noionvel,components_noionvel = nst.DT_sym_scatter_spec(I_DT,frac_D=frac_D,frac_T=frac_T)

# Neutron-averaged areal density in kg/m2
rhoL = 1.50
# Need convert this into single scattering amplitude A_1S
A_1s = nst.rhoR_2_A1s(rhoL)

# Consider only scattering of DT fusion neutrons
E_DTspec = np.linspace(13.0e6,15.0e6,100)
# Scattered spectra and primary spectra can be on separate grids
E_sspec = np.linspace(1.0e6,2.75e6,50,endpoint=False)
E_sspec = np.append(E_sspec,np.linspace(2.75e6,8.0e6,450,endpoint=False))
E_sspec = np.append(E_sspec,np.linspace(8.0e6,15.0e6,50))
# Ion velocity grid
varr = np.linspace(-1200e3,1200e3,150)

DT_spec_fit = nst.fitting.DT_fit_function(E_DTspec,E_sspec,varr)

Tion = 5.0e3 # eV

DT_spec_fit.set_primary_Tion(Tion)

P1_arr = np.linspace(-0.6,0.6,11)

DT_spec_fit.init_modeone_model(P1_arr)

E = np.linspace(1.0e6,8.0e6,400)
rhoL = 1.50
P1   = 0.2
vbar = 0.0e3
dv   = 200e3
fT   = 0.7
fD   = 0.3
Yn   = 1.0

E_data = np.linspace(1.0e6,9.0e6,1000)
data   = DT_spec_fit.model(E_data,rhoL,P1,vbar,dv,fT,fD,Yn)
np.random.seed(1)
data  *= np.random.normal(loc=1.0,scale=0.05,size=E_data.shape[0])

from scipy.optimize import minimize

def fit_maximum_a_posterior(E, spec, modelfunc, fit_range_func, initial, params, relative_err, bounds):
    idx_cut = fit_range_func(E)

    specerr = relative_err * spec
    soln = minimize(neg_log_likelihood, initial, args=(modelfunc, params, idx_cut, E, spec, specerr), bounds=bounds,
                    method='TNC', tol=1e-9)

    loss_soln = neg_log_likelihood(soln.x, modelfunc, params, idx_cut, E, spec, specerr) / len(idx_cut)
    loss_init = neg_log_likelihood(initial, modelfunc, params, idx_cut, E, spec, specerr) / len(idx_cut)
    return soln, loss_soln, loss_init


def neg_log_likelihood(theta, modelfunc, params, idx_cut, x, y, yerr):
    p = [*theta, *params]
    model = modelfunc(x, *p)
    # Chi^2
    return 0.5 * np.sum(((y[idx_cut] - model[idx_cut]) / yerr[idx_cut]) ** 2)


def fit_range(x):
    return (((x < 8.0e6) & (x > 3.0e6)))


# We don't know the areal density, mode 1 amplitude, and edge parameters
initial_guess = [1.500,0.0,0.0e3,150e3]
# Absolute fit tolerances to test
tolerances = [0.1,0.05,10e3,50e3]
# We know the fuel fractions and DT yield
params = [0.7,0.3,1.0]
bounds = ((0.0,np.inf),(-0.5,0.5),(-np.inf,np.inf),(50e3,np.inf))

soln,loss_soln,loss_init = fit_maximum_a_posterior(E_data,data,DT_spec_fit.model,fit_range
                                                   ,initial_guess,params,0.05,bounds)

true_values = [rhoL,P1,vbar,dv]
for i in range(len(true_values)):
    print("True: {}, Fit: {}".format(true_values[i],soln.x[i]))
    assert np.isclose(true_values[i],soln.x[i],atol=tolerances[i])

nToF_distance = 20.0 # m
flat_sens = nst.time_of_flight.get_unity_sensitivity()
tophat_10cmthickscintillator = nst.time_of_flight.get_transit_time_tophat_IRF(scintillator_thickness=0.1)
decayingGaussian_10cmthickscintillator = nst.time_of_flight.get_transit_time_tophat_w_decayingGaussian_IRF(scintillator_thickness=0.1,gaussian_FWHM=2.5e-9,decay_time=2e-9)



E_ntof = np.linspace(1.0e6,10.0e6,500)
dNdE = DT_spec_fit.model(E_ntof,rhoL,P1,vbar,dv,fT,fD,Yn)

test_20m_nToF_1 = nst.time_of_flight.nToF(nToF_distance,flat_sens,tophat_10cmthickscintillator)
t_det,normt_det,signal = test_20m_nToF_1.get_signal(E_ntof,dNdE)

test_20m_nToF_2 = nst.time_of_flight.nToF(nToF_distance,flat_sens,decayingGaussian_10cmthickscintillator)
t_det,normt_det,signal = test_20m_nToF_2.get_signal(E_ntof,dNdE)

NLO_pwrlaw_sensitivity    = nst.time_of_flight.get_power_law_NLO(1.5)
NLO_Verbinski_sensitivity = nst.time_of_flight.get_Verbinski_NLO()

E_ntof = np.linspace(0.1e6,20e6,200)

# Produce synthetic data
# Burn-avg temperature and thermal spectrum
Tion = 5.0e3 # eV
DTmean,_,_ = nst.DTprimspecmoments(Tion)
# Fluid velocity mean shift
vhs = 100.0e3 # m/s
DTmean = nst.neutron_velocity_addition(DTmean,vhs)
dEmean = DTmean - nst.E0_DT
# Apparent temperature
Tapp = 5.5e3 # eV
_,_,DTvar = nst.DTprimspecmoments(Tapp)

E_DTspec = np.linspace(12.5e6,15.5e6,200)

dNdE_DT = nst.QBrysk(E_DTspec,DTmean,DTvar)

synthetic_15m_nToF = nst.time_of_flight.nToF(15.0,NLO_Verbinski_sensitivity,decayingGaussian_10cmthickscintillator,normtime_end=10.0)
t_det,normt_det,signal = synthetic_15m_nToF.get_signal(E_DTspec,dNdE_DT)
_,_,perfect_signal = synthetic_15m_nToF.get_signal_no_IRF(E_DTspec,dNdE_DT)
relative_ntof_err = 0.05
absolute_ntof_err = 0.01*np.amax(signal)
np.random.seed(101)
noised_signal = signal*np.random.normal(loc=1.0,scale=relative_ntof_err,size=signal.shape[0])+np.random.normal(scale=absolute_ntof_err,size=signal.shape[0])


# Define fitting function
def DT_peak_fit(t, amp, mean_shift, Tapp, synthetic_nToF, E_arr, E0=nst.E0_DT, apply_IRF=True):
    mean = E0 + mean_shift

    _, _, variance = nst.DTprimspecmoments(Tapp)

    dNdE_DT = amp * nst.QBrysk(E_arr, mean, variance)

    if (apply_IRF):
        _, _, signal = synthetic_nToF.get_signal(E_DTspec, dNdE_DT)
    else:
        _, _, signal = synthetic_nToF.get_signal_no_IRF(E_DTspec, dNdE_DT)

    return signal


# Define fitting routines and likelihood
def fit_maximum_a_posterior(t, signal, modelfunc, fit_range_func, initial, params, absolute_err, relative_err, bounds):
    idx_cut = fit_range_func(t)

    signal_err = relative_err * signal + absolute_err
    soln = minimize(neg_log_likelihood, initial, args=(modelfunc, params, idx_cut, t, signal, signal_err),
                    bounds=bounds,
                    method='TNC')

    return soln


def neg_log_likelihood(theta, modelfunc, params, idx_cut, x, y, yerr):
    p = [*theta, *params]
    model = modelfunc(x, *p)
    # Chi^2
    return 0.5 * np.sum(((y[idx_cut] - model[idx_cut]) / yerr[idx_cut]) ** 2)


def DT_ntof_fit_range(x):
    return (((x < 320.0e-9) & (x > 260.0e-9)))

initial = [1.0,0.0,2.0e3] # amp,mean_shift,Tapp
params  = [synthetic_15m_nToF,E_DTspec]
bounds  = [(0.0,np.inf),(-1.0e6,1.0e6),(0.5e3,40.0e3)] # amp,mean_shift,Tapp

soln_fit = fit_maximum_a_posterior(t_det,noised_signal,DT_peak_fit,DT_ntof_fit_range,initial,params,absolute_ntof_err,relative_ntof_err,bounds)

true_values = [1.0,dEmean,Tapp]
tolerances = [0.1,0.2e4,0.25e3]
for i in range(len(true_values)):
    print("True: {}, Fit: {}".format(true_values[i],soln_fit.x[i]))
    assert np.isclose(true_values[i],soln_fit.x[i],atol=tolerances[i])

plt.figure(dpi=200)

plt.plot(t_det/1e-9,perfect_signal,label='Perfect nToF')
plt.plot(t_det/1e-9,noised_signal,label='Noised signal')

plt.plot(t_det/1e-9,DT_peak_fit(t_det,*soln_fit.x,*params),'k--',label='Fit')
plt.plot(t_det/1e-9,DT_peak_fit(t_det,*soln_fit.x,*params,apply_IRF=False),'k:',label='Fit, no IRF')

plt.xlim(250,350)
plt.xlabel("Time (ns)")
plt.legend(frameon=False)



plt.show()