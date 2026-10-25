import json
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "/home/wilsonhong/projects/stanford-cs336/assignment3-scaling/data/isoflops_curves.json"

with open(FILENAME, "r") as f:
    data = json.load(f)


def curve(x, a, b):
    return a * x + b

def get_opt(C, popt):
    return np.exp(curve(np.log(C), popt[0], popt[1]))

def fit_nopts():
    nopts = {}

    for d in data:
        N = d['parameters']
        L = d['final_loss']
        C = d['compute_budget']
        if C not in nopts or L < nopts[C][0]:
            nopts[C] = (L, N)

    # need log-space fit
    # log(y) = a * log(x) + b
    x_data = [np.log(k) for k in nopts.keys()]
    y_data = [np.log(v[1]) for v in nopts.values()]


    popt, pcov = curve_fit(curve, x_data, y_data, p0=[1.0, 1.0])
    for C in [1e23, 1e24]:
        print(f"{C=}, Nopt={get_opt(C, popt)}")

    plt.figure()
    Cs = np.logspace(np.log10(min(nopts)), 24, 100)
    plt.loglog(list(nopts.keys()), [v[1] for v in nopts.values()], 'o', label='IsoFLOP minima')
    plt.loglog(Cs, get_opt(Cs, popt), '--', label=f'fit: N ∝ C^{popt[0]:.3f}')
    plt.xlabel('Compute budget C (FLOPs)'); plt.ylabel('N_opt (parameters)'); plt.legend()
    plt.savefig('hw/chinchilla_isoflops_N.png', dpi=150)


def fit_dopts():
    dopts = {}

    for d in data:
        N = d['parameters']
        L = d['final_loss']
        C = d['compute_budget'] 
        D = C / (6*N)
        if C not in dopts or L < dopts[C][0]:
            dopts[C] = (L, D)

    # need log-space fit
    # log(y) = a * log(x) + b
    x_data = [np.log(k) for k in dopts.keys()]
    y_data = [np.log(v[1]) for v in dopts.values()]


    popt, pcov = curve_fit(curve, x_data, y_data, p0=[1.0, 1.0])
    for C in [1e23, 1e24]:
        print(f"{C=}, Dopt={get_opt(C, popt)}")

    plt.figure()
    Ds = np.logspace(np.log10(min(dopts)), 24, 100)
    plt.loglog(list(dopts.keys()), [v[1] for v in dopts.values()], 'o', label='IsoFLOP minima')
    plt.loglog(Ds, get_opt(Ds, popt), '--', label=f'fit: D ∝ C^{popt[0]:.3f}')
    plt.xlabel('Compute budget C (FLOPs)'); plt.ylabel('D_opt (tokens)'); plt.legend()
    plt.savefig('hw/chinchilla_isoflops_D.png', dpi=150)


fit_nopts()

fit_dopts()
