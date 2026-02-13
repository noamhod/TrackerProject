#!/usr/bin/python
import os
import math
import subprocess
import time
import array
import numpy as np
from iminuit import Minuit

from tracker_lib import config, utils

### --- Constants ---
X0_Si     = 93.7  # mm (Radiation length of Silicon)
Thickness = 0.05  # mm (Sensor thickness)



def highland_log_theta_sq(p_MeV):
    ### calculates expected scattering variance (theta^2) for a given momentum using the Highland formula.
    x_x0 = Thickness / X0_Si ### radiation length fraction
    # Highland formula: theta = (13.6/p) * sqrt(x/X0) * [1 + 0.038 ln(x/X0)]
    theta_rms = (13.6/p_MeV) * np.sqrt(x_x0) * (1 + 0.038*np.log(x_x0))
    return np.log(theta_rms**2)



def chi2_vectorized(params, x, y, z, ex, ey, ez):
    ### params: [x0, mx, y0, my], representing: x = x0 + mx*t, y = y0 + my*t, z = t
    ### point on line at z=0 and z=1, direction vector u = (mx, my, 1)
    ux, uy, uz = params[1], params[3], 1.0
    mag_u = np.sqrt(ux**2 + uy**2 + uz**2)
    ### unit direction
    ux /= mag_u
    uy /= mag_u
    uz /= mag_u
    ### vectors from point on line (x0, y0, 0) to data points (xi, yi, zi)
    dx = x - params[0]
    dy = y - params[2]
    dz = z - 0.0
    ### cross product v = (dx, dy, dz) X (ux, uy, uz)
    vx = dy * uz - dz * uy
    vy = dz * ux - dx * uz
    # vz = dx * uy - dy * ux # Usually not needed if errors are only in XY
    ### weighted chi2
    return np.sum((vx / ex)**2 + (vy / ey)**2)



def build_covariance_matrix_structure(z_pos):
    ### Pre-calculates the geometric part of the scattering matrix.
    ### V_ms = theta^2 * C_geom
    ### Returns C_geom matrix.
    N = len(z_pos)
    C_geom = np.zeros((N, N))
    ### assume scattering happens AT the detector plane k.
    ### deviation at i, j > k is (z_i - z_k)*(z_j - z_k)
    for k in range(N):
        for i in range(k + 1, N):
            for j in range(k + 1, N):
                value = (z_pos[i] - z_pos[k]) * (z_pos[j] - z_pos[k])
                C_geom[i, j] += value
                if i != j:  # Symmetrize matrix by filling lower triangle
                    C_geom[j, i] += value
    return C_geom



def negative_log_likelihood(params, x, y, z, ex, ey, C_geom):
    ### Objective function: -2 * ln(L)
    ### params: [x0, mx, y0, my, log_theta_sq]
    x0, mx, y0, my, log_theta_sq = params
    
    ### hard clip to prevent overflow - this acts as a safety valve before the math starts
    log_theta_sq = np.clip(log_theta_sq, -30, -6)
    theta_sq = np.exp(log_theta_sq) ### fit log to force positive variance
    
    ### build Total Covariance V = V_meas + theta^2 * C_geom
    ### V is block diagonal (V_x, V_y) if x/y errors are independent
    ### sum the log-likelihoods for X and Y components
    epsilon = 1e-6 ### tiny regularization term
    Vx = np.diag(ex**2) + theta_sq * C_geom + np.eye(len(x)) * epsilon
    Vy = np.diag(ey**2) + theta_sq * C_geom + np.eye(len(x)) * epsilon
    if(not np.all(np.isfinite(Vx)) or not np.all(np.isfinite(Vy))): return 1e15
    
    ### invert and Determinant, for small matrices (5x5), standard linalg is fine.
    ### use pseudo-inverse for stability against perfectly straight tracks (theta->0)
    ### --- solve for X ---
    ### residuals
    x_pred = x0 + mx * z
    rx = x - x_pred
    ### invert Vx
    try:
        sign_x, logdet_x = np.linalg.slogdet(Vx)
        if sign_x <= 0 or np.isnan(logdet_x):  # SAFETY: Matrix must be positive-definite
            return 1e15
        chi2_x = rx.T @ np.linalg.solve(Vx, rx)
    except(np.linalg.LinAlgError, ValueError, RuntimeWarning):
        return 1e15 ### penalty for singular matrix
    ### --- solve for Y ---
    ### residuals
    y_pred = y0 + my * z
    ry = y - y_pred
    try:
        sign_y, logdet_y = np.linalg.slogdet(Vy)
        if sign_y <= 0 or np.isnan(logdet_y):  # SAFETY: Matrix must be positive-definite
            return 1e15
        chi2_y = ry.T @ np.linalg.solve(Vy, ry)
    except(np.linalg.LinAlgError, ValueError, RuntimeWarning):
        return 1e15 ### penalty for singular matrix

    ### total NLL = Chi2 + LogDet
    nll = (chi2_x + logdet_x) + (chi2_y + logdet_y)

    return nll



def fit_line_3d_mle(x, y, z, ex, ey, ez, fixtheta2=False):
    # pre-compute geometry (constant for all calls)
    C_geom = build_covariance_matrix_structure(z)
    
    ### initial Guess, standard Least Squares for line parameters
    mx_guess = (x[-1] - x[0]) / (z[-1] - z[0])
    my_guess = (y[-1] - y[0]) / (z[-1] - z[0])
    x0_guess = x[0] - mx_guess * z[0]
    y0_guess = y[0] - my_guess * z[0]
    
    ### guess for scattering: start with a "typical" value (e.g. 2.5 GeV)
    log_theta_sq_guess = highland_log_theta_sq(p_MeV=2.5e3)
    initial_params = [x0_guess, mx_guess, y0_guess, my_guess, log_theta_sq_guess]
    
    ### minimization
    m = Minuit(lambda p: negative_log_likelihood(p, x, y, z, ex, ey, C_geom), initial_params)
    # m = Minuit(cost_func, x0=initial_params[0], mx=initial_params[1], y0=initial_params[2], my=initial_params[3])
    
    # define physical limits for log_theta_sq
    # lower bound: ~ -25 (theta ~ 1e-7 rad, effectively 0 scattering)
    # upper bound: ~ -6   (theta = 5e-2 rad, massive scattering but still OK)
    # m.limits["x4"] = (-25,-6)  ### "x4" is the 5th parameter (index 4)
    m.limits["x4"] = (-30,-6)  ### "x4" is the 5th parameter (index 4)
    
    ### set step sizes
    m.errors = [0.1, 0.001, 0.1, 0.001, 1.0] 
    m.errordef = Minuit.LIKELIHOOD # important: errordef=0.5 for NLL
    
    ### for the alkignment case, it has to be fixed!
    if(fixtheta2): m.fixed["x4"] = True
    ### push the parameters into the "good region" where Migrad can work.
    m.simplex()
    ### run the migrad minimizer
    m.migrad()
    ### calculate the reliable uncertainties (Hessian)
    m.hesse()
    ### extract results to match your original return structure
    params = np.array(m.values)
    parerr = np.array(m.errors) # the calculated statistical errors from Hesse
    parcov = None
    if(m.covariance is not None): parcov = m.covariance.tolist()
    else:                         parcov = np.zeros((len(params), len(params))).tolist()  ### return zeros if failed
    nll     = m.fval
    success = m.valid
    chisq   = chi2_vectorized(params[:4], x,y,z, ex,ey,ez) ### chi2_vectorized expects only [x0, mx, y0, my], not log_theta_sq
    ndof    = 2 * len(x) - len(params)
    return params, parerr, parcov, nll, chisq, ndof, success



def fit_3d_mle(points,errors):
    cfg = config.Config().map
    x = points[0]
    y = points[1]
    z = points[2]
    ex = errors[0]
    ey = errors[1]
    ez = errors[2]
    params,parerr,parcov,nll,chisq,ndof,success = fit_line_3d_mle(x,y,z,ex,ey,ez)
    theta2  = np.exp(params[4])
    x0,y0,z0 = utils.line(cfg["world"]["z"][0], params)
    x1,y1,z1 = utils.line(cfg["world"]["z"][1], params)
    #TODO: need to check this:
    xm,ym,zm = utils.line((cfg["world"]["z"][1]-cfg["world"]["z"][0])/2., params) #TODO
    centroid  = [xm,ym,zm]                     #TODO
    direction = [x1-x0,y1-y0,z1-z0]            #TODO
    return theta2,nll,chisq,ndof,direction,centroid,params,parerr,parcov,success