import numpy as np
from tqdm import tqdm
import json, argparse, os
from scipy.ndimage import gaussian_filter

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *

def main():

    initParser = argparse.ArgumentParser(description='model_Q_v_rho_alpha_newchi')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir
    
    if os.path.isfile(savedir+"/parameters.json"):
	    with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
              
    run       = parameters["run"]
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
    invgamma0 = parameters["invgamma0"]   # rate of Q alignment with mol field H
    zeta0     = parameters["zeta0"]
    epsilon   = parameters["epsilon"]
    alpha     = parameters["alpha"]    # active contractile stress
    chi       = parameters["chi"]      # coefficient of density gradients in Q's free energy
    KQ        = parameters["KQ"]
    epsilon   = parameters["epsilon"]
    lambd     = parameters["lambda"]   # flow alignment parameter
    p0        = parameters["p0"]       # pressure when cells are close packed, should be very high
    rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
    rho_seed  = parameters["rhoseed"] /rho_in     # seeding density, normalised by 100 mm^-2
    rho_iso   = parameters["rhoisoend"] /rho_in   # jamming density
    rho_nem   = parameters["rhonemend"] /rho_in   # jamming density max for nematic
    rj        = parameters["rrhoend"]
    ncluster  = parameters["ncluster"]
    rcluster  = parameters["rcluster"]
    rhocluster= parameters["rhocluster"]/rho_in
    rhobg     = parameters["rhobg"]/rho_in
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])

    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)
    dn_dump   = round(n_steps / n_dump)
    
     # Define the grid size.
    grid_size = np.array([mx, my])
    dr=np.array([dx, dy])

    k_list, k_grids = momentum_grids(grid_size, dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(grid_size, fourier_operators)

    # Create fields that undergo timestepping
    system.create_field('rho', k_list, k_grids, dynamic=True)
    system.create_field('Qxx', k_list, k_grids, dynamic=True)
    system.create_field('Qxy', k_list, k_grids, dynamic=True)
    # Create fields that don't timestep
    system.create_field('Hxx', k_list, k_grids, dynamic=False)
    system.create_field('Hxy', k_list, k_grids, dynamic=False)
    system.create_field('noiseQxx', k_list, k_grids, dynamic=False)
    system.create_field('noiseQxy', k_list, k_grids, dynamic=False)
    system.create_field('rho_J', k_list, k_grids, dynamic=False)
    system.create_field('rho_J_rho', k_list, k_grids, dynamic=False)
    system.create_field('rho_by_rho_J', k_list, k_grids, dynamic=False)

    system.create_field('pressure', k_list, k_grids, dynamic=False)
    system.create_field('iqxp', k_list, k_grids, dynamic=False)
    system.create_field('iqyp', k_list, k_grids, dynamic=False)

    system.create_field('Ident', k_list, k_grids, dynamic=False)
    system.create_field('Pxx', k_list, k_grids, dynamic=False)
    system.create_field('Pxy', k_list, k_grids, dynamic=False)
    system.create_field('S2', k_list, k_grids, dynamic=False)

    system.create_field('iqxrho', k_list, k_grids, dynamic=False)
    system.create_field('iqyrho', k_list, k_grids, dynamic=False)

    system.create_field('q2Qxx', k_list, k_grids, dynamic=False)
    system.create_field('q2Qxy', k_list, k_grids, dynamic=False)
    system.create_field('q4Qxx', k_list, k_grids, dynamic=False)
    system.create_field('q4Qxy', k_list, k_grids, dynamic=False)

    system.create_field('alphaf', k_list, k_grids, dynamic=False)
    system.create_field('vRHSx', k_list, k_grids, dynamic=False)
    system.create_field('vRHSy', k_list, k_grids, dynamic=False)
    system.create_field('detgamma', k_list, k_grids, dynamic=False)

    system.create_field('vx', k_list, k_grids, dynamic=False)
    system.create_field('vy', k_list, k_grids, dynamic=False)

    system.create_field('kappa_a_xy', k_list, k_grids, dynamic=False)
    system.create_field('kappa_s_xx', k_list, k_grids, dynamic=False)
    system.create_field('kappa_s_xy', k_list, k_grids, dynamic=False)
    
    # Create equations, if no function of rho, write None. If function and argument, supply as a tuple. 
    # Write your own functions in the function library or use numpy functions
    # if using functions that need no args, like np.tanh, write [("fieldname", (np.tanh, None))]
    # for functions with an arg, like, np.power(fieldname, n), write [("fieldname", (np.power, n))]
    # Define Identity # The way its written if you don't define a RHS, the LHS becomes zero at next timestep for Static Fields
    system.create_term("Ident", [("Ident", None)], [1, 0, 0, 0])
    
    system.create_term("Pxx", [("Pxx", None)], [1, 0, 0, 0]) #fix substrate orientation
    system.create_term("Pxy", [("Pxy", None)], [1, 0, 0, 0])

    # Define S2
    system.create_term("S2", [("Qxx", (np.square, None))], [4.0, 0, 0, 0])
    system.create_term("S2", [("Qxy", (np.square, None))], [4.0, 0, 0, 0])
    # Define RhoEnd
    system.create_term("rho_J", [("Ident", None)], [rho_iso, 0, 0, 0])
    system.create_term("rho_J", [("S2", None)], [(rho_nem-rho_iso), 0, 0, 0])
    # Define rho_end minus rho
    system.create_term("rho_J_rho", [("rho_J", None)], [1, 0, 0, 0])
    system.create_term("rho_J_rho", [("rho", None)], [-1, 0, 0, 0])
    # Define rho by rho_end
    system.create_term("rho_by_rho_J", [("rho", None), ("rho_J", (np.power, -1))], [1, 0, 0, 0])
    # Define Pressure
    system.create_term("pressure", [("rho_by_rho_J", (np.exp, None))], [p0, 0, 0, 0])
    system.create_term("iqxp", [("pressure", None)], [1, 0, 1, 0])
    system.create_term("iqyp", [("pressure", None)], [1, 0, 0, 1])
    # Define iqxrho and so on
    system.create_term("iqxrho", [("rho", None)], [1, 0, 1, 0])
    system.create_term("iqyrho", [("rho", None)], [1, 0, 0, 1])

    system.create_term("q2Qxx", [("Qxx", None)], [1, 1, 0, 0])
    system.create_term("q2Qxy", [("Qxy", None)], [1, 1, 0, 0])
    system.create_term("q4Qxx", [("Qxx", None)], [1, 2, 0, 0])
    system.create_term("q4Qxy", [("Qxy", None)], [1, 2, 0, 0])

    # Define Hxx
    system.create_term("Hxx", [("Qxx", None)], [-1, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("Qxx", None)], [1, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("S2", None), ("Qxx", None)], [-1, 0, 0, 0])
    system.create_term("Hxx", [("S2", None), ("Qxx", None)], [-1, 0, 0, 0])
    system.create_term("Hxx", [("q2Qxx", None), ("rho", None)], [-K, 0, 0, 0])
    system.create_term("Hxx", [("iqxrho", (np.power, 2))], [-chi/2, 0, 0, 0])
    system.create_term("Hxx", [("iqyrho", (np.power, 2))], [chi/2, 0, 0, 0])
    # Define Hxy
    system.create_term("Hxy", [("Qxy", None)], [-1, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("Qxy", None)], [1, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("S2", None), ("Qxy", None)], [-1, 0, 0, 0])
    system.create_term("Hxy", [("S2", None), ("Qxy", None)], [-1, 0, 0, 0])
    system.create_term("Hxy", [("q2Qxy", None), ("rho", None)], [-K, 0, 0, 0])
    system.create_term("Hxy", [("iqxrho", None), ("iqyrho", None)], [-chi, 0, 0, 0])
    # Define noise in Q
    system.create_term("noiseQxx", [("noiseQxx", None)], [1, 0, 0, 0])
    system.create_term("noiseQxy", [("noiseQxy", None)], [1, 0, 0, 0])
    
    # Define alphaf
    system.create_term("alphaf", [("rho", None), ("rho_J_rho", None), ("rho_J_rho", (np.heaviside, 0))], [alpha, 0, 0, 0])
    # Define vx
    system.create_term("vRHSx", [('Qxx', None), ("alphaf", None)], [1/zeta0, 0, 1, 0])
    system.create_term("vRHSx", [('Qxy', None), ("alphaf", None)], [1/zeta0, 0, 0, 1])
    system.create_term("vRHSx", [('iqxp', None)], [-1/zeta0, 0, 0, 0])
    # Define vy 
    system.create_term("vRHSy", [('Qxy', None), ("alphaf", None)], [1/zeta0, 0, 1, 0])
    system.create_term("vRHSy", [('Qxx', None), ("alphaf", None)], [-1/zeta0, 0, 0, 1])
    system.create_term("vRHSy", [('iqyp', None)], [-1/zeta0, 0, 0, 0])
    # Define vx
    system.create_term("vx", [('vRHSx', None), ('rho', (np.power, -1))], [1, 0, 0, 0])
    system.create_term("vx", [('vRHSx', None), ('Pxx', None), ('rho', (np.power, -1))], [epsilon, 0, 0, 0])
    system.create_term("vx", [('vRHSy', None), ('Pxy', None), ('rho', (np.power, -1))], [epsilon, 0, 0, 0])
    # Define vy
    system.create_term("vy", [('vRHSy', None), ('Ident', None), ('rho', (np.power, -1))], [1, 0, 0, 0])
    system.create_term("vy", [('vRHSy', None), ('Pxx', None), ('rho', (np.power, -1))], [-epsilon, 0, 0, 0])
    system.create_term("vy", [('vRHSx', None), ('Pxy', None), ('rho', (np.power, -1))], [epsilon, 0, 0, 0])
    # Define kappa_a_xy
    system.create_term("kappa_a_xy", [("vx", None)], [0.5, 0, 0, 1]) # iqy vx / 2
    system.create_term("kappa_a_xy", [("vy", None)], [-0.5, 0, 1, 0]) # -iqx vy / 2
    # Define kappa_s_xx
    system.create_term("kappa_s_xx", [("vx", None)], [0.5, 0, 1, 0])
    system.create_term("kappa_s_xx", [("vy", None)], [-0.5, 0, 0, 1])
    # Define kappa_s_xy
    system.create_term("kappa_s_xy", [("vx", None)], [0.5, 0, 0, 1])
    system.create_term("kappa_s_xy", [("vy", None)], [0.5, 0, 1, 0])

    # Create terms for rho timestepping
        # growth
    system.create_term("rho", [("rho", None)], [1, 0, 0, 0])
    system.create_term("rho", [("rho", (np.power, 2)), ('rho_J', (np.power, -1))], [-1/rj, 0, 0, 0])
        # advection
    system.create_term("rho", [("vx", None), ("rho", None)], [-1, 0, 1, 0])
    system.create_term("rho", [("vy", None), ("rho", None)], [-1, 0, 0, 1])
    
    # Create terms for Qxx timestepping
    system.create_term("Qxx", [("Hxx", None), ("rho_J_rho", None), ("rho_J_rho", (np.heaviside, 0))], [invgamma0, 0, 0, 0])
    system.create_term("Qxx", [("vx", None), ("Qxx", None)], [-1, 0, 1, 0])
    system.create_term("Qxx", [("vy", None), ("Qxx", None)], [-1, 0, 0, 1])
    system.create_term("Qxx", [("Qxy", None), ("kappa_a_xy", None)], [2, 0, 0, 0])
    system.create_term("Qxx", [("kappa_s_xx", None), ("rho_J_rho", (np.heaviside, 0))], [lambd, 0, 0, 0])
    system.create_term("Qxx", [("noiseQxx", None)], [1, 0, 0, 0])

    # Create terms for Qxy timestepping
    system.create_term("Qxy", [("Hxy", None), ("rho_J_rho", None), ("rho_J_rho", (np.heaviside, 0))], [invgamma0, 0, 0, 0])
    system.create_term("Qxy", [("vx", None), ("Qxy", None)], [-1, 0, 1, 0])
    system.create_term("Qxy", [("vy", None), ("Qxy", None)], [-1, 0, 0, 1])
    system.create_term("Qxy", [("Qxx", None), ("kappa_a_xy", None)], [-2, 0, 0, 0])
    system.create_term("Qxy", [("kappa_s_xy", None), ("rho_J_rho", (np.heaviside, 0))], [lambd, 0, 0, 0])
    system.create_term("Qxy", [("noiseQxy", None)], [1, 0, 0, 0])

    rho     = system.get_field('rho')
    Qxx     = system.get_field('Qxx')
    Qxy     = system.get_field('Qxy')
    vx      = system.get_field('vx')
    vy      = system.get_field('vy')
    pressure  = system.get_field('pressure')
    rhoJrho = system.get_field('rho_J_rho')

    # set init condition and synchronize momentum with the init condition, important!!

    nclusters = ncluster * np.int32(rho_seed/(50/rho_in)) #there's about 10 clusters when plated at 50cells/mm^2, scale appropriately
    rhobgs     = rhobg * rho_seed/(50/rho_in)
    set_rho_Q_islands(rho, Qxx, Qxy, nclusters, rcluster, rhocluster, rhobgs, rho_seed, rho_nem*rj, grid_size)
    while np.sum(np.where(rho.get_real()<=0, 1, 0))>0:
        set_rho_islands(rho, ncluster, rcluster, rho_seed, grid_size)
    
    #Initialize Pxx and Pxy into whatever pattern the substrate is in
    system.get_field('Pxx').set_real(np.ones([mx, my])) #properly normalized
    system.get_field('Pxx').synchronize_momentum()
    system.get_field('Pxy').set_real(np.zeros([mx, my]))
    system.get_field('Pxy').synchronize_momentum()

    # Initialise identity matrix 
    system.get_field('Ident').set_real(np.ones(shape=grid_size))
    system.get_field('Ident').synchronize_momentum()

    system.get_field('detgamma').set_real(np.ones(shape=grid_size))
    system.get_field('detgamma').synchronize_momentum()

    # Initial Conditions for rho_J
    system.get_field('rho_J').set_real(rho_iso*np.ones(shape=grid_size))
    system.get_field('rho_J').synchronize_momentum()
    # Initial Conditions for rho_J_rho
    rhoJrho.set_real(rho_iso*np.ones(shape=grid_size) - rho.get_real())
    rhoJrho.synchronize_momentum()
    # Initialise Pressure
    pressure.set_real(p0*np.exp(rho.get_real()))
    pressure.synchronize_momentum()

    if not os.path.exists(savedir+'/data/'):
        os.makedirs(savedir+'/data/')

    for t in tqdm(range(n_steps)):
        system.get_field('noiseQxx').set_real(KQ*rho.get_real()*rhoJrho.get_real()*np.heaviside(rhoJrho.get_real(), 0)*np.random.normal(size=[mx, my]))
        system.get_field('noiseQxx').synchronize_momentum()
        
        system.get_field('noiseQxy').set_real(KQ*rho.get_real()*rhoJrho.get_real()*np.heaviside(rhoJrho.get_real(), 0)*np.random.normal(size=[mx, my]))
        system.get_field('noiseQxy').synchronize_momentum()
        
        system.update_system(dt)

        if t % dn_dump == 0:
            np.savetxt(savedir+'/data/'+'rho.csv.'+ str(t//dn_dump), rho.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxx.csv.'+ str(t//dn_dump), Qxx.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxy.csv.'+ str(t//dn_dump), Qxy.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'vx.csv.'+ str(t//dn_dump), vx.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'vy.csv.'+ str(t//dn_dump), vy.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'vorticity.csv.'+ str(t//dn_dump), -system.get_field('kappa_a_xy').get_real(), delimiter=',')

def momentum_grids(grid_size, dr):
    k_list = [np.fft.fftfreq(grid_size[i], d=dr[i])*2*np.pi for i in range(len(grid_size))]
    # k is now a list of arrays, each corresponding to k values along one dimension.

    k_grids = np.meshgrid(*k_list, indexing='ij')
    #k_grids = np.meshgrid(*k_list, indexing='ij', sparse=True)
    # k_grids is now a list of 2D sparse arrays, each corresponding to k values in one dimension.

    return k_list, k_grids

def k_power_array(k_grids):
    k_squared = sum(ki**2 for ki in k_grids)
    #k_squared = k_grids[0]**2 + k_grids[1]**2
    k_abs = np.sqrt(k_squared)
    #inv_kAbs = np.divide(1.0, k_abs, where=k_abs!=0)

    k_power_arrays = [k_squared, 1j*k_grids[0], 1j*k_grids[1]]

    return k_power_arrays

def set_rho_Q_islands(rhofield, Qxxfield, Qxyfield, ncluster, rcluster, rhocluster, rhobg, rhoseed, rho_max, grid_size):
    centers = grid_size[0]*np.random.rand(ncluster,2); radii = rcluster
    angles = np.random.rand(ncluster) * np.pi
    tol = 0.001
    x   = np.arange(0+tol, grid_size[0]-tol, 1)
    y   = np.arange(0+tol, grid_size[1]-tol, 1)
    r   = np.meshgrid(x,y)

    rhoinit = np.zeros(grid_size)
    itheta = np.zeros(grid_size)
    n_overlap = np.zeros(grid_size)
    
    for i in np.arange(ncluster):
        distance = np.sqrt((r[0]-centers[i,0])**2+(r[1]-centers[i,1])**2)
        rhoinit += np.where(distance < radii, rhocluster, rhobg*np.random.rand(grid_size[0], grid_size[1])/ncluster) #numbers taken from expts
        itheta += np.where(distance < radii, angles[i], 0)
        n_overlap += np.where(distance < radii, 1, 0)

    mask = n_overlap>0
    itheta[mask] /= n_overlap[mask]

    meanrho = np.average(rhoinit)
    rhoinit = rhoinit * rhoseed / meanrho
    
    # if density is higher than nematic jamming density, reduce it. To avoid crashing.
    rhoinit = np.where(rhoinit>rho_max, rho_max, rhoinit)
    rhoinit = gaussian_filter(rhoinit, radii, mode='wrap') #kernel size = radii

    meanrho = np.average(rhoinit)
    rhoinit = rhoinit * rhoseed / meanrho
    
    rhofield.set_real(rhoinit)
    rhofield.synchronize_momentum()

    Sinit = np.where(rhoinit>1, np.sqrt(np.abs(rhoinit-1)/(rhoinit+1))/2, 0.01)

    itheta = np.where(itheta==0, np.random.rand(grid_size[0], grid_size[1])*np.pi, itheta)

    Qxxfield.set_real(Sinit*(np.cos(2*itheta))/2)
    Qxxfield.synchronize_momentum()
    Qxyfield.set_real(Sinit*(np.sin(2*itheta))/2)
    Qxyfield.synchronize_momentum()


if __name__=="__main__":
    main()
