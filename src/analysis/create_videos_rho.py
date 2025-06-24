import numpy as np
import json, argparse, os
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from matplotlib.widgets import TextBox


def main():

    initParser = argparse.ArgumentParser(description='model_Q_v_rho_create_videos')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir

    if not os.path.exists(savedir+'/videos/'):
        os.makedirs(savedir+'/videos/')
    
    if os.path.isfile(savedir+"/parameters.json"):
        with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
    
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)    
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])
    Lx        = mx*dx
    Ly        = my*dy
    rhoseed   = parameters["rhoseed"]
    rhonemend = parameters["rhonemend"]
    rho_in    = parameters["rho_in"]
    if parameters.get("rrhoend"):
        rrhoend = parameters["rrhoend"]
    else:
        rrhoend=1
    
    #setup a meshgrid
    tol = 0.001
    
    x   = np.linspace(0+tol, Lx-tol, mx)
    y   = np.linspace(0+tol, Ly-tol, my)
    xv, yv  = np.meshgrid(x,y, indexing='ij')
    
    times = np.arange(0, n_dump, 1)*dt_dump

    
    figrho, axrho= plt.subplots(figsize=(12, 8), ncols=1)
    figQ, axQ= plt.subplots(figsize=(12, 8), ncols=1)

    n=1
    p_factor = np.int32(mx/39)
    
    rho = np.loadtxt(savedir+'/data/'+'rho.csv.{:d}'.format(n), delimiter=',')
    Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(n), delimiter=',')
    Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(n), delimiter=',')
    #pixelate fields because the point density is too high
    #vx = pixelate(vx, p_factor)
    #vy = pixelate(vy, p_factor)
    pxv = xv[::p_factor, ::p_factor]
    pyv = yv[::p_factor, ::p_factor]
    S = 2*np.sqrt(Qxx**2+Qxy**2)
    theta = np.arctan2(Qxy, Qxx)/2
    nx    = np.cos(theta) [::p_factor, ::p_factor]
    ny    = np.sin(theta) [::p_factor, ::p_factor]
    Snx   = S [::p_factor, ::p_factor] * nx
    Sny   = S [::p_factor, ::p_factor] * ny
    vscale = 0.05
    nscale = 0.15

    #pcolors = np.ones_like(pxv) * (1- np.mean(rho)/(rrhoend*rhonemend/rho_in))
    
    crho = [axrho.pcolormesh(xv, yv, rho, cmap='viridis', vmin=0, vmax=rrhoend*rhonemend/rho_in), axrho.quiver(pxv, pyv, Snx, Sny, color='white', pivot='middle', headlength=0, headaxislength=0, scale=nscale, scale_units='xy')]
    cQ   = [axQ.pcolormesh(xv, yv, S, cmap='viridis', vmin=0, vmax=0.7), axQ.quiver(pxv, pyv,nx, ny, color='k', pivot='middle', headlength=0, headaxislength=0)]

    figrho.colorbar(crho[0])
    axrho.set_title(r"$\rho$")
    figQ.colorbar(cQ[0])
    axQ.set_title('S')
    
    tbaxrho = figrho.add_axes([0.2, 0.93, 0.04, 0.04])
    tbrho = TextBox(tbaxrho, 'time')
    tbaxQ = figQ.add_axes([0.2, 0.93, 0.04, 0.04])
    tbQ = TextBox(tbaxQ, 'time')    
    
    def plt_snapshot_rho(val):        
        rho = np.loadtxt(savedir+'/data/'+'rho.csv.{:d}'.format(val), delimiter=',')
        Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(val), delimiter=',')
        Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(val), delimiter=',')
        S = np.sqrt(Qxx**2+Qxy**2) 
        theta = np.arctan2(Qxy, Qxx)/2
        Snx    = (S*np.cos(theta)) [::p_factor, ::p_factor]
        Sny    = (S*np.sin(theta)) [::p_factor, ::p_factor]

        #pcolors = np.ones_like(pxv) * (1-(np.mean(rho)/(rrhoend*rhonemend/rho_in)))
        
        crho[0].set_array(rho)
        crho[1].set_UVC(Snx, Sny)
        tbrho.set_val(round(times[val],2))
        
        figrho.canvas.draw_idle()
    
    def plt_snapshot_Q(val):        
        Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(val), delimiter=',')
        Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(val), delimiter=',')
        S = np.sqrt(Qxx**2+Qxy**2)
        theta = np.arctan2(Qxy, Qxx)/2
        nx    = np.cos(theta)
        ny    = np.sin(theta)
        
        cQ[0].set_array(S)
        cQ[1].set_UVC(nx[::p_factor, ::p_factor], ny[::p_factor, ::p_factor])
        tbQ.set_val(round(times[val],2))

        figQ.canvas.draw_idle()

    
    from matplotlib.animation import FuncAnimation
    animrho = FuncAnimation(figrho, plt_snapshot_rho, frames = n_dump, interval=100, repeat=True)
    animrho.save(savedir+'/videos/'+'rho.mp4')

    animQ = FuncAnimation(figQ, plt_snapshot_Q, frames = n_dump, interval=100, repeat=True)
    animQ.save(savedir+'/videos/'+'Q.mp4')

if __name__=="__main__":
    main()
