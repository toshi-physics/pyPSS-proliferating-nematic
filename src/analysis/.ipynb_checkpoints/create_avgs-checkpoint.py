import numpy as np
import json, argparse, os

initParser = argparse.ArgumentParser(description='model_Q_v_rho_create_avgs')
initParser.add_argument('-s','--save_dir', help='directory to save data')
initargs = initParser.parse_args()
savedir = initargs.save_dir

if not os.path.exists(savedir+'/processed_data/'):
        os.makedirs(savedir+'/processed_data/')

if os.path.isfile(savedir+"/parameters.json"):
    with open(savedir+"/parameters.json") as jsonFile:
          parameters = json.load(jsonFile)

T         = parameters["T"]        # final time
dt_dump   = parameters["dt_dump"]
n_steps   = int(parameters["n_steps"])  # number of time steps

dt        = T / n_steps     # time step size
n_dump    = round(T / dt_dump)

meanrho = np.zeros(n_dump)
stdrho  = np.zeros(n_dump)
meanS   = np.zeros(n_dump)
globS   = np.zeros(n_dump)
globcso = np.zeros(n_dump)
stdS    = np.zeros(n_dump)
meantheta = np.zeros(n_dump)
stdtheta = np.zeros(n_dump)
meanmodvx = np.zeros(n_dump)
meanmodvy = np.zeros(n_dump)
meanmodv = np.zeros(n_dump)

i=0
for n in np.arange(n_dump):
    rho = np.loadtxt(savedir+'/data/rho.csv.{:d}'.format(n), delimiter=',')
    Qxx = np.loadtxt(savedir+'/data/Qxx.csv.{:d}'.format(n), delimiter=',')
    Qxy = np.loadtxt(savedir+'/data/Qxy.csv.{:d}'.format(n), delimiter=',')
    S   = 2*np.sqrt(Qxx**2 + Qxy**2)
    globS[i]   += 2*np.sqrt(np.mean(Qxx)**2 + np.mean(Qxy)**2)
    meanrho[i] += np.average(rho)
    stdrho[i]  += np.std(rho)
    meanS[i]   += np.sum(S*rho)/np.sum(rho)
    stdS[i]    += np.sum(((S-meanS[i])**2) * rho)/np.sum(rho)
    theta = np.arctan2(Qxy, Qxx)/2
    globcso[i] += np.sum(S*np.cos(2*theta)*rho)/np.sum(rho)
    meantheta[i] += np.average(theta)
    stdtheta[i]  += np.std(theta)

    
    vx = np.loadtxt(savedir+'/data/vx.csv.{:d}'.format(n), delimiter=',')
    vy = np.loadtxt(savedir+'/data/vy.csv.{:d}'.format(n), delimiter=',')
    v  = np.sqrt(vx**2 + vy**2)
    meanmodvx[i] += np.average(np.abs(vx))
    meanmodvy[i] += np.average(np.abs(vy))
    meanmodv[i]  += np.average(v)
    i+=1

np.savetxt(savedir+'/processed_data/'+'meanrho.csv', meanrho, delimiter=',')
np.savetxt(savedir+'/processed_data/'+'stdrho.csv', stdrho, delimiter=',')

np.savetxt(savedir+'/processed_data/'+'meanS.csv', meanS, delimiter=',')
np.savetxt(savedir+'/processed_data/'+'stdS.csv', stdS, delimiter=',')

np.savetxt(savedir+'/processed_data/'+'meantheta.csv', meantheta, delimiter=',')
np.savetxt(savedir+'/processed_data/'+'stdtheta.csv', stdtheta, delimiter=',')

np.savetxt(savedir+'/processed_data/'+'globS.csv', globS, delimiter=',')
np.savetxt(savedir+'/processed_data/'+'globcso.csv', globcso, delimiter=',')

np.savetxt(savedir+'/processed_data/'+'meanmodvx.csv', meanmodvx, delimiter=',')
np.savetxt(savedir+'/processed_data/'+'meanmodvy.csv', meanmodvy, delimiter=',')
np.savetxt(savedir+'/processed_data/'+'meanmodv.csv', meanmodv, delimiter=',')
