import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import corner
from forward_mapping6p import forward_mapping

burnin = 21000
x = np.loadtxt('output/probs.dat')
plt.plot(-x[burnin:])
plt.savefig('trace_plot.png')
truths = [1.1167,0.7824,58000.0,20000.0,29000.0,35000.0]
#range = np.array([(0.5*x,1.5*x) for x in truths])
#range = np.array([(0.0,10.0**5) for x in truths])
range = [(0.0,2.0),(0.0,2.0),(0.0,2.0*10.0**5),(0.0,10.0**5),(0.0,10.0**5),(0.0,10.0**5)]
labels = [r'$S_p\;(m^2)$', r'$S_n\;(m^2)$', r'$E_{a_{rep}}\;(J/mol)$', r'$E_{a_{ren}}\;(J/mol)$', r'$E_{a_{dip}}\;(J/mol)$', r'$E_{a_{din}}\;(J/mol)$']
y = np.loadtxt('output/chain.dat')
y = np.reshape(y,(np.int(np.size(y)/6),6))
samples = y[burnin:,:]
q = 0.25
corner.corner(samples, bins = 20, 
labels=labels,
range=range,
quantiles=[q,q,q,q], 
show_titles=True, 
title_kwargs={"fontsize": 12},         
truths=truths)
plt.savefig('posterior.png')

plt.figure()
fm = forward_mapping(ntsteps=16400)
n_samples = 10
temps = np.zeros((n_samples,fm.ntsteps+1))
for index in np.arange(n_samples):
    theta = samples[-index*100,:]
    fm.solve(theta)    
    temps[index,:]= fm.solution_temperature

median = np.median(temps,axis=0)
plt.plot(fm.ti,median,'b',lw=1., label='median temperature')
q1 = np.quantile(temps,0.05,axis=0)
q2 = np.quantile(temps,0.95,axis=0)
#plt.plot(q1,'k',lw=1., label='quantile'+r'$5%$')
#plt.plot(q2,'k',lw=1., label='quantile'+r'$95%$')
plt.fill_between(fm.ti,q1,q2,color='k', alpha=0.25)
fm.solve_true()
plt.plot(fm.ti,fm.solution_temperature,'r', lw=1., label='true temperature')
plt.legend(loc=0)
plt.savefig('pred_temperature.png')
