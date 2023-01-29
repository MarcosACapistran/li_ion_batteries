#coding: utf8
import time
import numpy as np
from numba import njit,jit, jitclass
from scipy import stats as ss
from forward_mapping6p import forward_mapping
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2021)
    
if __name__=="__main__":
    fm = forward_mapping(ntsteps=4100)
    fm.solve_true()
    
    V = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])
    where_are_NaNs = np.isnan(V)
    V[where_are_NaNs] = 0.0
    #data = V[30750:-1000:100]
    #data = V[:-1000:200]
    #data = V[:-1000:100]
    #data = V[0::100]
    data = V[0::75]
    sigma = np.max(data)/100.0
    data += sigma*np.random.normal(size=len(data))
        
    # NECESITAS ESTA FUNCION
    def distance(p):
        fm.solve(p)
        V = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])
        where_are_NaNs = np.isnan(V)
        V[where_are_NaNs] = 0.0
        #term1 = -np.sum(ss.norm.logpdf(data,loc=V[30750:-1000:100],scale=sigma))
        #term1 = -np.sum(ss.norm.logpdf(data,loc=V[:-1000:200],scale=sigma))
        #term1 = -np.sum(ss.norm.logpdf(data,loc=V[:-1000:100],scale=sigma))
        #term1 = -np.sum(ss.norm.logpdf(data,loc=V[0::100],scale=sigma))
        term1 = -np.sum(ss.norm.logpdf(data,loc=V[0::75],scale=sigma))
        return term1
    
    def root(p):
        fm.solve(p)
        V = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])
        np.warnings.filterwarnings('ignore')
        index = np.nonzero(V>3.0)[-1][-1]
        return fm.ti[index]

    def root_temp(p):
        fm.solve(p)
        np.warnings.filterwarnings('ignore')
        where_are_NaNs = np.isnan(fm.solution_temperature)
        fm.solution_temperature[where_are_NaNs] = 0.0
        return np.max(fm.solution_temperature)
        
    #NECESITAS ESTA FUNCION
    def root_evaluate(values):
        Y = np.zeros([values.shape[0]])    
        for i,p in enumerate(values):
            #Y[i]=root(p)
            #Y[i]=root_temp(p)            
            Y[i]=distance(p)
        return Y

    #NECESITAS ESTE ANALISIS
    from SALib.sample import saltelli
    from SALib.analyze import sobol    
    """
    Carry out Sobol sensitivity analysis in an hypecube of plus-minus twenty percent
    """
    pct = 0.25
    
    problem = {
    'num_vars': 6,
    'names': ['Sp', 'Sn', 'Earep', 'Earen', 'Eadip', 'Eadin'],
    'bounds': [    
    [1.1167*(1-pct), 1.1167*(1+pct)],
    [0.7824*(1-pct), 0.7824*(1+pct)],
    [58000.*(1-pct), 58000.*(1+pct)],    
    [20000.*(1-pct), 20000.*(1+pct)],
    [29000.*(1-pct), 29000.*(1+pct)],    
    [35000.*(1-pct), 35000.*(1+pct)]]
    }
    
    param_values = saltelli.sample(problem, 1000)
    Y = root_evaluate(param_values)
    Si = sobol.analyze(problem, Y, print_to_console=False)
    np.savetxt('S1.txt',Si['S1'])
    np.savetxt('S1_conf.txt',Si['S1_conf'])
