#coding: utf8
import time
import resource 
import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#plt.rcParams.update({'font.size': 16})
plt.style.use('seaborn-talk')
import warnings
warnings.filterwarnings("ignore")

class forward_mapping:
    
    """
    This class simulates the li-ion battery forward problem
    """
    
    def __init__(self, T0 = 298.15, N=100, ntsteps=41000, tf=4100.0):
        
        """
        Guo parameters with j = p
        """
        self.Lp = 0.00007           # m
        self.Sp = 1.1167            # m**2
        self.cspmax = 51410         # mol/m**3
        self.Rp = 8.5*10.0**-6      # m
        self.kpref = 6.6667*10**-11 # m**2.5*mol**(-0.5)/s
        self.Earep = 58000.0        # J/mol (OJO: Estaba en kJ/mol)
        self.Dspref = 1.0*10**-14     # m**2/s
        self.Eadip = 29000.0        # J/mol (OJO: Estaba en kJ/mol)
        self.alphaap = 0.5
        self.alphacp = 0.5
        
        """
        Guo parameters with j = n
        """
        
        self.Ln = 0.0000735         # m
        self.Sn = 0.7824            # m**2
        self.csnmax = 31833         # mol/m**3
        self.Rn = 12.5*10.0**-6     # m
        self.knref = 1.764*10**-11  # m**2.5*mol**(-0.5)/s
        self.Earen = 20000.0        # J/mol (OJO: Estaba en kJ/mol)
        self.Dsnref = 3.9*10**-14     # m**2/s
        self.Eadin = 35000.0        # J/mol (OJO: Estaba en kJ/mol)
        self.alphaan = 0.5
        self.alphacn = 0.5
        
        """
        Rest of Guo parameters
        """
        
        self.ce = 1000.0                # mol/m**3
        self.F = 96487.0                # C/mol
        self.rho = 1626.0               # kg/m**3
        self.nu = 0.199*0.08499*0.002   # m**3
        self.Cp = 750.0                 # J/(K*kg)
        self.Tref = 298.15              # K
        self.R = 8.3143                 # J/(mol*K)
        self.hA = 0.085                 # J/(s*K)
        self.Ls = 0.000025              # m
        self.C = 1.656                  # OJO: A*h
        self.rate = 1.0                 # OJO: h
        self.I = -self.C/self.rate      # A
        
        """
        Constants
        """
                
        self.T0 = T0 # Temperature initial condition
        self.N = N # number of spatial cells
        self.ntsteps = ntsteps # number of time steps
        self.tf = tf # final time
        self.tau = self.tf/self.ntsteps # time mesh size
        self.tau2 = self.tau*0.5
        self.h = 1.0/(self.N) # spatial mesh size (both anode and cathode)
        
        """
        Parameters Rcell (These values depend on T0)
        """
        self.tetha1=0.0159
        self.tetha2=0.0137
        
        """
        Auxiliary arrays
        """

        self.dp = np.zeros(self.N+1)
        self.dn = np.zeros(self.N+1)
        self.index = np.arange(self.N+1)
        self.r = np.linspace(0.0, 1.0, self.N+1) # spatial mesh (both anode and cathode)
        self.ti = np.linspace(0.0, self.tf, self.ntsteps+1) # time mesh
        self.rr, self.tt = np.meshgrid(self.r, self.ti)
        
        """
        Sparse stiffness Matrix 
        """  
        
        self.diag0Sj = -(3.0*self.index[1:]**2 - 3.0*self.index[1:] + 1.0)/3.0
        self.diag1Sj = 2.0*(3.0*self.index**2 + 1.0)/3.0
        self.diag2Sj = -(3.0*self.index[:-1]**2 + 3.0*self.index[:-1] + 1.0)/3.0
        self.diag1Sj[0] = 1.0/3.0
        self.diag1Sj[-1] = (3.0*self.N**2 - 3.0*self.N + 1.0)/3.0        
    
        """
        Sparse mass matrix (without h**2 factor)
        """
        
        self.diag0Mj = self.h**2*(10.0*self.index[1:]**2 - 10.0*self.index[1:] + 3.0)/60.0
        self.diag1Mj = self.h**2*(10.0*self.index**2 + 1.0)/15.0
        self.diag2Mj = self.h**2*(10.0*self.index[:-1]**2 + 10.0*self.index[:-1] + 3.0)/60.0               
        self.diag1Mj[0] = self.h**2/30.0
        self.diag1Mj[-1] = self.h**2*(10.0*self.N**2 - 5.0*self.N + 1.0)/30.0

        
        """
        Solution storage arrays
        """
        
        self.solution_SoCp = np.zeros((self.ntsteps+1,self.N+1))
        self.solution_SoCn = np.zeros((self.ntsteps+1,self.N+1))
        self.solution_temperature = np.zeros((self.ntsteps+1))
        
    
    def DpFun(self,T):    
        """
        Diffusion coefficient at current temperature (cathode)
        """
        return self.Dspref*np.exp((self.Eadip/self.R)*(1.0/T - 1.0/self.Tref))/(self.Rp**2)
    
    
    def DnFun(self,T):  
        """
        Diffusion coefficient at current temperature (anode)
        """
        return self.Dsnref*np.exp((self.Eadin/self.R)*(1.0/T - 1.0/self.Tref))/(self.Rn**2)
    
    
    def kpFun(self,T):      
        """
        Reaction rate at current temperature (cathode)
        """
        return self.kpref*np.exp((self.Earep/self.R)*(1.0/T - 1.0/self.Tref))
    
    
    def knFun(self,T):        
        """
        Reaction rate at current temperature (anode)
        """
        return self.knref*np.exp((self.Earen/self.R)*(1.0/T - 1.0/self.Tref))

    
    def Upref(self,uRp): # To compute Up
        """
        Open circuit potencial (OCP) at reference temperature Tref (cathode)
        """
        return (4.04596 + np.exp(-42.30027*uRp + 16.56714) - 0.04880*np.arctan(50.01833*uRp -26.48897) 
                - 0.05447*np.arctan(18.99678*uRp - 12.32362) - np.exp(78.24095*uRp - 78.68074))
  
      
    def Unref(self,uRn): # To compute Un
        """
        Open circuit potencial (OCP) at reference temperature Tref (anode)
        """
        return (0.13966 + 0.68920*np.exp(-49.20361*uRn) + 0.41903*np.exp(-254.40067*uRn) 
                - np.exp(49.97886*uRn - 43.37888) - 0.028221*np.arctan(22.52300*uRn - 3.65328) 
                - 0.01308*np.arctan(28.34801*uRn - 13.43960))
        
    
    def Up(self,uRp,T): # Just for campute the potencial V
        """
        Open circuit potencial (OCP) (cathode)
        """
        return self.Upref(uRp) + self.dUpdT(uRp)*(T-self.Tref)
        
        
    def Un(self,uRn,T): # Just for campute the potencial V    
        """
        Open circuit potencial (OCP) (anode)
        """
        return self.Unref(uRn) + self.dUndT(uRn)*(T-self.Tref)
        
    
    def dUpdT(self,uRp):      
        """
        Coefficient of T in the overpotential at current temperature (cathode)
        """
        return 10**(-3)*((-0.19952 + 0.92837*uRp - 1.36455*uRp**2 + 0.61154*uRp**3)/
                (1 - 5.66148*uRp + 11.47636*uRp**2 - 9.82431*uRp**3 + 3.04876*uRp**4))


    def dUndT(self,uRn):      
        """
        Coefficient of T in the overpotential at current temperature (anode)
        """
        return 10**(-3)*((0.00527 + 3.29927*uRn - 91.79326*uRn**2 + 1004.91101*uRn**3 - 5812.27813*uRn**4 
                + 19329.75490*uRn**5 - 37147.89470*uRn**6 + 38379.18127*uRn**7 - 16515.05308*uRn**8)/
                (1 - 48.09287*uRn + 1017.23480*uRn**2 - 10481.80419*uRn**3 + 59431.30001*uRn**4 
                 - 195881.64880*uRn**5 + 374577.31520*uRn**6 - 385821.16070*uRn**7 + 165705.85970*uRn**8))


    def Cethap(self,uRp,T):      
        """
        Coefficient of T in the overpotential at current temperature (cathode)
        """
        mp=self.I/(self.F*self.kpFun(T)*self.Sp*self.cspmax*self.ce**0.5*(1.0-uRp)**0.5*uRp**0.5)
        return 2.0*self.R*np.log(((mp**2+4.0)**0.5+mp)/2.0)/self.F

    
    def Cethan(self,uRn,T):        
        """
        Coefficient of T in the overpotential at current temperature (anode)
        """
        mn=self.I/(self.F*self.knFun(T)*self.Sn*self.csnmax*self.ce**0.5*(1.0-uRn)**0.5*uRn**0.5)
        return -2.0*self.R*np.log(((mn**2+4.0)**0.5+mn)/2.0)/self.F

        
    def initial_condition_SoCp(self,r):        
        """
        Initial condition or SoC concentration (cathode)
        """
        return 0.4952*np.ones(len(r))

    
    def initial_condition_SoCn(self,r):        
        """
        Initial condition or SoC concentration (anode)
        """
        return 0.7522*np.ones(len(r)) 
        
    
    def boundary_condition_SoCp(self,t):
        """
        SoC boundary condition at r = Rp (cathode)
        """
        return -self.I/(self.F*self.Sp*self.cspmax*self.Rp)*np.ones(len(t))

    
    def boundary_condition_SoCn(self,t):
        """
        SoC boundary condition at r = Rn (anode)
        """
        return self.I/(self.F*self.Sn*self.csnmax*self.Rn)*np.ones(len(t))

           
    def AT(self,uRp,uRn,T):        
        """
        Coefficient of T in the temperature equation
        """
        aux = self.dUpdT(uRp)-self.dUndT(uRn)
        aux += self.Cethap(uRp,T)-self.Cethan(uRn,T)
        aux += self.I*self.tetha2
        aux += -self.hA/self.I
        return aux*self.I/(self.rho*self.nu*self.Cp)       

    
    def BT(self): # In fact, it is a constant   
        """
        Coefficient of 1 in the temperature equation
        """
        return ((self.I/(self.rho*self.nu*self.Cp))*
                (self.I*(self.tetha1-self.tetha2*self.T0)+self.hA*self.T0/self.I))


    def solve(self,p):        
        """
        Solve coupled problems marching in time
        """
                
        self.Earep = p[0]
        self.Earen = p[1]      
        self.Eadip = p[2]
        self.Eadin = p[3]          
        
        
        #@jit(nopython=True)        
        @njit()
        def tridiag(a,b,c,d):
            """
            Tridiagonal matrix solver 
            """        
            nn = len(d) # number of equations
            for it in np.arange(1, nn):
                m = a[it-1]/b[it-1]
                b[it] = b[it] - m*c[it-1] 
                d[it] = d[it] - m*d[it-1]        	    
                x = b
            x[-1] = d[-1]/b[-1]
            for il in np.arange(nn-2, -1, -1):
                x[il] = (d[il]-c[il]*x[il+1])/b[il]
            return x
        
        #@jit(nopython=True)
        @njit()        
        def multridiag(a,b,c,y):
            """
            Tridiagonal matrix vector product
            """
            n = len(y)
            v=np.zeros(n)
            v[0]=b[0]*y[0]+c[0]*y[1]
            for i in np.arange(1,n-1):
                v[i]=a[i-1]*y[i-1]+b[i]*y[i]+c[i]*y[i+1]
            v[n-1]=a[n-2]*y[n-2]+b[n-1]*y[n-1]
            return v
        
        self.solution_SoCp[0,:] = self.initial_condition_SoCp(self.r)
        self.solution_SoCn[0,:] = self.initial_condition_SoCn(self.r)
        self.solution_temperature[0] = self.T0
        Dp=np.zeros((self.ntsteps+1))
        Dp[0]=self.DpFun(self.T0)*self.tau2
        Dn=np.zeros((self.ntsteps+1))
        Dn[0]=self.DnFun(self.T0)*self.tau2
        Hp=self.boundary_condition_SoCp(self.ti)
        Hn=self.boundary_condition_SoCn(self.ti)
        for k in np.arange(self.ntsteps):
            uRkp=self.solution_SoCp[k,-1]
            uRkn=self.solution_SoCn[k,-1]
            #print(k,uRkp,uRkn,self.solution_temperature[k])
            k1RK = self.AT(uRkp,uRkn,self.solution_temperature[k])*self.solution_temperature[k] + self.BT()
            self.solution_temperature[k+1] = self.solution_temperature[k] + self.tau*k1RK
            Dp[k+1]=self.DpFun(self.solution_temperature[k+1])*self.tau2
            Dn[k+1]=self.DnFun(self.solution_temperature[k+1])*self.tau2
            Ap0 = self.diag0Mj + Dp[k+1]*self.diag0Sj
            Ap1 = self.diag1Mj + Dp[k+1]*self.diag1Sj
            Ap2 = self.diag2Mj + Dp[k+1]*self.diag2Sj
            An0 = self.diag0Mj + Dn[k+1]*self.diag0Sj
            An1 = self.diag1Mj + Dn[k+1]*self.diag1Sj
            An2 = self.diag2Mj + Dn[k+1]*self.diag2Sj
            Bp0 = self.diag0Mj - Dp[k]*self.diag0Sj
            Bp1 = self.diag1Mj - Dp[k]*self.diag1Sj
            Bp2 = self.diag2Mj - Dp[k]*self.diag2Sj
            Bn0 = self.diag0Mj - Dn[k]*self.diag0Sj
            Bn1 = self.diag1Mj - Dn[k]*self.diag1Sj
            Bn2 = self.diag2Mj - Dn[k]*self.diag2Sj
            self.dp[-1] = (Hp[k]+Hp[k+1])*self.tau2/self.h
            self.dn[-1] = (Hn[k]+Hn[k+1])*self.tau2/self.h
            Cp = multridiag(Bp0,Bp1,Bp2,self.solution_SoCp[k,:]) + self.dp
            Cn = multridiag(Bn0,Bn1,Bn2,self.solution_SoCn[k,:]) + self.dn
            self.solution_SoCp[k+1,:] = tridiag(Ap0,Ap1,Ap2,Cp)
            self.solution_SoCn[k+1,:] = tridiag(An0,An1,An2,Cn)
            

    def solve_true(self):
        """
        Solve coupled problems marching in time
        """
        
        #@jit(nopython=True)
        @njit()        
        def tridiag(a,b,c,d):
            """
            Tridiagonal matrix solver 
            """        
            nn = len(d) # number of equations
            for it in np.arange(1, nn):
                m = a[it-1]/b[it-1]
                b[it] = b[it] - m*c[it-1] 
                d[it] = d[it] - m*d[it-1]        	    
                x = b
            x[-1] = d[-1]/b[-1]
            for il in np.arange(nn-2, -1, -1):
                x[il] = (d[il]-c[il]*x[il+1])/b[il]
            return x
        
        #@jit(nopython=True)
        @njit()        
        def multridiag(a,b,c,y):
            """
            Tridiagonal matrix vector product
            """
            n = len(y)
            v=np.zeros(n)
            v[0]=b[0]*y[0]+c[0]*y[1]
            for i in np.arange(1,n-1):
                v[i]=a[i-1]*y[i-1]+b[i]*y[i]+c[i]*y[i+1]
            v[n-1]=a[n-2]*y[n-2]+b[n-1]*y[n-1]
            return v
        
        self.solution_SoCp[0,:] = self.initial_condition_SoCp(self.r)
        self.solution_SoCn[0,:] = self.initial_condition_SoCn(self.r)
        self.solution_temperature[0] = self.T0
        Dp=np.zeros((self.ntsteps+1))
        Dp[0]=self.DpFun(self.T0)*self.tau2
        Dn=np.zeros((self.ntsteps+1))
        Dn[0]=self.DnFun(self.T0)*self.tau2
        Hp=self.boundary_condition_SoCp(self.ti)
        Hn=self.boundary_condition_SoCn(self.ti)
        for k in np.arange(self.ntsteps):
            uRkp=self.solution_SoCp[k,-1]
            uRkn=self.solution_SoCn[k,-1]
            #print(k,uRkp,uRkn,self.solution_temperature[k])
            k1RK = self.AT(uRkp,uRkn,self.solution_temperature[k])*self.solution_temperature[k] + self.BT()
            self.solution_temperature[k+1] = self.solution_temperature[k] + self.tau*k1RK
            Dp[k+1]=self.DpFun(self.solution_temperature[k+1])*self.tau2
            Dn[k+1]=self.DnFun(self.solution_temperature[k+1])*self.tau2
            Ap0 = self.diag0Mj + Dp[k+1]*self.diag0Sj
            Ap1 = self.diag1Mj + Dp[k+1]*self.diag1Sj
            Ap2 = self.diag2Mj + Dp[k+1]*self.diag2Sj
            An0 = self.diag0Mj + Dn[k+1]*self.diag0Sj
            An1 = self.diag1Mj + Dn[k+1]*self.diag1Sj
            An2 = self.diag2Mj + Dn[k+1]*self.diag2Sj
            Bp0 = self.diag0Mj - Dp[k]*self.diag0Sj
            Bp1 = self.diag1Mj - Dp[k]*self.diag1Sj
            Bp2 = self.diag2Mj - Dp[k]*self.diag2Sj
            Bn0 = self.diag0Mj - Dn[k]*self.diag0Sj
            Bn1 = self.diag1Mj - Dn[k]*self.diag1Sj
            Bn2 = self.diag2Mj - Dn[k]*self.diag2Sj
            self.dp[-1] = (Hp[k]+Hp[k+1])*self.tau2/self.h
            self.dn[-1] = (Hn[k]+Hn[k+1])*self.tau2/self.h
#            Cp = self.multridiag(Bp0,Bp1,Bp2,self.solution_SoCp[k,:]) + self.dp
#            Cn = self.multridiag(Bn0,Bn1,Bn2,self.solution_SoCn[k,:]) + self.dn
            Cp = multridiag(Bp0,Bp1,Bp2,self.solution_SoCp[k,:]) + self.dp
            Cn = multridiag(Bn0,Bn1,Bn2,self.solution_SoCn[k,:]) + self.dn
            self.solution_SoCp[k+1,:] = tridiag(Ap0,Ap1,Ap2,Cp)
            self.solution_SoCn[k+1,:] = tridiag(An0,An1,An2,Cn)
            

    def Vcell(self,uRp,uRn,T):
        """
        Cell potential
        """
        aux = self.Up(uRp,T)-self.Un(uRn,T)
        aux += (self.Cethap(uRp,T) - self.Cethan(uRn,T) )*T
        aux += self.I*(self.tetha1 + self.tetha2*(T-self.T0))
        return aux
    
    
if __name__=="__main__":
    theta = np.array([58000.0,20000.0,29000.0,35000.0])
     
    #convergence plot
    fm = forward_mapping(ntsteps=328000)    
    fm.solve(theta)
    V000 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    where_are_NaNs = np.isnan(V000)
    V000[where_are_NaNs] = 0.0
    index000 = np.nonzero(V000)[0]
    tim000 = fm.ti[index000][::32]
    V000 = V000[index000][::32]

    
    fm = forward_mapping(ntsteps=164000)
    fm.solve(theta)
    V00 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    where_are_NaNs = np.isnan(V00)
    V00[where_are_NaNs] = 0.0
    index00 = np.nonzero(V00)[0]
    tim00 = fm.ti[index00][::16]
    V00 = V00[index00][::16]
    
    
    fm = forward_mapping(ntsteps=82000)
    fm.solve(theta)
    V0 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    where_are_NaNs = np.isnan(V0)
    V0[where_are_NaNs] = 0.0
    index0 = np.nonzero(V0)[0]
    tim0 = fm.ti[index0][::8]
    V0 = V0[index0][::8]        
    
    fm = forward_mapping(ntsteps=41000)    
    fm.solve(theta)
    V1 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    where_are_NaNs = np.isnan(V1)
    V1[where_are_NaNs] = 0.0
    index1 = np.nonzero(V1)[0]
    tim1 = fm.ti[index1][::4]
    V1 = V1[index1][::4]    
    
    fm = forward_mapping(ntsteps=20500)
    fm.solve(theta)
    V2 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    where_are_NaNs = np.isnan(V2)
    V2[where_are_NaNs] = 0.0
    index2 = np.nonzero(V2)[0]
    tim2 = fm.ti[index2][::2]
    V2 = V2[index2][::2]

    fm = forward_mapping(ntsteps=10250)    
    fm.solve(theta)
    V3 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    where_are_NaNs = np.isnan(V3)
    V3[where_are_NaNs] = 0.0
    index3 = np.nonzero(V3)[0]
    tim3 = fm.ti[index3]
    V3 = V3[index3]    

        
    # plt.plot(tim1,V1,'k.',label='data1',alpha=0.25)
    # plt.plot(tim2,V2,'b.',label='data2',alpha=0.25)
    # plt.plot(tim3,V3,'r.',label='data3',alpha=0.25)    
    # plt.legend()
    # plt.show()
    
    
    err_L2 = np.zeros(5)
    err_H1 = np.zeros(5)    
    hs = np.array([(tim00[1]-tim00[0])/32.0,(tim00[1]-tim00[0])/16.0,(tim0[1]-tim0[0])/8.0,(tim0[1]-tim0[0])/4.0,(tim0[1]-tim0[0])/2.0])
    
    # err_L2[0] = np.sqrt(np.sum((V000-V00)**2/hs[0])**2)
    # err_L2[1] = np.sqrt(np.sum((V00-V0)**2/hs[0])**2)
    # err_L2[2] = np.sqrt(np.sum((V0-V1)**2/hs[0])**2)    
    # err_L2[3] = np.sqrt(np.sum((V1-V2)**2/hs[0])**2)
    # err_L2[4] = np.sqrt(np.sum((V2-V3)**2/hs[0])**2)

    err_L2[0] = np.max(abs(V000-V00))
    err_L2[1] = np.max(abs(V00-V0))
    err_L2[2] = np.max(abs(V0-V1))
    err_L2[3] = np.max(abs(V1-V2))
    err_L2[4] = np.max(abs(V2-V3))
    
    
    p0 = np.log2(err_L2[1]/err_L2[0])
    p1 = np.log2(err_L2[2]/err_L2[1])
    p2 = np.log2(err_L2[3]/err_L2[2])
    p3 = np.log2(err_L2[4]/err_L2[3])    
    
    
    # err_L2[0] = np.max(V000-V00)#/hs[0])**2)
    # err_L2[1] = np.max(V00-V0)#/hs[0])**2)
    # err_L2[2] = np.max(V0-V1)#/hs[0])**2)    
    # err_L2[3] = np.max(V1-V2)#/hs[0])**2)
    # err_L2[4] = np.max(V2-V3)#/hs[0])**2)
 
    
    err_H1[0] = np.sqrt(np.sum((V000-V00)**2)/hs[0]**2+np.sum(np.diff(V000-V00)/hs[0])**4)
    err_H1[1] = np.sqrt(np.sum((V00-V0)**2)/hs[0]**2+np.sum(np.diff(V00-V0)/hs[0])**4)
    err_H1[2] = np.sqrt(np.sum((V0-V1)**2)/hs[0]**2+np.sum(np.diff(V0-V1)/hs[0])**4)
    err_H1[3] = np.sqrt(np.sum((V1-V2)**2)/hs[0]**2+np.sum(np.diff(V1-V2)/hs[0])**4)
    err_H1[4] = np.sqrt(np.sum((V2-V3)**2)/hs[0]**2+np.sum(np.diff(V2-V3)/hs[0])**4)

    # fig, axis = plt.subplots(1,1)
    # axis.loglog(hs,err_L2,'b*')
    # axis.xaxis.set_major_locator(MaxNLocator(5))     
    # axis.yaxis.set_major_locator(MaxNLocator(5)) 
    # axis.set(xlabel='Mesh size '+r'$h$',ylabel=r'$L_2$'+' norm')
    # plt.savefig('convergence_L2.png')    

    # fig, axis = plt.subplots(1,1)    
    # axis.loglog(hs,err_H1,'r*')    
    # axis.xaxis.set_major_locator(MaxNLocator(5))
    # axis.yaxis.set_major_locator(MaxNLocator(5))
    # axis.set(xlabel='Mesh size '+r'$h$',ylabel=r'$H_1$'+' norm')
    # plt.savefig('convergence_H1.png')

    # fig, axis = plt.subplots(1,2,sharex=True,sharey=True)
    # axis[0].loglog(hs,err_L2,'bo-',label='L2 norm error')
    # #axis[0].loglog(hs,hs**2,'go-',label='h cuadrado')    
    # axis[0].xaxis.set_major_locator(MaxNLocator(5))     
    # axis[0].yaxis.set_major_locator(MaxNLocator(5)) 
    # #axis[0].set(xlabel='Mesh size '+r'$h$',ylabel='Error')
    # axis[0].set(xlabel='Mesh size '+r'$h$')
    # axis[0].legend()
    # axis[1].loglog(hs,err_H1,'ro-',label='H1 norm error')
    # #axis[1].loglog(hs,hs,'go-',label='h cuadrado')        
    # axis[1].xaxis.set_major_locator(MaxNLocator(5))     
    # axis[1].loglog(hs,hs**2,'go-',label='h cuadrado')    
    # #axis[1].set(xlabel='Mesh size '+r'$h$',ylabel='Error')
    # axis[1].set(xlabel='Mesh size '+r'$h$')
    # axis[1].legend(loc = 'best')
    # plt.savefig('convergence.png')    
    


    # fm = forward_mapping(ntsteps=4100)    
    # time_start = time.perf_counter()
    # fm.solve(theta)
    # V1 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V1)
    # V1[where_are_NaNs] = 0.0
    # index1 = np.nonzero(V1)[0]
    # tim1 = fm.ti[index1]
    # V1 = V1[index1][::50]
    # time_elapsed = (time.perf_counter() - time_start)
    # memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    # print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))
    
    # fm = forward_mapping(ntsteps=41000)    
    # time_start = time.perf_counter()
    # fm.solve(theta)
    # V1 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V1)
    # V1[where_are_NaNs] = 0.0
    # index1 = np.nonzero(V1)[0]
    # tim1 = fm.ti[index1]
    # V1 = V1[index1][::50]
    # time_elapsed = (time.perf_counter() - time_start)
    # memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    # print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))    

    
    # fm = forward_mapping(ntsteps=4100)
    # fm.solve(theta)
    # V1 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V1)
    # V1[where_are_NaNs] = 0.0
    # index1 = np.nonzero(V1)[0]
    # tim1 = fm.ti[index1]
    # V1 = V1[index1][::50]

    # fm = forward_mapping(ntsteps=8200)
    # fm.solve(theta)
    # V2 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V2)
    # V2[where_are_NaNs] = 0.0
    # index2 = np.nonzero(V2)[0]
    # tim2 = fm.ti[index2]
    # V2 = V2[index2][::100]

    # fm = forward_mapping(ntsteps=16400)
    # fm.solve(theta)
    # V3 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V3)
    # V3[where_are_NaNs] = 0.0
    # index3 = np.nonzero(V3)[0]
    # tim3 = fm.ti[index3]
    # V3 = V3[index3][::200]

    # fm = forward_mapping(ntsteps=41000)
    # fm.solve(theta)
    # V4 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V4)
    # V4[where_are_NaNs] = 0.0
    # index4 = np.nonzero(V4)[0]
    # tim4 = fm.ti[index4]
    # V4 = V4[index4][::500]

    
    # fm = forward_mapping(ntsteps=4100)
    # fm.solve(theta)
    # V1 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V1)
    # V1[where_are_NaNs] = 0.0
    # V1 = V1[0::50]
    

    # fm = forward_mapping(ntsteps=41000)
    # fm.solve(theta)
    # V2 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V2)
    # V2[where_are_NaNs] = 0.0
    # V2 = V2[0::500]
    # V2 = V2[:-1]

    # fm = forward_mapping(ntsteps=8200)
    # fm.solve(theta)
    # V3 = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.ntsteps+1)])    
    # where_are_NaNs = np.isnan(V3)
    # V3[where_are_NaNs] = 0.0
    # V3 = V3[0::100]
    # V3 = V3[:-1]
    
