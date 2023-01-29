import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 12})
from numba import jit
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2021)

class forward_mapping:
    
    """
    This class simulates the li-ion battery forward problem
    """
    
    def __init__(self, T0 = 298.15, N=100, ntsteps=4100, tf=4100.0, resample=0, alpha = 5*10**12,nend=500):

        self.alpha =alpha
        self.resample = resample
        self.nend = nend
        
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
        
        self.Nmeas=self.ntsteps
        self.solfit_SoCp=np.zeros((self.Nmeas+1,self.N+1))
        self.solfit_SoCn=np.zeros((self.Nmeas+1,self.N+1))

        
    
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


    def solve(self):        
        """
        Solve coupled problems marching in time
        """
        
        @jit(nopython=True)
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
        
        @jit(nopython=True)
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
            

    def solveUnCoupled(self, Tfit):        
        """
        Solve coupled problems marching in time
        """
        
        @jit(nopython=True)
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
        
        @jit(nopython=True)
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
        
        self.solfit_SoCp[0,:] = self.initial_condition_SoCp(self.r)
        self.solfit_SoCn[0,:] = self.initial_condition_SoCn(self.r)
        Dp=np.zeros((self.Nmeas+1))
        Dp[0]=self.DpFun(self.T0)*self.tau2
        Dn=np.zeros((self.Nmeas+1))
        Dn[0]=self.DnFun(self.T0)*self.tau2
        Hp=self.boundary_condition_SoCp(self.ti)
        Hn=self.boundary_condition_SoCn(self.ti)
        for k in np.arange(self.Nmeas):
            Dp[k+1]=self.DpFun(Tfit[k+1])*self.tau2
            Dn[k+1]=self.DnFun(Tfit[k+1])*self.tau2
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
            Cp = multridiag(Bp0,Bp1,Bp2,self.solfit_SoCp[k,:]) + self.dp
            Cn = multridiag(Bn0,Bn1,Bn2,self.solfit_SoCn[k,:]) + self.dn
            self.solfit_SoCp[k+1,:] = tridiag(Ap0,Ap1,Ap2,Cp)
            self.solfit_SoCn[k+1,:] = tridiag(An0,An1,An2,Cn)
            
    def RevHeat(self,uRp,uRn,T,DerT):
        """
        Reversible heat
        """        
        aux=(self.rho*self.nu*self.Cp)*DerT/(self.I*T)
        aux += -(self.Cethap(uRp,T)-self.Cethan(uRn,T))
        aux += -self.I*self.tetha2
        aux += self.hA/self.I
        aux += -(self.I*(self.tetha1-self.tetha2*self.T0)+self.hA*self.T0/self.I)/T
        return aux       


    def Vcell(self,uRp,uRn,T):
        """
        Cell potential
        """
        aux = self.Up(uRp,T)-self.Un(uRn,T)
        aux += (self.Cethap(uRp,T) - self.Cethan(uRn,T) )*T
        aux += self.I*(self.tetha1 + self.tetha2*(T-self.T0))
        return aux
    
    def volterratikhonov(self,TrueT,T,sigma):
        """
        Approximate derivative (Tikhonov applied to Volterra operator)
        """
        T0=TrueT[0]
        Tn = T-TrueT[0] # rescale data to start at T0 = 0
        Tn = Tn[1:]
        # make volterra operator via midpoint rule
        L = np.zeros((self.Nmeas,self.Nmeas))
        i,j = np.indices(L.shape)
        L[i>=j]=1.0
        L*=self.tau  

        A = np.eye(self.nend)    
        m = (self.Nmeas+1-2*self.nend)//2
        a = np.zeros(m)
        a[0]=1.0
        b = np.zeros(m)
        b[0]=0.5
        b[1]=0.5
    
        R = np.zeros((self.Nmeas-2*self.nend,m))    
        for i in np.arange(m):
            R[2*i,:] = np.roll(a,i)
            R[2*i-1,:] = np.roll(b,i-1)
        
        C = np.block([[A,np.zeros((self.nend,m)),np.zeros(A.shape)],
                      [np.zeros((self.Nmeas-2*self.nend,self.nend)),R,np.zeros((self.Nmeas-2*self.nend,self.nend))],
                      [np.zeros(A.shape),np.zeros((self.nend,m)),A]])  
        
        if self.resample == 1:

            L = np.dot(L,C)
                                    
        if self.resample == 0:
            n = self.Nmeas
        else:
            nd = np.loadtxt('naive_dimension.txt')
            n = np.int(2*self.nend+(nd-2*self.nend)//2)
                                              
        Ld = (2.0*np.eye(n)-1.0*np.eye(n,k=1)-1.0*np.eye(n,k=-1))
        M = np.dot(Ld.T,Ld)
        Dev = np.diag(sp.linalg.inv(M))
        delta = 1.0/np.sqrt(Dev[np.int(n/2)])
        Lr = np.copy(Ld)
        Lr[-1,-1] = 2.0*delta
        Lr[0,0] = 2.0*delta
        Lr[0,1] = 0.0
        Lr[-1,-2] = 0.0
        gmrf = 0.5*np.dot(Lr.T,Lr)
                                       
        
        # solve
        H = sigma**-2*np.dot(L.T,L)        
        prec_post = H + self.alpha*gmrf
        cov_post = sp.linalg.inv(prec_post)
        std_pr = np.sqrt(np.diag(sp.linalg.inv(self.alpha*gmrf)))        
        std_post = np.sqrt(np.diag(cov_post))
        plt.semilogy(std_pr,'b-',label = 'pointswise prior standard deviation')
        plt.semilogy(std_post,'r-',label = 'pointswise posterior standard deviation')
        plt.xlabel('Time (S)')
        plt.ylabel(r'$\sigma$')        
        plt.legend()
        if self.resample==0:
            plt.savefig('pointwise_standard_deviation'+np.str(fm.ntsteps))          
        else:
            plt.savefig('pointwise_standard_deviation_res'+np.str(fm.ntsteps))          
            
        DerTrue = np.linalg.solve(H,sigma**-2*np.dot(L.T,TrueT[1:]-TrueT[0]))
        DerTrue = np.insert(DerTrue,0,DerTrue[0])
        DerT = np.linalg.solve(prec_post,sigma**-2*np.dot(L.T,Tn))
        unc = 2*np.sqrt(np.diag(cov_post))
        q1 = DerT-unc
        q2 = DerT+unc      
        Tfit=np.dot(L,DerT)+TrueT[0]
        Tfit=np.insert(Tfit,0,TrueT[0])
        Tq1=np.dot(L,q1)+T0
        Tq1=np.insert(Tq1,0,T0)
        Tq2=np.dot(L,q2)+T0
        Tq2=np.insert(Tq2,0,T0) 
        q1 = np.insert(q1,0,q1[0])
        q2 = np.insert(q2,0,q2[0])
        DerT2=(DerT[:-1]+DerT[1:])/2
        DerT2=np.insert(DerT2,0,DerT[0])
        DerT2=np.insert(DerT2,-1,DerT[-1])        
        DerT = DerT2
                  
        return DerTrue, DerT, Tfit, q1, q2, Tq1, Tq2, H, gmrf, cov_post, C
    
if __name__=="__main__":
    ntsteps = 16400
    resample = 1
    fm = forward_mapping(ntsteps=ntsteps,resample=resample,alpha=16*ntsteps**3,nend=ntsteps//10)
    """
    ntsteps  - number of degrees of freedom
    resample - boolean to resample
    alpha    - regularization parameter
    nend     - number of end points without resampling    
    """
    fm.solve()
    
    where_are_NaNs = np.isnan(fm.solution_temperature)
    itend=np.where(where_are_NaNs == True)[0][0]
    TempTot=fm.solution_temperature[:itend]
    timTot=fm.ti[:itend]
    tim = timTot[0::1]   # One measurement per second
    Temp = TempTot[0::1] 
    fm.Nmeas=len(Temp)-1
    std = Temp.max()/(1.0*10.0**2)
    Tmeas = Temp + std*np.random.randn(fm.Nmeas+1) # Mediciones
    
    DerTrue, DerT, Tfit, q1, q2, Tq1, Tq2, H, gmrf, cov_post, C= fm.volterratikhonov(Temp,Tmeas,std)

    
    if fm.resample == 0:
        np.savetxt('naive_dimension.txt',[len(TempTot)])
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.plot(tim,DerTrue,'k',label='True temperature derivative')
        ax0.plot(tim,DerT,'r',label='Temperature derivative mean')    
        ax0.fill_between(tim,q1,q2,color='b',label='5-95 percentile', alpha=0.25) 
        ax0.legend(loc='upper left')
        plt.xlabel('Time (S)')
        plt.ylabel(r'$\frac{dT}{dt}$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))        
        plt.savefig('derivative'+np.str(fm.ntsteps))
        
        fm.solveUnCoupled(Tfit)
        
        revheat=fm.RevHeat(fm.solfit_SoCp[:(fm.Nmeas+1),-1],fm.solfit_SoCn[:(fm.Nmeas+1),-1],Tfit,DerT)
        revheat *= fm.I*Tfit
        revheat_q1=fm.RevHeat(fm.solfit_SoCp[:(fm.Nmeas+1),-1],fm.solfit_SoCn[:(fm.Nmeas+1),-1],Tq1,q1)
        revheat_q1 *= fm.I*Tfit
        revheat_q2=fm.RevHeat(fm.solfit_SoCp[:(fm.Nmeas+1),-1],fm.solfit_SoCn[:(fm.Nmeas+1),-1],Tq2,q2)    
        revheat_q2 *= fm.I*Tfit
        revheat2=fm.dUpdT(fm.solfit_SoCp[:(fm.Nmeas+1),-1]) - fm.dUndT(fm.solfit_SoCn[:(fm.Nmeas+1),-1])
        revheat2 *= fm.I*Temp
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(tim,revheat,'r-',label='Reversible heat mean')
        ax1.fill_between(tim,revheat_q1,revheat_q2,color='b',label='5-95 percentile', alpha=0.25)
        ax1.plot(tim,revheat2,'k',label='True reversible heat')
        plt.xlabel('Time (S)')
        plt.ylabel(r'$q_{rev}$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
        ax1.legend(loc='upper left')
        plt.savefig('reversible_heat'+np.str(fm.ntsteps))
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(tim,Tmeas,'k',label='Observed temperature',lw=0.5,alpha=0.25)
        ax2.plot(tim,Tfit,'r-',label='Temperature Mean',lw=0.5)
        ax2.fill_between(tim,Tq1,Tq2,color='b',label='5-95 percentile', alpha=0.25)    
        ax2.plot(tim,Temp,'k',label='True temperature')    
        plt.xlabel('Time (S)')
        plt.ylabel(r'$T$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
        ax2.legend(loc='upper left')
        plt.savefig('temperature'+np.str(fm.ntsteps))
        
        
        V = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.Nmeas+1)])
        Vfit = np.array([fm.Vcell(fm.solfit_SoCp[k,-1],fm.solfit_SoCn[k,-1],Tfit[k]) for k in np.arange(fm.Nmeas+1)])
        Vq1 = np.array([fm.Vcell(fm.solfit_SoCp[k,-1],fm.solfit_SoCn[k,-1],Tq1[k]) for k in np.arange(fm.Nmeas+1)])
        Vq2 = np.array([fm.Vcell(fm.solfit_SoCp[k,-1],fm.solfit_SoCn[k,-1],Tq2[k]) for k in np.arange(fm.Nmeas+1)])    
        
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(tim,Vfit,'r-',label='Predicted voltage')  
        ax3.fill_between(tim,Vq1,Vq2,color='b',label='5-95 percentile', alpha=0.25)    
        ax3.plot(tim,V,'k',label='True voltage')
        ax3.legend(loc='lower left')
        plt.xlabel('Time (S)')
        plt.ylabel(r'$V_{cell}$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
        plt.savefig('voltage'+np.str(fm.ntsteps))
    
    else:
               
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.plot(tim[1:],np.dot(C,DerTrue[1:]),'k',label='True temperature derivative')
        ax0.plot(tim[1:],np.dot(C,DerT[1:]),'r',label='Mean temperature derivative')    
        ax0.fill_between(tim[1:],np.dot(C,q1[1:]),np.dot(C,q2[1:]),color='b',label='5-95 percentile', alpha=0.25) 
        ax0.legend(loc='upper left')
        plt.xlabel('Time (S)')
        plt.ylabel(r'$\frac{dT}{dt}$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))         
        plt.savefig('derivative_res'+np.str(fm.ntsteps))        
        
        fm.solveUnCoupled(Tfit)

        CDerT2 = np.dot(C,DerT[1:])
        CDerT2 = np.insert(CDerT2,0,CDerT2[0])
        revheat=fm.RevHeat(fm.solfit_SoCp[:(fm.Nmeas+1),-1],fm.solfit_SoCn[:(fm.Nmeas+1),-1],Tfit,CDerT2)
        revheat *= fm.I*Tfit
        Cq1 = np.dot(C,q1[1:])
        Cq1 = np.insert(Cq1,0,Cq1[0])
        revheat_q1=fm.RevHeat(fm.solfit_SoCp[:(fm.Nmeas+1),-1],fm.solfit_SoCn[:(fm.Nmeas+1),-1],Tq1,Cq1)
        revheat_q1 *= fm.I*Tfit
        Cq2 = np.dot(C,q2[1:])
        Cq2 = np.insert(Cq2,0,Cq2[0])        
        revheat_q2=fm.RevHeat(fm.solfit_SoCp[:(fm.Nmeas+1),-1],fm.solfit_SoCn[:(fm.Nmeas+1),-1],Tq2,Cq2)    
        revheat_q2 *= fm.I*Tfit
        revheat2=fm.dUpdT(fm.solfit_SoCp[:(fm.Nmeas+1),-1]) - fm.dUndT(fm.solfit_SoCn[:(fm.Nmeas+1),-1])
        revheat2 *= fm.I*Temp
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(tim,revheat,'r-',label='Reversible heat median')
        ax1.fill_between(tim,revheat_q1,revheat_q2,color='b',label='5-95 percentile', alpha=0.25)
        ax1.plot(tim,revheat2,'k',label='True reversible heat')
        plt.xlabel('Time (S)')
        plt.ylabel(r'$q_{rev}$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
        ax1.legend(loc='upper left')
        plt.savefig('reversible_heat_res'+np.str(fm.ntsteps))
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(tim,Tmeas,'k',label='Observed temperature',lw=0.5,alpha=0.25)
        ax2.plot(tim,Tfit,'r-',label='Temperature Median',lw=0.5)
        ax2.fill_between(tim,Tq1,Tq2,color='b',label='5-95 percentile', alpha=0.25)    
        ax2.plot(tim,Temp,'k',label='True temperature')    
        plt.xlabel('Time (S)')
        plt.ylabel(r'$T$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
        ax2.legend(loc='upper left')
        plt.savefig('temperature_res'+np.str(fm.ntsteps))
        
        
        V = np.array([fm.Vcell(fm.solution_SoCp[k,-1],fm.solution_SoCn[k,-1],fm.solution_temperature[k]) for k in np.arange(fm.Nmeas+1)])
        Vfit = np.array([fm.Vcell(fm.solfit_SoCp[k,-1],fm.solfit_SoCn[k,-1],Tfit[k]) for k in np.arange(fm.Nmeas+1)])
        Vq1 = np.array([fm.Vcell(fm.solfit_SoCp[k,-1],fm.solfit_SoCn[k,-1],Tq1[k]) for k in np.arange(fm.Nmeas+1)])
        Vq2 = np.array([fm.Vcell(fm.solfit_SoCp[k,-1],fm.solfit_SoCn[k,-1],Tq2[k]) for k in np.arange(fm.Nmeas+1)])    
        
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(tim,Vfit,'r-',label='Voltage median')  
        ax3.fill_between(tim,Vq1,Vq2,color='b',label='5-95 percentile', alpha=0.25)    
        ax3.plot(tim,V,'k',label='True voltage')
        ax3.legend(loc='lower left')
        plt.xlabel('Time (S)')
        plt.ylabel(r'$V_{cell}$')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
        plt.savefig('voltage_res'+np.str(fm.ntsteps))
