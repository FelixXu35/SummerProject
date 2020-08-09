
def maser_coupledeqns(gs,kc,ks,N,gamma,time_span,num,delta):
    
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    
    """ This function is used to show the evolution of the light in a maser.
    It achieves by a set of 4 ODEs w.r.t. time. The main puropose of the 
    function is that by given the constant as inputs, we can reproduce 
    the graphs in the Breeze's paper.
    
    Input:
        gs:       spin-photon coupling constant
        kc        cavity decay rate
        ks        spin decoherence rate
        N         number of spins
        gamma     the spin-lattice relaxation rate 
        time_span the time iterval required 
        num       the number time moments that the evulation takes place
        delta     the frequency detuning parameter, omega_c-omega_s
    Output:
        figure
        
    """
    
    """ defining the variables required to solve 
    <a.dag()a>=n_bar
    <Sa+a>=s_up_ave_a
    <Sa-a.dag()>=np.conj(s_up_ave_a)
    <Saz>=s_z_ave
    <sa+sa->=s_ave_up_dn
    """
    #y= np.array([n_bar,s_up_ave_a,s_z_ave,s_ave_up_dn])
    
    def f(t,y):
        
        f0=-kc*y[0]+kc*4.3*(10**3)+(0+1j)*gs*np.sqrt(N)*(y[1]-np.conj(y[1]))
        
        f1=-(kc/2 + gamma/2 + ks/2 + (0+1j)*delta)*y[1]-(0+1j)*gs*np.sqrt(N)*(((y[2]+1)/2)+(1-1/N)*y[3]+y[0]*y[2])
        
        f2=-gamma*y[2]-2*(0+1j)*gs*(1/np.sqrt(N))*(y[1]-np.conj(y[1]))
        
        f3=-(gamma+ks)*y[3]+(0+1j)*gs*np.sqrt(N)*y[2]*(y[1]-np.conj(y[1]))
        
        return np.array([f0,f1,f2,f3])
    
    # apply the initial conditions at time=0
    y0=[4.3*(10**3)+0j,0.+0j,0.8+0j,0.+0j]
    
    # now solve the differential eqns
    sol=solve_ivp(f,time_span,y0,method='RK45',t_eval=np.linspace(time_span[0]+0j,time_span[1]+0j,num))
    
    # print(sol.y[0,:])
    
    
    # plot the result
    #fig, axes = plt.subplot(4,1,sharex=True)
    

    return sol.y
 

import numpy as np


gs=0.042*(2.*np.pi)
kc=0.18*(10**6)*(2.*np.pi)
ks=0.11*(10**6)*(2.*np.pi)
N=7.*(10.**14.)
gamma=0
time_span=[0,10*10**(-6)]
num=1000000
delta=0

maser_coupledeqns(gs,kc,ks,N,gamma,time_span,num,delta)


        
        
    
    
    