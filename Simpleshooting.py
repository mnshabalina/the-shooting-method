

# SOLVE THE GIVEN BOUNDARY VALUE PROBLEM USING SHOOTING METHOD



import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
#matplotlib inline



###########################################################
# PROBLEM DEFINITION
###########################################################

# Boundary conditions: y(t1) = alpha. y(t2) =beta
# Boundary conditions: y(0) = 4. y(1) =1
# Global variables:
a   = 0.0 #t1
b   = 1.0 #t2
alpha = 4.0 # alpha
beta = 1.0  # beta
N = 500000 # number of steps in integration

#%% Function y'' = F(t,y1,y2) is defined in this section
def F(t,y1,y2):
    return (3.0/2.0)*y1**2

#%% This function carries out Euler integration for a given initial value of s and number of iterations
def integrate_euler(s,Niter):
    h = (b-a)/Niter
    y1 = np.zeros((Niter+1,1))
    y2 = np.zeros((Niter+1,1))
    
    y1[0] = alpha
    y2[0] = s
    
    # Marching in t using forward euler
    for i in range(0,Niter,1):  # marching N steps in time with the guess y2_0
        ti  = a + i*h
        y1[i+1] = y1[i] + h*y2[i]
        y2[i+1] = y2[i] + h*F(ti,y1[i],y2[i])
    # difference in final value of y1 and boundary condition specified
    Fs = y1[Niter]-beta
    return Fs,y1

    
    
#%% Newton Raphson method for solving the IVP for an initial guess value sgN

def NewtonRaphson(sgN):
    Err1    = 1          # initializing error
    perturb = 0.0001     # delta s for calculating deltaFs numerically   
    ErrTol  = 1e-13      # Error tolerance
    s       = sgN        # Initial value for s
    k       = 0
    kmax    = 20         # Max number of iterations
    while Err1>ErrTol and k<kmax:
        Fs,y1 = integrate_euler(s,N)
        Fsd,y1d = integrate_euler(s+perturb,N) 
        DeltaFs = (Fsd-Fs)/perturb
        snew    = s - Fs/DeltaFs
        Err1    = abs(snew-s)    
        s       = snew
        print('Error in iteration:',k,'=',Err1)
        k = k+1
    if(Err1<=ErrTol):
        print('Error tolerance is achieved in Newton method')
        print('For the initial guess:',sgN,'Value of s using Newton method is:',s)
    else:
        print('Maximum number of iterations reached without achieving convergence')
    return s    
    
        
#%% Bisection method for solving the IVP for an initial guess range a0,b0
def Bisection(aig,big):
    Err1    = 1             # initializing error
    ErrTol  = 1e-13         # Error tolerance 
    ai      = aig           # Initial guess range - start point
    bi      = big           # Initial guess range - end point
    xi      = (aig+big)/2
    k       = 0
    kmax    = 60
    while Err1>ErrTol and k < kmax:
        Fsl,yl = integrate_euler(ai, N)
        Fsr,yr = integrate_euler(bi, N)
        Fsm,ym = integrate_euler(xi, N)
        # following loop to check if initial guess range for s is proper
        if k ==0:
            FsaiFsbi = (Fsl)*(Fsr)
            if FsaiFsbi<0:
                print('Initial guess of ai and bi is such that Fs(ai)*Fs(bi)<0, bi-section method will proceed')
            else:
                print(('Initial guess of ai and bi is such that Fs(ai)*Fs(bi)>=0,\n revise the initial guess'))
                break
    
        # now, compare the value of Fs at middle point and proceed with bisection
        
        if Fsl*Fsm<=0:
            ai= ai
            bi= (ai+bi)/2
        else:
            ai=(ai+bi)/2
            bi=bi
        
        xi = (ai+bi)/2
        
            
            
        Err1 = abs((ai-bi)/2)       
        print('Error in iteration:',k,'=',Err1)
        k = k+1

    print('For the initial guess range (',aig,',',big,')','value of s is:',xi)
    return xi

#%% Generate a plot of F(s) vs s to obtain a good guess value
sguess = np.linspace(-100,0,201);
Fsguess = np.zeros((np.size(sguess),1))
Nguess = 1000 # calculating with lower number of points for finding a guess for s

for k in range(0,np.size(sguess),1):
    Fsguess[k],y1temp = integrate_euler(sguess[k],Nguess)

## plotting the results
plt.plot(sguess,Fsguess)
plt.xlabel('s')
plt.ylabel('F(s)')
plt.grid(1)


## Check the values of Fsguess and prompt the approximate location of zeros for a better guess

zero_crossing = np.where(np.diff(np.sign(Fsguess),axis = 0))[0]
if len(zero_crossing)==0:
    print('Function Fs does not intersect with y = 0 in given range of guesses of s')
else:
    print('Function Fs intersects with y = 0 line approximately at:',sguess[zero_crossing])
                
#%% Main section: carry out calculations using Newton method and bisection method 

# 1. Newton Method
# Guess values are selected based on the output from previous section showing Fs vs s
# First guess for s for Newton method
guess1 =  (beta-alpha)/(b-a)
print('#############################################################')
print('Newton method solution for guess value of s = ',guess1)
sn1 = NewtonRaphson(guess1)  # carry out newton raphson iterations for given guess
# Second guess for s for Newton method
guess2 =  -30
print('#############################################################')
print('Newton method solution for guess value of s = ',guess2)
sn2 = NewtonRaphson(guess2)  # carry out newton raphson iterations for given guess

# 1. Bisection
# Guess values are selected based on the output from previous section showing Fs vs s
# First guess for (a0,b0) for bisection method

aiguess1 = -8-5
biguess1 = -8+5
print('#############################################################')
print('Bisection method solution for guess value of range of s = (',aiguess1,',',biguess1,')')
sb1 = Bisection(aiguess1, biguess1) # carry out newton raphson iterations for given guess
# Second guess for (a0,b0) for bisection method
aiguess2 = -36-5
biguess2 = -36+5
print('#############################################################')
print('Bisection method solution for guess value of range of s = (',aiguess2,',',biguess2,')')
sb2 = Bisection(aiguess2, biguess2) # carry out newton raphson iterations for given guess

    