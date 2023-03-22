import GPy
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('code')
import GP_varsel as varsel

from scipy.linalg import cholesky,cho_solve
from GPy.util import linalg

#Kullback-Leibler Divergence between Gaussian distributions
def KLD_Gaussian(m1,V1,m2,V2,use_diag=False):
    Dim = m1.shape[0]
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        L1 = np.diag(np.sqrt(np.diag(V1)))
        L2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0/np.diag(V2))
    else:
        L1 = cholesky(V1, lower=True)
        L2 = cholesky(V2, lower=True)
        V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
        #V2_inv = np.linalg.inv(V2)
    #print(V2_inv)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1-m2).T, np.dot(V2_inv, (m1-m2))) \
         - 0.5 * Dim + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L2)))) - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L1))))
    return np.abs(KL)  #This is to avoid any negative due to numerical instability

def KLD_Gaussian_NoChol(m1,V1,m2,V2,use_diag=False):
    Dim = m1.shape[0]
    #print("shape m1", m1.shape)
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        V1 = np.diag(np.sqrt(np.diag(V1)))
        V2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0/np.diag(V2))
    else:
        #V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
        V2_inv = np.linalg.inv(V2)
    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1 - m2).T, np.dot(V2_inv, (m1 - m2))) \
         - 0.5 * Dim + 0.5 * np.log(np.linalg.det(V2)) - 0.5 * np.log(np.linalg.det(V1))
    return KL  #This is to avoid any negative due to numerical instability

def KLRel_Juan(train_x,model,delta):
    N,P = train_x.shape   #Do we need to merge the train with val in here?
    relevance = np.zeros((N,P))
    #delta = 1.0e-4
    jitter = 1.0e-15
    #which_p = int(config.feature)
    print(f"Analysing {P} Features...")
    for p in range(P):
        for n in range(N):
            #x_plus = X[n,:].copy()
            x_plus = train_x[n:n+1, :].copy()
            x_minus = train_x[n:n + 1, :].copy()
            x_plus[0,p] = x_plus[0,p]+delta
            x_minus[0, p] = x_minus[0, p] - delta

            # Make predictions
            m1, V1 = GPy.models.GPRegression.predict(model, train_x[n:n+1,:], full_cov=False)
            m2, V2 = GPy.models.GPRegression.predict(model, x_plus, full_cov=False)
            m2_minus, V2_minus = GPy.models.GPRegression.predict(model, x_minus, full_cov=False)
            use_diag = False
            KL_plus = np.sqrt(KLD_Gaussian(m1,V1,m2,V2,use_diag=use_diag)+jitter) #In code the authors don't use the Mult. by 2
            KL_minus = np.sqrt(KLD_Gaussian(m1, V1, m2_minus,V2_minus,use_diag=use_diag)+jitter)  # In code the authors don't use the Mult. by 2
            relevance[n, p] = 0.5*(KL_plus+KL_minus)/delta
    return np.mean(relevance,0)

np.random.seed(1)

# number of repetitions to average over
repeats = 2
# number of covariates
m = 8
# number of data points
n = 300
# Delta for KL method
delta = 0.0001
# number of quadrature points for VAR method
nquadr = 11

# if x are uniformly distributed
# compute the analytical scaling coefficients for the m components
phi = np.pi*np.linspace(0.1,1,m);
Aunif = np.zeros(m)
for i in range(0,m):
    Aunif[i] = np.sqrt( 4/(2 -1/phi[i]*np.sin(2*phi[i])) )

# if x are normally distributed
# compute the scaling coefficients for the m components
nquadr2 = 11
[points,weights] = np.polynomial.hermite.hermgauss(nquadr2)
std = 0.4
Anor = np.zeros(m)
for i in range(0, m):
    coeff = np.sin(phi[i]*np.sqrt(2)*std*points)*np.sin(phi[i]*np.sqrt(2)*std*points)
    var = 1.0/np.sqrt(np.pi)*np.dot(coeff.T,weights)
    Anor[i] = 1/np.sqrt(var)

rel_ARD = np.zeros(m)
rel_KL = np.zeros(m)
rel_KL_juan = np.zeros(m)
rel_VAR = np.zeros(m)
for i in range(0, repeats):
    x = np.random.uniform(-1.,1.,(n,m))
    #x = np.random.normal(0.0,std,(n,m))
    print('starting repetition ', i + 1,'/',repeats)
    phi = np.tile(np.linspace(np.pi/10,np.pi,m),(n,1))
    xphi = np.multiply(x[:,0:m],phi)
    f = np.sin(xphi)
    #Aunif[2] = Aunif[2]*1.3
    #Anor[0] = Anor[0] + 0.5
    for j in range(0,m):
        f[:,j] = f[:,j]*Aunif[j]#*(1.7-j*0.1)
        #f[:,j] = f[:,j]*Anor[j]

    #f = f[:, ::-1].copy()
    #f = f/f.std(0)
    # y is a sum of the m components plus Gaussian noise
    yval = f.sum(axis=1) + np.random.normal(0,0.3,(n,))
    y = np.asmatrix(yval).T

    # RBF kernel plus constant term
    kernel = GPy.kern.RBF(input_dim=m,ARD=True) + GPy.kern.Bias(input_dim=m)
    model = GPy.models.GPRegression(x,y,kernel)
    model.optimize()
    
    # ARD relevance value is the inverse of the length scale
    rel_ARD = rel_ARD + 1/model.sum.rbf.lengthscale
    
    # KL relevance value
    rel_KL = rel_KL + varsel.KLrel(x,model,delta)

    rel_KL_juan = rel_KL_juan + KLRel_Juan(x,model,delta)
    
    # VAR relevance value
    rel_VAR = rel_VAR + varsel.VARrel(x,model,nquadr)
    
 
# normalize the relevance values      
rel_ARD_nor = rel_ARD/np.max(rel_ARD)
rel_KL_juan_nor = rel_KL_juan/np.max(rel_KL_juan)
rel_KL_nor = rel_KL/np.max(rel_KL)
rel_VAR_nor = rel_VAR/np.max(rel_VAR)

# True relevance, the covariates are equally relevant in the L2 sense
rel_true = np.ones(m)

# plot
covariates = np.arange(1,m+1) 
fig2 = plt.figure()
ax2 = plt.subplot(111)
ax2.plot(covariates,rel_ARD_nor,'-ob',label='ARD')
ax2.plot(covariates,rel_KL_nor,'-or',label='KL')
ax2.plot(covariates,rel_KL_juan_nor,'--og',label='KL_Juan')
ax2.plot(covariates,rel_VAR_nor,'-oc',label='VAR')
ax2.plot(covariates,rel_true,'--ok',label='True')
ax2.legend()
ax2.set_ylabel('relevance')
ax2.set_xlabel('input')
plt.show()









