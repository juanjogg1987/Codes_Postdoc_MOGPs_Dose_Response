import numpy as np
import matplotlib.pyplot as plt
import GPy

from scipy.linalg import cholesky, cho_solve
from GPy.util import linalg

n = 1000
m = 6

# Kullback-Leibler Divergence between Gaussian distributions
def KLD_Gaussian(m1, V1, m2, V2, use_diag=False):
    Dim = m1.shape[0]
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        L1 = np.diag(np.sqrt(np.diag(V1)))
        L2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0 / np.diag(V2))
    else:
        L1 = cholesky(V1, lower=True)
        L2 = cholesky(V2, lower=True)
        V2_inv, _ = linalg.dpotri(np.asfortranarray(L2))
        # V2_inv = np.linalg.inv(V2)
    # print(V2_inv)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1 - m2).T, np.dot(V2_inv, (m1 - m2))) \
         - 0.5 * Dim + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L2)))) - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L1))))
    return np.abs(KL)  # This is to avoid any negative due to numerical instability

def KLD_Gaussian_NoChol(m1, V1, m2, V2, use_diag=False):
    Dim = m1.shape[0]
    # print("shape m1", m1.shape)
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        V1 = np.diag(np.sqrt(np.diag(V1)))
        V2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0 / np.diag(V2))
    else:
        # V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
        V2_inv = np.linalg.inv(V2)
    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1 - m2).T, np.dot(V2_inv, (m1 - m2))) \
         - 0.5 * Dim + 0.5 * np.log(np.linalg.det(V2)) - 0.5 * np.log(np.linalg.det(V1))
    return KL  # This is to avoid any negative due to numerical instability

def IndGP_KLRel_Juan_GPy(train_x, model, delta):
    N, P = train_x.shape  # Do we need to merge the train with val in here?
    relevance = np.zeros((N, P))
    # delta = 1.0e-4
    jitter = 1.0e-15
    # which_p = int(config.feature)
    print(f"Analysing {P} Features...")
    for p in range(P):
        for n in range(N):
            # x_plus = X[n,:].copy()
            x_plus = train_x[n:n + 1, :].copy()
            x_minus = train_x[n:n + 1, :].copy()
            x_plus[0, p] = x_plus[0, p] + delta
            x_minus[0, p] = x_minus[0, p] - delta

            # Make conditional predictions

            m1, V1 = model.predict(train_x[n:n + 1, :])  # predict_xn.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2, V2 = model.predict(x_plus)  # predict_xn_delta.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2_minus, V2_minus = model.predict(x_minus)  # predict_xn_delta_min.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            # np.random.seed(1)
            use_diag = False
            KL_plus = np.sqrt(KLD_Gaussian(m1, V1, m2, V2, use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            KL_minus = np.sqrt(KLD_Gaussian(m1, V1, m2_minus, V2_minus,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            relevance[n, p] = 0.5 * (KL_plus + KL_minus) / delta
    return np.mean(relevance, 0)

def Cond_KLRel_Juan_GPy(train_x,X2,y2, model, delta,use_diag = False):

    def pred_cond(x1_new,X2,y2,model):
        Ntasks = model.mul.coregion.B.shape[0]
        B = model.mul.coregion.B + np.eye(Ntasks) * model.mul.coregion.kappa
        K_X2X2 = model.kern.rbf.K(X2)
        b22K_inv = np.linalg.inv(B[1, 1] * K_X2X2 + np.eye(K_X2X2.shape[0]) * model.likelihood.variance)
        K_x1X2 = model.kern.rbf.K(x1_new, X2)
        b12Kx1X2_K22_inv = np.dot(B[0, 1] * K_x1X2, b22K_inv)
        m1_2 = np.dot(b12Kx1X2_K22_inv, y2)
        K_x1x1 = model.kern.rbf.K(x1_new)
        var1_2 = B[0, 0] * K_x1x1 + model.likelihood.variance - np.dot(b12Kx1X2_K22_inv, B[1, 0] * K_x1X2.T)
        return m1_2,var1_2

    N, P = train_x.shape  # Do we need to merge the train with val in here?
    relevance = np.zeros((N, P))
    # delta = 1.0e-4
    jitter = 1.0e-15
    #Ntasks = model.mul.coregion.B.shape[0]
    # which_p = int(config.feature)
    print(f"Analysing {P} Features...")
    for p in range(P):
        for n in range(N):
            # x_plus = X[n,:].copy()
            x_plus = train_x[n:n + 1, :].copy()
            x_minus = train_x[n:n + 1, :].copy()
            x_plus[0, p] = x_plus[0, p] + delta
            x_minus[0, p] = x_minus[0, p] - delta

            # Make conditional predictions

            m1,V1 = pred_cond(train_x[n:n + 1, :],X2,y2,model) #predict_xn.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2,V2 = pred_cond(x_plus ,X2,y2,model) #predict_xn_delta.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            m2_minus,V2_minus = pred_cond(x_minus,X2,y2,model) #predict_xn_delta_min.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
            # np.random.seed(1)
            use_diag = False
            KL_plus = np.sqrt(KLD_Gaussian(m1, V1, m2, V2,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            KL_minus = np.sqrt(KLD_Gaussian(m1, V1, m2_minus, V2_minus,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            relevance[n, p] = 0.5 * (KL_plus + KL_minus) / delta
    return np.mean(relevance, 0)

#x = np.linspace(0,1,1000)
#x = np.random.uniform(-1,1,1000)
np.random.seed(30)
x = np.random.uniform(-1., 1., (n, m))
#x = np.tile(np.linspace(-1, 1, n), (m, 1)).T
f = np.zeros((n,m))
f[:,0] = np.sin(2*np.pi*1*0.1*x[:,0])
f[:,1] = np.cos(2*np.pi*2*0.1*x[:,1])
f[:,2] = np.sin(2*np.pi*3*0.1*x[:,2])
f[:,3] = np.cos(2*np.pi*4*0.1*x[:,3])
f[:,4] = np.sin(2*np.pi*5*0.1*x[:,3])
#f[:,4] = np.exp(x[:,4]/4)
f[:,5] = np.cos(2*np.pi*4*0.1*x[:,5]**2)

f1 = f.copy()
f2 = f.copy()

f1 = f1/f1.std(0)
f2 = f2/f2.std(0)

f2 = f2 * np.array([1.0,0.1,1.0,0.1,0.1,1.0])
f1 = f1 * np.array([1.0,0.1,0.1,0.1,0.1,0.1])

y1 = (f1[:,0]+f1[:,1]+f1[:,2]+f1[:,3])+f1[:,4]+f1[:,5]+ np.random.normal(0, 0.3, (n,))
y2_all = (f2[:,0]+f2[:,1]+f2[:,2]+f2[:,3])+f2[:,4]+f2[:,5]+ np.random.normal(0, 0.3, (n,))

plt.figure(3)
plt.plot(x,y1,'.')

Nsmall = 25
rand_ind = np.random.permutation(n)
y2 = y2_all[rand_ind[0:Nsmall]].copy()
#y1 = y1_all[rand_ind[0:Nsmall]].copy()
#y2 = y2_all[0:Nsmall].copy()
Yall = np.concatenate((np.asmatrix(y1),np.asmatrix(y2)),1).T

#Ntrain = n  # //3
Xall = x.copy()
X1 = Xall.copy()
#X2 = Xall.copy()
X2 = Xall[rand_ind[0:Nsmall], :].copy()
#X1 = Xall[rand_ind[0:Nsmall], :].copy()
#X2 = Xall[0:Nsmall, :].copy()
Xtrain = np.concatenate((X1,X2))
Ylabel = np.concatenate((0.0*np.ones(X1.shape[0]),1.0*np.ones(X2.shape[0])))[:,None]
Xtrain = np.concatenate((Xtrain,Ylabel),1)

Ntasks = 2
kern = GPy.kern.RBF(m, active_dims=[0,1,2,3,4,5]) ** GPy.kern.Coregionalize(1, output_dim=Ntasks, rank=Ntasks)

model = GPy.models.GPRegression(Xtrain, Yall, kern)
model.optimize()

m_pred,v_pred = model.predict(Xtrain,full_cov=False)
plt.figure(1)
plt.plot(Yall,'bx')
plt.plot(m_pred, 'ro')
plt.plot(m_pred+2*np.sqrt(v_pred), '--m')
plt.plot(m_pred - 2 * np.sqrt(v_pred), '--m')

#x1_new = X1[0:1,:]

delta = 0.0001
rel_CondKL_juan_MOGP = Cond_KLRel_Juan_GPy(X1,X2, y2[:,None],model, delta)
#rel_CondKL_juan_MOGP = Cond_KLRel_Juan_GPy(X2, X1, y1[:, None], model, delta)
#rel_CondKL_juan_MOGP = Cond2_KLRel_Juan_GPy(X1, X2, y2[:, None], model, delta)
#rel_CondKL_juan_MOGP = Cond3_KLRel_Juan_GPy(Xtrain[0:n], Xtrain[n:], model, delta)
#rel_CondKL_juan_MOGP = Cond3_KLRel_Juan_GPy(Xtrain[n:],Xtrain[0:n], model, delta)
#rel_CondKL_juan_MOGP = Cond3_KLRel_Juan_GPy(Xtrain, Xtrain, model, delta)
rel_CondKL_juan_norm_MOGP = rel_CondKL_juan_MOGP/rel_CondKL_juan_MOGP.max()

#Repeats_CondKL_juan_MOGP.append(rel_CondKL_juan_norm_MOGP.copy())

print(rel_CondKL_juan_norm_MOGP)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
kern_Ind = GPy.kern.RBF(m, active_dims=[0,1,2,3,4,5])

model_Ind = GPy.models.GPRegression(X2,y2[:,None], kern_Ind)
model_Ind.optimize()

m_pred2, v_pred2 = model_Ind.predict(X2, full_cov=False)
plt.figure(2)
plt.plot(y2[:,None], 'bx')
plt.plot(m_pred2, 'ro')
plt.plot(m_pred2 + 2 * np.sqrt(v_pred2), '--m')
plt.plot(m_pred2 - 2 * np.sqrt(v_pred2), '--m')

rel_KL_juan_IndGP = IndGP_KLRel_Juan_GPy(X2, model_Ind, delta)
rel_KL_juan_norm_IndGP = rel_KL_juan_IndGP / rel_KL_juan_IndGP.max()
print(rel_KL_juan_norm_IndGP)
