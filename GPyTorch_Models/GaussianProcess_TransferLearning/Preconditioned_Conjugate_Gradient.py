import torch

Nseed = 1
torch.manual_seed(Nseed)
t = 5
N = 30
A0 = torch.randn(N,N)
A = torch.matmul(A0.T,A0)
#y = torch.tensor([1.3,-1.1,2.34,0.055,1.075])[:,None] #torch.randn((N,1))
y = torch.rand((N,1))
zt = torch.randn((N,t))
U_true = torch.cat([y,zt],1)
# Matrix to solve against B has shape N x (t+1)
# B = [y z1 z2 ... zt]
B = torch.matmul(A,U_true)
# Preconditioner Matrix
Phat_inv = torch.eye(B.shape[0])

U0 = torch.zeros_like(B) # Current solutions
# In the paper BBMM appears as R0 = torch.matmul(A,U0) - B,
# but the correct reference of the Saar: "Iterative methods for sparse linear systems"
# is R0 = B - torch.matmul(A,U0)
R0 = B - torch.matmul(A,U0) # Current errors
Z0 = torch.matmul(Phat_inv,R0) # Preconditioned errors
D0 = Z0.clone()
p_iter = 5
# Each matrix Tt is p_iter x p_iter
# In the supplementary doc. for BBMM method the for loop appears as range(0,t)
# though to avoid confusion with the t vectors zt, and following the main paper before Eq. (6)
# then each Tt is p x p (with p as number of iterations)
Tt = torch.zeros(t+1,p_iter,p_iter)  #here t+1, though the first T0 is related to the solution A^-1y, should not be used
Dj = D0.clone()
Rj = R0.clone()
Uj = U0.clone()
Zj = Z0.clone()
vect_1 = torch.ones_like(y) # Vector of ones with same size of y
Tolerance = 1e-3*torch.ones(t+1)
for i in range(p_iter):
    Vj = torch.matmul(A,Dj)
    alphaj = torch.matmul((Rj * Zj).t(), vect_1)/torch.matmul((Dj*Vj).t(), vect_1)
    Uj = Uj + torch.matmul(Dj,torch.diag(alphaj.flatten()))
    Rj = Rj - torch.matmul(Vj,torch.diag(alphaj.flatten()))
    "TODO: check what to do whan it ends and it does not complete all p_iter"
    # if torch.prod((torch.norm(Rj,dim=0)<Tolerance)):
    #     print(f"error: {torch.norm(Rj,dim=0)}, iter:{i}")
    #     break
    Zj_old = Zj.clone()
    Zj = torch.matmul(Phat_inv,Rj)
    betaj = torch.matmul((Zj * Zj).t(), vect_1)/torch.matmul((Zj_old * Zj_old).t(), vect_1)
    Dj = Zj + torch.matmul(Dj,torch.diag(betaj.flatten()))
    if i<p_iter-1:
        # Here the diagonal i,i+1 and i+1,i is sqrt(beta0)/sqrt(alpha0) ...
        Tt[:, i, i+1] = (torch.sqrt(betaj) / alphaj).flatten()
        Tt[:, i+1, i] = (torch.sqrt(betaj) / alphaj).flatten() #Tt[:, i, i+1]
    if i == 0:
        Tt[:, i, i] = (1.0/alphaj).flatten()
    else:
        # Here the diagonal i,i is 1.0 / alphaj for i=0; and 1.0 / alphaj + beta_old/alpha_old
        print(i)
        Tt[:, i, i] = (1.0 / alphaj + betaj_old / alphaj_old).flatten()
        #print(f"betaj:{betaj}")
        #print(f"betaj_old:{betaj_old}")

    alphaj_old = alphaj.clone()
    betaj_old = betaj.clone()

import numpy as np
import scipy.linalg
# Compute eigendecomposition of matrices Tt
lambdai,Mi=torch.linalg.eig(Tt)
# Create the vector e1 which is equal to the first row of the identity matrix
e1 = np.zeros(Tt[0,:,:].shape[0])[:,None]
Tt_np = Tt.numpy()
e1[0] = 1.0
aprx_log_det = 0.0
# Here the average is for only Tt matrices related to the vectors zt, so we avoid the first Tt, i.e., T0
for i in range(1,t+1):
    #print(i)
    aprx_log_det += np.matmul(np.matmul(e1.T,scipy.linalg.logm(Tt_np[i,:,:])),e1)
    print(aprx_log_det)
aprx_log_det = (N)*aprx_log_det/(t)
print(f"Approx. log|A|={aprx_log_det}")
print(f"Actual log|A|={torch.log(torch.linalg.det(A))}")