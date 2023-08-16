import torch

Nseed = 1
torch.manual_seed(Nseed)
t = 0
N = 30000
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
p_iter = 20
# Each matrix Tt is p_iter x p_iter
# In the supplementary doc. for BBMM method the for loop appears as range(0,t)
# though to avoid confusion with the t vectors zt, and following the main paper before Eq. (6)
# then each Tt is p x p (with p as number of iterations)
Tt = torch.zeros(t,p_iter,p_iter)
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
    if torch.prod((torch.norm(Rj,dim=0)<Tolerance)):
        print(f"error: {torch.norm(Rj,dim=0)}, iter:{i}")
        break
    Zj_old = Zj.clone()
    Zj = torch.matmul(Phat_inv,Rj)
    betaj = torch.matmul((Zj * Zj).t(), vect_1)/torch.matmul((Zj_old * Zj_old).t(), vect_1)
    Dj = Zj + torch.matmul(Dj,torch.diag(betaj.flatten()))

