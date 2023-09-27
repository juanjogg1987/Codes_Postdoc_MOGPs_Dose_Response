import torch

def CG_Lanczos(A,y,t = 100,p_iter = 30):

    assert A.shape[0] == A.shape[1]   # This is to assert square matrix
    assert y.shape[1] == 1            # This is to assert y as column vector

    N = A.shape[0]
    zt = torch.randn((N,t))
    B = torch.cat([y,zt],1)
    # Matrix to solve against B has shape N x (t+1)
    # B = [y z1 z2 ... zt]
    #U_true = torch.linalg.solve(A,B)
    # Preconditioner Matrix
    Phat_inv = torch.eye(B.shape[0])

    U0 = torch.zeros_like(B) # Current solutions
    # In the paper BBMM appears as R0 = torch.matmul(A,U0) - B,
    # but the correct reference of the Saar: "Iterative methods for sparse linear systems"
    # is R0 = B - torch.matmul(A,U0)
    R0 = B - torch.matmul(A,U0) # Current errors
    Z0 = torch.matmul(Phat_inv,R0) # Preconditioned errors
    D0 = Z0.clone()

    if p_iter > N:p_iter = N;
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
    Tolerance = 1e-2*torch.ones(t+1)
    print(f"A:{A}")
    print(f"Dj:{Dj}")
    for i in range(p_iter):
        Vj = torch.matmul(A,Dj)
        alphaj = torch.matmul((Rj * Zj).t(), vect_1)/torch.matmul((Dj*Vj).t(), vect_1)
        #print(f"it:{i} Rj",Rj)
        Uj = Uj + torch.matmul(Dj,torch.diag(alphaj.flatten()))
        Rj = Rj - torch.matmul(Vj,torch.diag(alphaj.flatten()))
        "TODO: check what to do when it ends by tolerance and it does not complete all p_iter"
        if torch.prod((torch.norm(Rj,dim=0)<Tolerance)):
            #print(f"error: {torch.norm(Rj,dim=0)}, iter:{i}")
            Tt = Tt[:, 0:i+1, 0:i+1]
            break
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
            #print(i)
            Tt[:, i, i] = (1.0 / alphaj + betaj_old / alphaj_old).flatten()

        alphaj_old = alphaj.clone()
        betaj_old = betaj.clone()

    # Compute eigendecomposition of matrices Tt
    try:
        lambdai,Mi=torch.linalg.eig(Tt)
        Mi = torch.real(Mi)
    except:
        print("Exception")
        Tt = Tt[:, 0:-1, 0:-1]
        print(Tt[1,:,:])
        lambdai, Mi = torch.linalg.eig(Tt)

    aprx_log_det = 0.0
    # Here the average is for only Tt matrices related to the vectors zt, so we avoid the first Tt, i.e., T0
    for i in range(1,t+1):
        "The paper of BBMM does not include the multiplication by N, probably a typo!"
        "Multiplying torch.matmul(e1,Mi[i,:,:]) is equivalent to just access the first row of each Mi"
        "So we directly use Mi[i, 0:1, :] in the multiplication below"
        aprx_log_det += N*torch.matmul(torch.matmul(Mi[i, 0:1, :], torch.diag(torch.log(torch.abs(lambdai[i, :])))), Mi[i, 0:1, :].t())/(t)
        #print(aprx_log_det)
    print(f"Approx. log|A|={aprx_log_det}")
    print(f"Actual log|A|={torch.linalg.slogdet(A)}")

    return Uj[:,0][:,None],aprx_log_det  #return the solution A^(-1)y (Uj[:,0][:,None]) and Log|A| (aprx_log_det)