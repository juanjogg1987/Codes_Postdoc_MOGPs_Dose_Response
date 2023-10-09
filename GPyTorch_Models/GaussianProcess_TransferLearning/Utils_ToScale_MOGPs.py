import torch

def CG_Lanczos(A,y,t = 100,p_iter = 30,tolerance = None):

    assert A.shape[0] == A.shape[1]   # This is to assert square matrix
    assert y.shape[1] >= 1            # This is to assert y as column vector
    " Ncols_y is to know how many columns to return for the solution A^(-1)y, y is a matrix N x Ncols_y"
    Ncols_y = y.shape[1]  # This variable should coincide with the columns of Uj return at the end

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
    " Here t+Ncols_y, the first T0 to TNcols_y is related to the solution A^-1y, should not be used"
    Tt = torch.zeros(t+Ncols_y,p_iter,p_iter)
    Dj = D0.clone()
    Rj = R0.clone()
    Uj = U0.clone()
    Zj = Z0.clone()
    vect_1 = torch.ones(N,1) #torch.ones_like(y) # Vector of ones with same N length as y
    "NOTE: Here the Tolerance should be bigger if the matrix size is bigger"
    "It seems that we have to be less strict with the approximation if having large matrices"
    "This Tolerance is key for making the approximation work properly."
    if tolerance is None:
        Tolerance = 20.0 * N/100.0 * torch.ones(t + Ncols_y) #3.0 * N/100.0 * torch.ones(t + Ncols_y)
    else:
        Tolerance = tolerance*torch.ones(t+Ncols_y)
    for i in range(p_iter):
        Vj = torch.matmul(A,Dj)
        alphaj = torch.matmul((Rj * Zj).t(), vect_1)/torch.matmul((Dj*Vj).t(), vect_1)
        #alphaj = torch.clip(alphaj,1.0e-2,torch.inf)
        #print(f"it:{i} Rj",Rj)
        Uj = Uj + torch.matmul(Dj,torch.diag(alphaj.flatten()))
        Rj = Rj - torch.matmul(Vj,torch.diag(alphaj.flatten()))
        #Rj = torch.sign(Rj)*torch.clip(torch.abs(Rj),1.0e-9,torch.inf)
        #print(Rj)
        "TODO: check what to do when it ends by tolerance and it does not complete all p_iter"
        if torch.prod((torch.norm(Rj,dim=0)<Tolerance)):
            #print(f"error: {torch.norm(Rj,dim=0)}, iter:{i}")
            #print(f"alpha:{alphaj}")
            #print(f"beta:{betaj}")
            Tt = Tt[:, 0:i+1, 0:i+1]
            # if i>7:
            #     Tt = Tt[:, 0:i-7, 0:i-7]
            # else:
            #     Tt = Tt[:, 0:i + 1, 0:i + 1]
            break
        Zj_old = Zj.clone()
        Zj = torch.matmul(Phat_inv,Rj)
        betaj = torch.matmul((Zj * Zj).t(), vect_1)/torch.matmul((Zj_old * Zj_old).t(), vect_1)
        #betaj = torch.clip(betaj, 1.0e-2, torch.inf)
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
    #try:
    lambdai,Mi=torch.linalg.eig(Tt)
    Mi = torch.real(Mi)
    # except:
    # print("Exception")
    # Tt = Tt[:, 0:-1, 0:-1]
    # print(Tt[1,:,:])
    # lambdai, Mi = torch.linalg.eig(Tt)

    aprx_log_det = 0.0
    # Here the average is for only Tt matrices related to the vectors zt, so we avoid the first Tt, i.e., T0
    for i in range(Ncols_y,t+Ncols_y):
        "The paper of BBMM does not include the multiplication by N, probably a typo!"
        "Multiplying torch.matmul(e1,Mi[i,:,:]) is equivalent to just access the first row of each Mi"
        "So we directly use Mi[i, 0:1, :] in the multiplication below"
        aprx_log_det += N*torch.matmul(torch.matmul(Mi[i, 0:1, :], torch.diag(torch.log(torch.abs(lambdai[i, :])))), Mi[i, 0:1, :].t())/(t)
        #print(aprx_log_det)
    print(f"Approx. log|A|={aprx_log_det}")
    #print(f"Actual log|A|={torch.linalg.slogdet(A)}")

    return Uj[:,0:Ncols_y],aprx_log_det  #return the solution A^(-1)y (Uj[:,0][:,None]) and Log|A| (aprx_log_det)