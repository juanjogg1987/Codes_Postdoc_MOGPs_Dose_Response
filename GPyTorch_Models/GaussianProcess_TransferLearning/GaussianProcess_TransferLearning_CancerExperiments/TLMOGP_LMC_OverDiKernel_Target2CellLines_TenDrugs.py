import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from importlib import reload
import TransferLearning_Kernels
reload(TransferLearning_Kernels)
from TransferLearning_Kernels import TL_Kernel_var, Kernel_CrossDomains, TLRelatedness

from numpy import linalg as la
import numpy as np

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
def isPD_torch(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = torch.linalg.cholesky(B)
        return True
    except torch.linalg.LinAlgError:
        return False
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    if isPD(A):
        return A

    print("Correcting Matrix to be PSD!!")
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

class LogMarginalLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,mu,L,y):
        y = y.T.reshape(-1, 1)
        assert mu.shape == y.shape
        alpha = torch.linalg.solve(L.t(), torch.linalg.solve(L, y-mu))
        N = y.shape[0]
        return -0.5*torch.matmul(y.t()-mu.t(),alpha)-torch.diag(L).log().sum()-N*0.5*torch.log(torch.tensor([2*torch.pi]))
class TLMOGaussianProcess(nn.Module):
    'This model expects the data from source and target domains with:'
    'idxS as a list of labels with integer values between 0 to NDomains-2'
    'by deafult the idxT for the target domain will be labeled as NDomains-1'
    'i.e., if NDomains = 3, the idxS is a list with values between 0 and 1, by default idxT has all values equal to 2'
    'The model expects the input and output of the target domain: xT (shape: [N,P]), yT (shape: [N,D])'
    'with P as the number of input features and D as the number of outputs'
    'The input and output of the source domains: xS, yS with a list idxS with their labels as per their domains'
    'NOTE: The model would expect the data to be sorted according to the idxS'
    'For instance, idxS = [0,0,1,2,2,2] implies NDomains = 4 with N = 2 in first domain,'
    'N = 1 in second domain, and N = 3 in third domain, if N = 2 in fourth (target) domain, then idxT = [3,3]'
    'The model also would expect a list of the Drug Concentrations used, it should have the same size as D output'
    def __init__(self,xT,yT,xS,yS,idxS,DrugC,NDomains):
        super().__init__()
        self.Douts = yT.shape[1]
        self.DrugC = torch.Tensor(DrugC)[:,None] #We treat the Drug Concentrations as a float tensor vector [N,1]
        self.NDomains = NDomains
        #self.Q = 2    #Number of Kernel_CrossDomains to linearly combine
        assert self.DrugC.shape[0] == yT.shape[1]  #DrugC length should be equal to D number of outputs
        self.Pfeat = xT.shape[1]
        #torch.kron(mat1, mat2.reshape(-1)).reshape(-1, 429)
        #self.xT = torch.kron(torch.ones(self.Douts, 1),xT)  #This is to replicate xT as per the D number of outputs
        self.xT = torch.kron(torch.ones(self.Douts, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)  # This is to replicate xT as per the D number of outputs
        #self.xS = torch.kron(torch.ones(self.Douts, 1), xS)  # This is to replicate xS as per the D number of outputs
        self.xS = torch.kron(torch.ones(self.Douts, 1),xS.reshape(-1)).reshape(-1, self.Pfeat)  #This is to replicate xS as per the D number of outputs
        self.yT = yT.T.reshape(-1, 1)  # This is to vectorise by staking the columns (vect(yT))
        self.yS = yS.T.reshape(-1,1) #This is to vectorise by staking the columns (vect(yS))
        self.DrugC_xT = torch.kron(self.DrugC,torch.ones(xT.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        self.DrugC_xS = torch.kron(self.DrugC,torch.ones(xS.shape[0], 1)) #Rep. Concentr. similar to coreginalisation
        self.idxS = idxS * self.Douts #Replicate the Source domain index as per the number of outputs
        self.idxT = [NDomains - 1] * xT.shape[0] * self.Douts #Replicate the target domain index as per the number of outputs
        self.all_y = torch.cat([self.yS, self.yT])
        self.TLCovariance = [Kernel_CrossDomains(NDomains=NDomains),Kernel_CrossDomains(NDomains=NDomains),Kernel_CrossDomains(NDomains=NDomains)] #gpytorch.kernels.RBFKernel()
        self.LambdaDiDj = TLRelatedness(NDomains=NDomains)
        #self.CoregCovariance = gpytorch.kernels.RBFKernel()
        self.CoregCovariance = [gpytorch.kernels.MaternKernel(1.5),gpytorch.kernels.MaternKernel(2.5),gpytorch.kernels.MaternKernel(2.5)]
        self.Train_mode = True
        self.lik_std_noise = torch.nn.Parameter(1.0*torch.ones(NDomains)) #0.3*torch.rand(NDomains)+0.01
        #self.lik_std_noise = 0.05 * torch.ones(NDomains)
        self.mu_star = torch.zeros(self.yT.shape) #mu has the shape of the new replicated along the outputs yT
        self.L = torch.eye(self.yT.shape[0])
        self.all_L = torch.eye(self.all_y.shape[0])
        "TODO: I might need the self.LSS also here in order to be able to predict without optimising"

    def forward(self,xT, DrugC_new = None,noiseless = True):
        if self.Train_mode:
            #xT = torch.kron(torch.ones(self.Douts, 1), xT)
            xT = torch.kron(torch.ones(self.Douts, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)
            assert (xT == self.xT).sum() == (xT.shape[0]*xT.shape[1]) #This is just to check if the xT to init the model is the same
            assert xT.shape[1] == self.xT.shape[1]  #Thiis is to check if the xT to init the model has same Pfeatures

            "Below the TLCovariance and LambdaDiDj have to coincide in the same indexes idx1 and idx2"
            # Here we compute the Covariance matrices between source-target, source-source and target domains
            KTS = self.LambdaDiDj(xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate()*(self.CoregCovariance[0](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[0](xT,self.xS,idx1=self.idxT,idx2=self.idxS).evaluate() + \
                                                                                        self.CoregCovariance[1](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[1](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate()+\
                                                                                        self.CoregCovariance[2](self.DrugC_xT,self.DrugC_xS).evaluate()*self.TLCovariance[2](xT, self.xS, idx1=self.idxT, idx2=self.idxS).evaluate())
            KSS = self.LambdaDiDj(self.xS,idx1=self.idxS).evaluate()*(self.CoregCovariance[0](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[0](self.xS,idx1=self.idxS).evaluate()+ \
                                                                      self.CoregCovariance[1](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[1](self.xS, idx1=self.idxS).evaluate()+\
                                                                      self.CoregCovariance[2](self.DrugC_xS,self.DrugC_xS).evaluate()*self.TLCovariance[2](self.xS, idx1=self.idxS).evaluate())
            KTT = self.LambdaDiDj(self.xT,idx1=self.idxT).evaluate()*(self.CoregCovariance[0](self.DrugC_xT,self.DrugC_xT).evaluate()*self.TLCovariance[0](self.xT,idx1=self.idxT).evaluate()+ \
                                                                      self.CoregCovariance[1](self.DrugC_xT,self.DrugC_xT).evaluate() * self.TLCovariance[1](self.xT, idx1=self.idxT).evaluate()+\
                                                                      self.CoregCovariance[2](self.DrugC_xT,self.DrugC_xT).evaluate() * self.TLCovariance[2](self.xT, idx1=self.idxT).evaluate())

            # Here we include the respective noise terms associated to each domain
            CSS = KSS + torch.diag(self.lik_std_noise[self.idxS].pow(2))
            CTT = KTT + torch.diag(self.lik_std_noise[self.idxT].pow(2))

            # The code below aim to correct for numerical instabilities when CSS becomes Non-PSD
            if not isPD_torch(CSS):
                CSS_aux = CSS.clone()
                with torch.no_grad():
                    CSS_aux = torch.from_numpy(nearestPD(CSS_aux.numpy()))
                CSS = 0.0*CSS + CSS_aux  #This operation aims to keep the gradients working over lik_std_noise

            self.LSS = torch.linalg.cholesky(CSS)
            alphaSS1 = torch.linalg.solve(self.LSS, self.yS)
            alphaSS = torch.linalg.solve(self.LSS.t(), alphaSS1)

            # Compute the mean of the conditional distribution p(yT|XT,XS,yS)
            self.mu_star = torch.matmul(KTS,alphaSS)
            # Compute the Covariance of the conditional distribution p(yT|XT,XS,yS)
            vTT = torch.linalg.solve(self.LSS, KTS.t())
            C_star = CTT - torch.matmul(vTT.t(),vTT) #+ 1e-4*torch.eye(xT.shape[0])  #jitter?
            # The code below aim to correct for numerical instabilities when CSS becomes Non-PSD
            if not isPD_torch(C_star):
                C_star_aux = C_star.clone()
                with torch.no_grad():
                    C_star_aux = torch.from_numpy(nearestPD(C_star_aux.numpy()))
                C_star = 0.0 * C_star + C_star_aux  # This operation aims to keep the gradients working over lik_std_noise
            self.L = torch.linalg.cholesky(C_star)

            return self.mu_star, self.L  # here we return the mean and covariance
        else:
            "We replicate the target domain index as per the number of drug concentration we want to test"
            "notice that it is not limited to D number of outputs, but actually the number of concentrations"
            # Here we compute the full covariance of xS and xT together
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS + self.idxT
            self.DrugC_xSxT = torch.cat([self.DrugC_xS, self.DrugC_xT])
            all_K_xST = self.LambdaDiDj(xST, idx1=idxST).evaluate()*(self.CoregCovariance[0](self.DrugC_xSxT).evaluate() * self.TLCovariance[0](xST, idx1=idxST).evaluate()+ \
                                                                     self.CoregCovariance[1](self.DrugC_xSxT).evaluate() * self.TLCovariance[1](xST, idx1=idxST).evaluate()+\
                                                                     self.CoregCovariance[2](self.DrugC_xSxT).evaluate() * self.TLCovariance[2](xST, idx1=idxST).evaluate())
            all_K_xST_noise = all_K_xST + torch.diag(self.lik_std_noise[idxST].pow(2))  # + 0.1*torch.eye(xST.shape[0]) #Jitter?
            if not isPD_torch(all_K_xST_noise):
                with torch.no_grad():
                    all_K_xST_noise = torch.from_numpy(nearestPD(all_K_xST_noise.numpy()))
            self.all_L = torch.linalg.cholesky(all_K_xST_noise)  # + 1e-4*torch.eye(xST.shape[0])

            # Here we receive a list of possible drug concentrations to predict
            if DrugC_new is None:
                DrugC_new = self.DrugC
            else:
                DrugC_new = torch.Tensor(DrugC_new)[:, None]
            #Below we replicate the target domain index as per the number of drug concentration we want to test
            NewDouts = DrugC_new.shape[0]
            idxT = [self.NDomains - 1] * xT.shape[0] * NewDouts
            DrugC_xT = torch.kron(DrugC_new, torch.ones(xT.shape[0], 1))
            "Be careful with operation using xT.shape, from here it changes the original shape"
            #xT = torch.kron(torch.ones(NewDouts, 1), xT)
            xT = torch.kron(torch.ones(NewDouts, 1), xT.reshape(-1)).reshape(-1, self.Pfeat)
            alpha1 = torch.linalg.solve(self.all_L, self.all_y)
            alpha = torch.linalg.solve(self.all_L.t(), alpha1)
            KTT_xnew_xnew = self.LambdaDiDj(xT, idx1=idxT).evaluate()*(self.CoregCovariance[0](DrugC_xT).evaluate() * self.TLCovariance[0](xT, idx1=idxT).evaluate()+ \
                                                                       self.CoregCovariance[1](DrugC_xT).evaluate() * self.TLCovariance[1](xT,idx1=idxT).evaluate()+\
                                                                       self.CoregCovariance[2](DrugC_xT).evaluate() * self.TLCovariance[2](xT,idx1=idxT).evaluate())
            xST = torch.cat([self.xS, self.xT])
            idxST = self.idxS+self.idxT

            # Rep. Concentr. similar to coreginalisation
            K_xnew_xST = self.LambdaDiDj(xT,xST, idx1=idxT,idx2=idxST).evaluate()*(self.CoregCovariance[0](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[0](xT,xST, idx1=idxT,idx2=idxST).evaluate()+ \
                                                                                   self.CoregCovariance[1](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[1](xT, xST, idx1=idxT, idx2=idxST).evaluate()+\
                                                                                   self.CoregCovariance[2](DrugC_xT,self.DrugC_xSxT).evaluate() * self.TLCovariance[2](xT, xST, idx1=idxT, idx2=idxST).evaluate())

            f_mu = torch.matmul(K_xnew_xST, alpha)
            v = torch.linalg.solve(self.all_L, K_xnew_xST.t())

            if noiseless:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) #+ 1e-2*torch.eye(xT.shape[0])  #I had to add this Jitter
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            else:
                f_Cov = KTT_xnew_xnew - torch.matmul(v.t(),v) + torch.diag(self.lik_std_noise[idxT].pow(2)) + 1e-5*torch.eye(xT.shape[0])
                f_Cov = torch.from_numpy(nearestPD(f_Cov.numpy()))
            return f_mu, f_Cov

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nseed = 2
torch.manual_seed(Nseed)
import math
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
random.seed(Nseed)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we preprocess and prepare our data"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small
    k = -1/s (s -shape parameter)
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid

    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import getopt
import sys

warnings.filterwarnings("ignore")
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:k:w:r:p:c:a:n:')
        # opts = dict(opts)
        # print(opts)
        self.N_iter_epoch = 1200    #number of iterations
        self.which_seed = 1011    #change seed to initialise the hyper-parameters
        self.rank = 7
        self.scale = 1
        self.weight = 1
        self.bash = "1"
        self.N_CellLines_perc = 100   #Here we treat this variable as percentage. Try to put this values as multiple of Num_drugs?
        self.sel_cancer = 3
        self.seed_for_N = 1

        for op, arg in opts:
            # print(op,arg)
            if op == '-i':
                self.N_iter_epoch = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg
            if op == '-k':  # ran(k)
                self.rank = arg
            if op == '-s':  # (r)and seed
                self.scale = arg
            if op == '-p':  # (p)ython bash
                self.bash = arg
            if op == '-w':
                self.weight = arg
            if op == '-c':
                self.N_CellLines_perc = arg
            if op == '-a':
                self.sel_cancer = arg
            if op == '-n':
                self.seed_for_N = arg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
config = commandLine()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
              2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

#indx_cancer = np.array([0,1,2,3,4])
indx_cancer_train = np.array([int(config.sel_cancer)])

name_file_cancer = dict_cancers[indx_cancer_train[0]]
name_file_cancer_target = dict_cancers[0]
print("Source Cancer:",name_file_cancer)
print("Target Cancer:",name_file_cancer_target)

df_to_read = pd.read_csv(_FOLDER + name_file_cancer)#.sample(n=N_CellLines,random_state = rand_state_N)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df_to_read_target = pd.read_csv(_FOLDER + name_file_cancer_target)#.sample(n=N_CellLines,random_state = rand_state_N)

"Split of data into Training and Testing for the Source and Target domains"
Index_sel = (df_to_read["DRUG_ID"] == 1036) | (df_to_read["DRUG_ID"] == 1061)| (df_to_read["DRUG_ID"] == 1373) \
            | (df_to_read["DRUG_ID"] == 1039) | (df_to_read["DRUG_ID"] == 1560) | (df_to_read["DRUG_ID"] == 1057) \
            | (df_to_read["DRUG_ID"] == 1059)| (df_to_read["DRUG_ID"] == 1062) | (df_to_read["DRUG_ID"] == 2096) \
            | (df_to_read["DRUG_ID"] == 2045)

df_SourceCancer_all = df_to_read[Index_sel]
df_all = df_SourceCancer_all.reset_index().drop(columns=['index'])
df_source = df_all.dropna()

Index_sel_target = (df_to_read_target["DRUG_ID"] == 1036) | (df_to_read_target["DRUG_ID"] == 1061)| (df_to_read_target["DRUG_ID"] == 1373) \
            | (df_to_read_target["DRUG_ID"] == 1039) | (df_to_read_target["DRUG_ID"] == 1560) | (df_to_read_target["DRUG_ID"] == 1057) \
            | (df_to_read_target["DRUG_ID"] == 1059)| (df_to_read_target["DRUG_ID"] == 1062) | (df_to_read_target["DRUG_ID"] == 2096) \
            | (df_to_read_target["DRUG_ID"] == 2045)

df_TargetCancer_all = df_to_read_target[Index_sel_target]
df_all_target = df_TargetCancer_all.reset_index().drop(columns=['index'])
df_all_target = df_all_target.dropna()

"The ones below have 9 drugs tested"
"The cell line 946359 (all nores); 749709 (1 resp 1 parcial)  has been tested in 9 of the 10 drugs"
"907046 (2 resp 1 parcial); 1290798 (3 parcial); 905946 (1 resp 1 parcial)"
"908121 (2 resp 1 parcial); 1240172 (non resp); 906826 (2 resp should try this one)"

"The ones below have 8 drugs tested"
"1298157 (1 resp 1 parcial); 910927 (2 resp 1 parcial)"
CosmicID_target = [749709,908121,907046]
Index_sel_CosmicIDs = (df_all_target['COSMIC_ID'] == CosmicID_target[0]) | (df_all_target['COSMIC_ID'] == CosmicID_target[1])| (df_all_target['COSMIC_ID'] == CosmicID_target[2])
#CosmicID_target = #749709 #910927 #1298157 #906826 #1240172 #908121 #905946 #1290798 #907046 #749709 #946359
#df_target = df_all_target[df_all_target['COSMIC_ID']==CosmicID_target].reset_index().drop(columns=['index'])

df_target = df_all_target[Index_sel_CosmicIDs].reset_index().drop(columns=['index'])

idx_train = np.array([0,1,2,6,7,8,12,13,14])  #Exp1:3,4,8 ,Exp2 (906826):0,2,6  Exp3 ():
idx_test = np.delete(np.arange(0,df_target.shape[0]),idx_train)

df_target_test = df_target.iloc[idx_test]
df_target_train = df_target.iloc[idx_train]

Name_DrugID_train = df_target_train['DRUG_ID'].values
Name_DrugID_test = df_target_test['DRUG_ID'].values

df_source_and_target = pd.concat([df_source,df_target_train])

# Here we just check that from the column index 25 the input features start
start_pos_features = 25
print("first feat Source:",df_all.columns[start_pos_features])
print("first feat Target:",df_all_target.columns[start_pos_features])
assert df_all.columns[start_pos_features] == df_all_target.columns[start_pos_features]

df_feat = df_source_and_target[df_source_and_target.columns[start_pos_features:]]
Names_All_features = df_source_and_target.columns[start_pos_features:]
Idx_Non_ZeroStd = np.where(df_feat.std()!=0.0)
Names_features_NonZeroStd = Names_All_features[Idx_Non_ZeroStd]

"This is a scaler to scale features in similar ranges"
scaler = MinMaxScaler().fit(df_source_and_target[Names_features_NonZeroStd])

xT_train = scaler.transform(df_target_train[Names_features_NonZeroStd])
xT_test = scaler.transform(df_target_test[Names_features_NonZeroStd])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we extract from the dataframe the D outputs related to each dose concentration"
"Below we select 7 concentration since GDSC2 has that number for the Drugs"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Dnorm_cell = 7  #For GDSC2 this is the number of dose concentrations

"Here we extract the target domain outputs yT"

yT = np.clip(df_target["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(yT.shape)
for i in range(2, Dnorm_cell+1):
    yT = np.concatenate((yT, np.clip(df_target["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain size: ", yT.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Since we fitted the dose response curves with a Sigmoid4_parameters function"
"We extract the optimal coefficients in order to reproduce such a Sigmoid4_parameters fitting"
"Here we extract the target domain parameters"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
params_4_sig_target = df_target["param_" + str(1)].values[:, None]
for i in range(2, 5):  #here there are four params for sigmoid4
    params_4_sig_target = np.concatenate((params_4_sig_target, df_target["param_" + str(i)].values[:, None]), 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"In this sections we will extract the summary metrics AUC, Emax and IC50 from the Sigmoid4_parameters functions"
"These metrics are used as the references to compute the error metrics"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn import metrics
from importlib import reload
import Utils_SummaryMetrics_KLRelevance
import Utils_SummaryMetrics_KLRelevance as MyUtils
reload(MyUtils)

"Be careful that x starts from 0.111111 for 9 or 5 drug concentrations in GDSC1 dataset"
"but x starts from 0.142857143 for the case of 7 drug concentrations in GDSC2 dataset"
"The function Get_IC50_AUC_Emax is implemented in Utils_SummaryMetrics_KLRelevance.py to extract the summary metrics"
x_lin = np.linspace(0.142857, 1, 1000)
x_real_dose = np.linspace(0.142857, 1, Dnorm_cell)  #Here is Dnorm_cell due to using GDSC2 that has 7 doses
Ydose50,Ydose_res,IC50,AUC,Emax = MyUtils.Get_IC50_AUC_Emax(params_4_sig_target,x_lin,x_real_dose)

def my_plot(posy,fig_num,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug):
    plt.figure(fig_num)
    plt.plot(x_lin, Ydose_res[posy])
    plt.plot(x_real_dose, y_train_drug[posy, :], '.')
    plt.plot(IC50[posy], Ydose50[posy], 'rx')
    plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
    plt.title(f"AUC = {AUC[posy]}")
    plt.legend(['Sigmoid4','Observations','IC50','Emax'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Here we can visualise the values of the GDSC2 dataset with the fitting of Sigmoid4_parameters function"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
posy = 8   #select the location you want to plot, do not exceed the Ytrain length
my_plot(posy,0,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,yT)

yT_train = yT[idx_train]
yT_test = yT[idx_test]

"Here we define the set of all the possible cell-lines (Cosmic_ID) that belong to Melanoma cancer"
"NOTE: Not all cell-lines have been tested with all ten drugs, so some cell-line might have more"
"dose response curves than others"

myset_source = set(df_source['COSMIC_ID'].values)
myLabels = np.arange(0,myset_source.__len__())
"TODO: the line list(myset_source) create a disordered list, I need to put it in order"
CosmicID_labels = list(myset_source)
CosmicID_labels.sort()
dict_CosmicID2Label = dict(zip(CosmicID_labels,myLabels))
df_source_sort = df_source.sort_values(by='COSMIC_ID')
CosmicID_source = df_source_sort['COSMIC_ID'].values
idx_S = [dict_CosmicID2Label[CosID] for CosID in CosmicID_source]

"Here we extract the source domain inputs xS so that they coincide with yS sorted by COSMIC_ID"
xS_train = scaler.transform(df_source_sort[Names_features_NonZeroStd])

"Here we extract the source domain outputs yS"

yS_train = np.clip(df_source_sort["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
print(yS_train.shape)
for i in range(2, Dnorm_cell+1):
    yS_train = np.concatenate((yS_train, np.clip(df_source_sort["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

print("Ytrain size: ", yS_train.shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Make all variable passed to the model tensor to operate in pytorch"
xT_train = torch.from_numpy(xT_train)
yT_train = torch.from_numpy(yT_train)
xS_train = torch.from_numpy(xS_train)
yS_train = torch.from_numpy(yS_train)
xT_test = torch.from_numpy(xT_test)
yT_test = torch.from_numpy(yT_test)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"This is the number of source domains plus target domain."
"In this cancer application we are refering to each domain as each cell-line with the Cosmic_ID"
"There are 54 cell-lines or CosmicIDs in the Source (Melanoma) + 1 cell-line of the Target"
"Then the total NDomains = 55"
NDomains = 55
DrugC = list(np.linspace(0.142857,1.0,7))
assert DrugC.__len__() == yS_train.shape[1] and DrugC.__len__() == yT_train.shape[1]
model = TLMOGaussianProcess(xT_train,yT_train,xS_train,yS_train,idxS=idx_S,DrugC=DrugC,NDomains=NDomains)
# model.covariance.length=0.05
torch.manual_seed(35)   #Ex1: 15 (run 100 iter)  #Exp2 (906826): 35  (run 100 iter)
with torch.no_grad():
    #model.lik_std_noise= torch.nn.Parameter(0.5*torch.ones(NDomains)) #torch.nn.Parameter(0.5*torch.randn(NDomains))
    model.lik_std_noise = torch.nn.Parameter(1.0*torch.randn(NDomains))
    model.TLCovariance[0].length = 2*np.sqrt(xT_train.shape[1])*torch.rand(NDomains)[:,None]
    model.TLCovariance[1].length = 6*np.sqrt(xT_train.shape[1]) * torch.rand(NDomains)[:, None]
    model.TLCovariance[2].length = 10*np.sqrt(xT_train.shape[1]) * torch.rand(NDomains)[:, None]
    model.CoregCovariance[0].lengthscale = torch.rand(1) #1*
    model.CoregCovariance[1].lengthscale = torch.rand(1)  #1*
    model.CoregCovariance[2].lengthscale = torch.rand(1)  #1*
    model.LambdaDiDj.muDi = torch.rand(NDomains)[:, None]
    model.LambdaDiDj.bDi = torch.rand(NDomains)[:, None]
    print(model.LambdaDiDj.muDi)
print(f"Noises std: {model.lik_std_noise}")

"Training process below"
myLr = 3e-2
Niter = 200
optimizer = optim.Adam(model.parameters(),lr=myLr)
loss_fn = LogMarginalLikelihood()

for iter in range(Niter):
    # Forward pass
    mu, L = model(xT_train)

    # Backprop
    loss = -loss_fn(mu,L,yT_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter==70:  #70
        optimizer.param_groups[0]['lr']=1e-3
    #print(model.TLCovariance.length)
    print(f"i: {iter+1}, Loss: {loss.item()}")

"Here we have to assign the flag to change from self.Train_mode = True to False"
print("check difference between model.eval and model.train")
model.eval()
model.Train_mode = False
plot_test = False
if plot_test:
    x_test = xT_test.clone()
    Name_DrugID_plot = Name_DrugID_test
    plotname = 'Test'
else:
    x_test = xT_train.clone()
    Name_DrugID_plot = Name_DrugID_train
    plotname = 'Train'

Oversample_N = 3
DrugCtoPred = list(np.linspace(0.142857,1,7+6*(Oversample_N-1)))
#DrugCtoPred = list(np.linspace(0.142857,1,28))
sel_concentr = 0
with torch.no_grad():
    #mpred1,Cpred1 = model(x)
    mpred, Cpred = model(x_test,DrugC_new = DrugCtoPred,noiseless=True)

yT_pred = mpred.reshape(DrugCtoPred.__len__(),x_test.shape[0]).T
from torch.distributions.multivariate_normal import MultivariateNormal
plt.close('all')
for i in range(x_test.shape[0]):
    plt.figure(i+1)
    plt.ylim([-0.02,1.05])
    #plt.plot(x,mpred1,'.')
    plt.plot(DrugCtoPred,yT_pred[i,:],'blue')
    if plot_test:
        plt.plot(DrugC, yT_test[i,:], 'ro')
    else:
        plt.plot(DrugC, yT_train[i, :], 'ro')

    plt.title(f"CosmicID: {CosmicID_target}, {plotname} DrugID: {Name_DrugID_plot[i]}",fontsize=14)
    plt.xlabel('Dose concentration',fontsize=14)
    plt.ylabel('Cell viability',fontsize=14)
    plt.grid(True)

    for j in range(30):
        i_sample = MultivariateNormal(loc=mpred[:, 0], covariance_matrix=Cpred)
        yT_pred_sample = i_sample.sample().reshape(DrugCtoPred.__len__(), x_test.shape[0]).T
        plt.plot(DrugCtoPred, yT_pred_sample[i, :], alpha=0.1)

if plot_test:
    Test_MSE = torch.mean((yT_test-yT_pred[:,::Oversample_N])**2)
    print(f"Test_MSE: {Test_MSE}")
else:
    Train_MSE = torch.mean((yT_train - yT_pred[:, ::Oversample_N]) ** 2)
    print(f"Train_MSE: {Train_MSE}")