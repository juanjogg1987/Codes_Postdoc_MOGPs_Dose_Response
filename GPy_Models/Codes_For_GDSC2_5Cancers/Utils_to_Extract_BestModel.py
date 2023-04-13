import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
import os

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def model_pred_test(path_to_read_bash,path_to_model):
    df_bash_data = pd.read_csv(path_to_read_bash, header=None, skiprows=6)
    params_vect = df_bash_data[0].values[0][121:].split(" ")
    for k, op in enumerate(params_vect):
        if op == '-i':
            N_iter_epoch = int(params_vect[k + 1])
        if op == '-r':  # (r)and seed
            which_seed = int(params_vect[k + 1])
        if op == '-k':  # ran(k)
            rank = int(params_vect[k + 1])
        if op == '-s':  # (r)and seed
            scale = float(params_vect[k + 1])
        if op == '-p':  # (p)ython bash
            bash = int(params_vect[k + 1])
        if op == '-w':
            weight = float(params_vect[k + 1])
        if op == '-c':
            N_CellLines = int(params_vect[k + 1])
        if op == '-a':
            sel_cancer = int(params_vect[k + 1])
        if op == '-n':
            seed_for_N = int(params_vect[k + 1])
        if op == '-t':
            N_5thCancer_ToBe_Included = int(params_vect[k + 1])

    _FOLDER = "/home/ac1jjgg/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
    #_FOLDER = "/home/juanjo/Work_Postdoc/my_codes_postdoc/Dataset_5Cancers/GDSC2_EGFR_PI3K_MAPK_Top5cancers/"
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



    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #config = commandLine()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dict_cancers={0:'GDSC2_EGFR_PI3K_MAPK_Breast_1000FR.csv',1:'GDSC2_EGFR_PI3K_MAPK_COAD_1000FR.csv',
                  2:'GDSC2_EGFR_PI3K_MAPK_LUAD.csv',3:'GDSC2_EGFR_PI3K_MAPK_melanoma.csv',4:'GDSC2_EGFR_PI3K_MAPK_SCLC.csv'}

    indx_cancer = np.array([0,1,2,3,4])
    indx_cancer_test = np.array([int(sel_cancer)])
    indx_cancer_train = np.delete(indx_cancer,indx_cancer_test)

    name_feat_file = "GDSC2_EGFR_PI3K_MAPK_allfeatures.csv"
    name_feat_file_nozero = "GDSC2_EGFR_PI3K_MAPK_features_NonZero_Drugs1036_1061_1373.csv" #"GDSC2_EGFR_PI3K_MAPK_features_NonZero.csv"
    Num_drugs = 3
    N_CellLines_perDrug = int(N_CellLines)//Num_drugs
    N_CellLines = int(N_CellLines)
    rand_state_N = int(seed_for_N)
    for i in range(0,4):
        name_for_KLrelevance = dict_cancers[indx_cancer_train[i]]
        print(name_for_KLrelevance)
        if i==0:
            df_to_read = pd.read_csv(_FOLDER + name_for_KLrelevance)#.sample(n=N_CellLines,random_state = rand_state_N)
            df_4Cancers_train_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna()#.sample(n=N_CellLines,random_state = rand_state_N)
            df_4Cancers_train_d2 = df_to_read[df_to_read["DRUG_ID"]==1061].dropna()#.sample(n=N_CellLines,random_state = rand_state_N)
            df_4Cancers_train_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna()#.sample(n=N_CellLines,random_state=rand_state_N)
            df_4Cancers_train_Only_d1 = df_4Cancers_train_d1.copy()
            df_4Cancers_train_Only_d2 = df_4Cancers_train_d2.copy()
            df_4Cancers_train_Only_d3 = df_4Cancers_train_d3.copy()
        else:
            df_to_read = pd.read_csv(_FOLDER + name_for_KLrelevance)#.sample(n=N_CellLines,random_state = rand_state_N)
            df_4Cancers_train_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna()#.sample(n=N_CellLines,random_state = rand_state_N)
            df_4Cancers_train_d2 = df_to_read[df_to_read["DRUG_ID"] == 1061].dropna()#.sample(n=N_CellLines,random_state=rand_state_N)
            df_4Cancers_train_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna()#.sample(n=N_CellLines,random_state=rand_state_N)
            #N_per_drug = [df_4Cancers_train_d1.shape[0], df_4Cancers_train_d2.shape[0], df_4Cancers_train_d3.shape[0]]
            #Nd1, Nd2, Nd3 = np.clip(N_CellLines_perDrug, 1, N_per_drug[0]), np.clip(N_CellLines_perDrug, 1, N_per_drug[1]), np.clip(N_CellLines_perDrug, 1, N_per_drug[2])
            #df_4Cancers_train = pd.concat([df_4Cancers_train,df_4Cancers_train_d1.sample(n=Nd1,random_state = rand_state_N), df_4Cancers_train_d2.sample(n=Nd2,random_state = rand_state_N),df_4Cancers_train_d3.sample(n=Nd3,random_state = rand_state_N)])
            df_4Cancers_train_Only_d1 = pd.concat([df_4Cancers_train_Only_d1,df_4Cancers_train_d1])
            df_4Cancers_train_Only_d2 = pd.concat([df_4Cancers_train_Only_d2, df_4Cancers_train_d2])
            df_4Cancers_train_Only_d3 = pd.concat([df_4Cancers_train_Only_d3, df_4Cancers_train_d3])
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Here we extract the test dataframe for the one cancer left out"
    Name_cancer_test = dict_cancers[indx_cancer_test[0]]
    df_to_read = pd.read_csv(_FOLDER + Name_cancer_test)
    df_4Cancers_test_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna()
    df_4Cancers_test_d2 = df_to_read[df_to_read["DRUG_ID"]==1061].dropna()
    df_4Cancers_test_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna()
    df_4Cancers_test = pd.concat([df_4Cancers_test_d1, df_4Cancers_test_d2,df_4Cancers_test_d3])

    "This section of code is to allow including few observations from the 5th Cancer as part of the training data"
    #a_bit_of_5thCancer = True

    if int(N_5thCancer_ToBe_Included)!=0:
        N_ToInclude_per_Drug = int(N_5thCancer_ToBe_Included)//Num_drugs
        N_ToInclude_per_Drug = np.clip(N_ToInclude_per_Drug, 1, df_4Cancers_test.shape[0] // 3 - 10)
        df_4Cancers_test_d1 = df_to_read[df_to_read["DRUG_ID"]==1036].dropna().reset_index().drop(columns='index')
        df_4Cancers_test_d2 = df_to_read[df_to_read["DRUG_ID"]==1061].dropna().reset_index().drop(columns='index')
        df_4Cancers_test_d3 = df_to_read[df_to_read["DRUG_ID"] == 1373].dropna().reset_index().drop(columns='index')
        N_per_drug = [df_4Cancers_test_d1.shape[0], df_4Cancers_test_d2.shape[0], df_4Cancers_test_d3.shape[0]]
        #indx_test_to_include = [,np.random.permutation(np.arange(0,N_per_drug[1])),np.random.permutation(np.arange(0,N_per_drug[2]))]
        "The seed below is to guarantee always having the same values of 5th cancer to be included in Training regardless"
        "of all the different cross-validations, we do not want them to change at each different cross-validation of MOGP"
        if int(N_5thCancer_ToBe_Included) == 9:
            if indx_cancer_test == 0:
                np.random.seed(6)   #There are 3 IC50s, we put the 9 observations (1 IC50s and 8 non-IC50)
            elif indx_cancer_test == 1:
                np.random.seed(1)   #There are 8 IC50s, we put the 9 observations (3 IC50s and 6 non-IC50)
            elif indx_cancer_test == 2:
                np.random.seed(1)  #There is only 1 IC50s for this Cancer almost all data are non-responsive (Test with no IC50)
            elif indx_cancer_test == 3:
                np.random.seed(1)  #There are 63 IC50s and 68 Non-IC50, we put the 9 observations (5 IC50s and 4 non-IC50)
            elif indx_cancer_test == 4:
                np.random.seed(1)    #There are not IC50s for this Cancer all data are non-responsive
        elif int(N_5thCancer_ToBe_Included) == 45:
            if indx_cancer_test == 0:
                np.random.seed(6)   #There are 3 IC50s, we put the 45 observations (1 IC50s and 44 non-IC50)
            elif indx_cancer_test == 1:
                np.random.seed(3)   #There are 8 IC50s, we put the 45 observations (4 IC50s and 40 non-IC50)
            elif indx_cancer_test == 2:
                np.random.seed(1)  #There is only 1 IC50s for this Cancer almost all data are non-responsive (Test with no IC50)
            elif indx_cancer_test == 3:
                np.random.seed(1)  #There are 63 IC50s and 68 Non-IC50, we put the 45 observations (19 IC50s and 26 non-IC50)
            elif indx_cancer_test == 4:
                np.random.seed(1)    #There are not IC50s for this Cancer all data are non-responsive
        else:
            print("SELECTING THE IC50s RANDOMLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        Test_drugs_indexes = [np.random.permutation(np.arange(0, N_per_drug[myind])) for myind in range(Num_drugs)]
        indx_test_to_NotInclude = [np.delete(Test_drugs_indexes[myind],np.arange(0,N_ToInclude_per_Drug)) for myind in range(Num_drugs)]
        indx_test_to_include = [Test_drugs_indexes[myind][np.arange(0,N_ToInclude_per_Drug)] for myind in range(Num_drugs)]
        print("Indexes to Include:",indx_test_to_include)
        df_4Cancers_test = pd.concat([df_4Cancers_test_d1.iloc[indx_test_to_NotInclude[0]], df_4Cancers_test_d2.iloc[indx_test_to_NotInclude[1]],df_4Cancers_test_d3.iloc[indx_test_to_NotInclude[2]]])
        df_4Cancers_test_ToInclude = pd.concat([df_4Cancers_test_d1.iloc[indx_test_to_include[0]], df_4Cancers_test_d2.iloc[indx_test_to_include[1]],df_4Cancers_test_d3.iloc[indx_test_to_include[2]]])
        df_4Cancers_test_Only_d1 = df_4Cancers_test_d1.iloc[indx_test_to_include[0]].copy()
        df_4Cancers_test_Only_d2 = df_4Cancers_test_d2.iloc[indx_test_to_include[1]].copy()
        df_4Cancers_test_Only_d3 = df_4Cancers_test_d3.iloc[indx_test_to_include[2]].copy()

    drug1_train = df_4Cancers_train_Only_d1[df_4Cancers_train_Only_d1.columns[25:]].values
    drug1_test = df_4Cancers_test_Only_d1[df_4Cancers_test_Only_d1.columns[25:]].values
    drug2_train = df_4Cancers_train_Only_d2[df_4Cancers_train_Only_d2.columns[25:]].values
    drug2_test = df_4Cancers_test_Only_d2[df_4Cancers_test_Only_d2.columns[25:]].values
    drug3_train = df_4Cancers_train_Only_d3[df_4Cancers_train_Only_d3.columns[25:]].values
    drug3_test = df_4Cancers_test_Only_d3[df_4Cancers_test_Only_d3.columns[25:]].values

    from scipy.spatial.distance import minkowski

    "Be careful that metric distance is a DISIMILARITY not SIMILARITY"
    def Disimilarity_Metric(drug_train,drug_test):
        M_dist = np.zeros((drug_train.shape[0], drug_test.shape[0]))
        for i in range(drug_train.shape[0]):
            for j in range(drug_test.shape[0]):
                M_dist[i,j] = minkowski(drug_train[i,:],drug_test[j,:],1)
        return  M_dist

    def My_Categorical(prob,Nsamples):
        prob = prob.flatten()
        cum_prob = np.cumsum(prob, axis=-1)
        myreap = np.repeat(cum_prob[None, :], Nsamples, axis=0)
        r = np.random.uniform(0,1,Nsamples)[:,None]
        sample = np.argmax(myreap>r,axis=-1) #My_Categorical(prob)
        return sample

    M_dist_d1 = Disimilarity_Metric(drug1_train,drug1_test).sum(1)
    M_dist_d2 = Disimilarity_Metric(drug2_train,drug2_test).sum(1)
    M_dist_d3 = Disimilarity_Metric(drug3_train,drug3_test).sum(1)

    N_All_Cell = N_CellLines*4   #here it is 4 due to having 4 cancers to cross-val
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "To DO: Create function for each Vec_N_index per drug"
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    M_dist_d1_norm = M_dist_d1.max()/M_dist_d1
    M_dist_d2_norm = M_dist_d2.max()/M_dist_d2
    M_dist_d3_norm = M_dist_d3.max()/M_dist_d3

    prob_drugs = [M_dist_d1_norm/M_dist_d1_norm.sum(0),M_dist_d2_norm/M_dist_d2_norm.sum(0),M_dist_d3_norm/M_dist_d3_norm.sum(0)]
    "Here I just set the different seed to sample from My_Categorical"
    np.random.seed(rand_state_N)
    ind_d1 = My_Categorical(prob_drugs[0],N_All_Cell//Num_drugs)
    ind_d2 = My_Categorical(prob_drugs[1],N_All_Cell//Num_drugs)
    ind_d3 = My_Categorical(prob_drugs[2],N_All_Cell//Num_drugs)

    df_4Cancers_train_Only_d1 = df_4Cancers_train_Only_d1.reset_index().drop(columns='index')
    df_4Cancers_train_Only_d2 = df_4Cancers_train_Only_d2.reset_index().drop(columns='index')
    df_4Cancers_train_Only_d3 = df_4Cancers_train_Only_d3.reset_index().drop(columns='index')

    df_4Cancers_train = pd.concat([df_4Cancers_train_Only_d1.iloc[ind_d1],df_4Cancers_train_Only_d2.iloc[ind_d2],df_4Cancers_train_Only_d3.iloc[ind_d3]])
    df_4Cancers_train = pd.concat([df_4Cancers_train, df_4Cancers_test_ToInclude])

    # Here we just check that from the column index 25 the input features start
    start_pos_features = 25
    print(df_4Cancers_train.columns[start_pos_features])

    df_feat_Names = pd.read_csv(_FOLDER + name_feat_file)  # Contain Feature Names
    df_feat_Names_nozero = pd.read_csv(_FOLDER + name_feat_file_nozero)  # Contain Feature Names
    indx_nozero = df_feat_Names_nozero['index'].values[start_pos_features:]

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    scaler = MinMaxScaler().fit(df_4Cancers_train[df_4Cancers_train.columns[indx_nozero]])
    X_train_features = scaler.transform(df_4Cancers_train[df_4Cancers_train.columns[indx_nozero]])
    X_test_features = scaler.transform(df_4Cancers_test[df_4Cancers_test.columns[indx_nozero]])

    "Below we select just 7 concentration since GDSC2 only has such a number"
    y_train_drug = np.clip(df_4Cancers_train["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
    y_test_drug = np.clip(df_4Cancers_test["norm_cells_" + str(1)].values[:, None], 1.0e-9, np.inf)
    print(y_train_drug.shape)
    for i in range(2, 8):
        y_train_drug = np.concatenate((y_train_drug, np.clip(df_4Cancers_train["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)
        y_test_drug = np.concatenate((y_test_drug, np.clip(df_4Cancers_test["norm_cells_" + str(i)].values[:, None], 1.0e-9, np.inf)), 1)

    print("Ytrain size: ", y_train_drug.shape)
    print("Ytest size: ", y_test_drug.shape)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    params_4_sig_train = df_4Cancers_train["param_" + str(1)].values[:, None]
    params_4_sig_test = df_4Cancers_test["param_" + str(1)].values[:, None]
    for i in range(2, 5):
        params_4_sig_train = np.concatenate((params_4_sig_train, df_4Cancers_train["param_" + str(i)].values[:, None]), 1)
        params_4_sig_test = np.concatenate((params_4_sig_test, df_4Cancers_test["param_" + str(i)].values[:, None]), 1)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    import matplotlib.pyplot as plt
    from sklearn import metrics

    plt.close('all')
    #0.142857143
    "Be careful that x starts from 0.111111 for 9 drugs,"
    "but x starts from 0.142857143 for the case of 7 drugs"
    x_lin = np.linspace(0.142857143, 1, 1000)
    x_real_dose = np.linspace(0.142857143, 1, 7)  #Here is 7 due to using GDSC2 that has 7 doses
    def Get_IC50_AUC_Emax(params_4_sig_train,x_lin,x_real_dose):
        x_lin_tile = np.tile(x_lin, (params_4_sig_train.shape[0], 1))
        # (x_lin,params_4_sig_train.shape[0],1).shape
        Ydose_res = []
        AUC = []
        IC50 = []
        Ydose50 = []
        Emax = []
        for i in range(params_4_sig_train.shape[0]):
            Ydose_res.append(sigmoid_4_param(x_lin_tile[i, :], *params_4_sig_train[i, :]))
            AUC.append(metrics.auc(x_lin_tile[i, :], Ydose_res[i]))
            Emax.append(Ydose_res[i][-1])
            res1 = (Ydose_res[i] < 0.507)
            res2 = (Ydose_res[i] > 0.493)
            if (res1 & res2).sum() > 0:
                Ydose50.append(Ydose_res[i][res1 & res2].mean())
                IC50.append(x_lin[res1 & res2].mean())
            elif Ydose_res[i][-1]<0.5:
               for dose_j in range(x_lin.shape[0]):
                   if(Ydose_res[i][dose_j] < 0.5):
                       break
               Ydose50.append(Ydose_res[i][dose_j])
               aux_IC50 = x_lin[dose_j]  #it has to be a float not an array to avoid bug
               IC50.append(aux_IC50)
            else:
                Ydose50.append(0.5)
                IC50.append(1.5) #IC50.append(x_lin[-1])

        return Ydose50,Ydose_res,IC50,AUC,Emax

    Ydose50,Ydose_res,IC50,AUC,Emax = Get_IC50_AUC_Emax(params_4_sig_train,x_lin,x_real_dose)
    Ydose50_test,Ydose_res_test,IC50_test,AUC_test,Emax_test = Get_IC50_AUC_Emax(params_4_sig_test,x_lin,x_real_dose)

    def my_plot(posy,fig_num,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug):
        plt.figure(fig_num)
        plt.plot(x_lin, Ydose_res[posy])
        plt.plot(x_real_dose, y_train_drug[posy, :], '.')
        plt.plot(IC50[posy], Ydose50[posy], 'rx')
        plt.plot(x_lin, np.ones_like(x_lin)*Emax[posy], 'r') #Plot a horizontal line as Emax
        plt.title(f"AUC = {AUC[posy]}")
        print(AUC[posy])

    #posy = 18
    #my_plot(posy,0,Ydose50,Ydose_res,IC50,AUC,Emax,x_lin,x_real_dose,y_train_drug)
    #my_plot(posy,1,Ydose50_test,Ydose_res_test,IC50_test,AUC_test,Emax_test,x_lin,x_real_dose,y_test_drug)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    AUC = np.array(AUC)
    IC50 = np.array(IC50)
    Emax = np.array(Emax)
    AUC_test = np.array(AUC_test)
    IC50_test = np.array(IC50_test)
    Emax_test = np.array(Emax_test)

    "Below we select just the columns with std higher than zero"
    Name_Features_5Cancers = df_feat_Names_nozero['feature'].values[start_pos_features:]
    Xall = X_train_features.copy()
    Yall = y_train_drug.copy()

    AUC_all = AUC[:, None].copy()
    IC50_all = IC50[:, None].copy()
    Emax_all = Emax[:, None].copy()

    AUC_test = AUC_test[:, None].copy()
    IC50_test = IC50_test[:, None].copy()
    Emax_test = Emax_test[:, None].copy()

    print("AUC train size:", AUC_all.shape)
    print("IC50 train size:", IC50_all.shape)
    print("Emax train size:", Emax_all.shape)
    print("X all train data size:", Xall.shape)
    print("Y all train data size:", Yall.shape)

    print("AUC test size:", AUC_test.shape)
    print("IC50 test size:", IC50_test.shape)
    print("Emax test size:", Emax_test.shape)
    print("X all test data size:", X_test_features.shape)
    print("Y all test data size:", y_test_drug.shape)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Create a K-fold for cross-validation"
    from sklearn.model_selection import KFold, cross_val_score
    Ndata = Xall.shape[0]
    Xind = np.arange(Ndata)
    nsplits = 5 #Ndata
    k_fold = KFold(n_splits=nsplits, shuffle=True, random_state=1)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Ntasks = 7

    Nfold = nsplits
    print(f"Train ovell all Data")
    Xval_aux = X_test_features.copy() #Xall[test_ind].copy()
    Ylabel_val = np.array([i * np.ones(Xval_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]

    Xval = np.concatenate((np.tile(Xval_aux, (Ntasks, 1)), Ylabel_val), 1)

    Xtrain_aux = Xall.copy()
    Ylabel_train = np.array([i * np.ones(Xtrain_aux.shape[0]) for i in range(Ntasks)]).flatten()[:, None]
    Xtrain = np.concatenate((np.tile(Xtrain_aux, (Ntasks, 1)), Ylabel_train), 1)

    Yval = y_test_drug.T.flatten().copy()[:,None]
    Ytrain = Yall.T.flatten().copy()[:,None]

    Emax_val = Emax_test.copy()
    AUC_val = AUC_test.copy()
    IC50_val = IC50_test.copy()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    import GPy
    from matplotlib import pyplot as plt

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    rank = int(rank)  # Rank for the MultitaskKernel
    "Below we substract one due to being the label associated to the output"
    Dim = Xtrain.shape[1]-1

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    myseed = int(which_seed)
    np.random.seed(myseed)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Product Kernels"
    "Below we use the locations:"
    "0:279 for Mutation"
    "279:697 for PANCAN"
    "697:768 for COPY-Number"
    "768:end for Drugs compounds"
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    split_dim = 4
    AddKern_loc = [279, 697,768,Dim]
    mykern = GPy.kern.RBF(AddKern_loc[0], active_dims=list(np.arange(0, AddKern_loc[0])))
    print(list(np.arange(0, AddKern_loc[0])))
    for i in range(1, split_dim):
        mykern = mykern * GPy.kern.RBF(AddKern_loc[i]-AddKern_loc[i-1],active_dims=list(np.arange(AddKern_loc[i-1], AddKern_loc[i])))
        print(list(np.arange(AddKern_loc[i-1], AddKern_loc[i])))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    mykern.rbf.lengthscale = float(scale)* np.sqrt(Dim) * np.random.rand()
    mykern.rbf.variance.fix()
    for i in range(1,split_dim):
        eval("mykern.rbf_"+str(i)+".lengthscale.setfield(float(scale)* np.sqrt(Dim) * np.random.rand(), np.float64)")
        eval("mykern.rbf_" + str(i) + ".variance.fix()")

    kern = mykern ** GPy.kern.Coregionalize(1, output_dim=Ntasks,rank=rank)
    model = GPy.models.GPRegression(Xtrain, Ytrain, kern)
    Init_Ws = float(weight) * np.random.randn(Ntasks,rank)
    model.kern.coregion.W = Init_Ws
    #model.optimize(optimizer='lbfgsb',messages=True,max_iters=5)
    #model.optimize(max_iters=int(config.N_iter_epoch))
    #model.optimize()

    #model[:] = np.load('/home/juanjo/Work_Postdoc/my_codes_postdoc/GPy_Models/Codes_For_GDSC2_5Cancers/Test_Data_ToPlot_GDSC2_5Cancers/Three_drugs/m_924.npy')
    model[:] = np.load(path_to_model)

    m_pred, v_pred = model.predict(Xval, full_cov=False)
    plt.figure(Nfold+1)
    plt.plot(Yval, 'bx')
    plt.plot(m_pred, 'ro')
    plt.plot(m_pred + 2 * np.sqrt(v_pred), '--m')
    plt.plot(m_pred - 2 * np.sqrt(v_pred), '--m')

    Yval_curve = Yval.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    m_pred_curve = m_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()
    v_pred_curve = v_pred.reshape(Ntasks, Xval_aux.shape[0]).T.copy()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    "Negative Log Predictive Density (NLPD)"
    Val_NMLL = -np.mean(model.log_predictive_density(Xval,Yval))
    print("NegLPD Val", Val_NMLL)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    from scipy.interpolate import interp1d
    from scipy.interpolate import pchip_interpolate

    "Be careful that x starts from 0.111111 for 9 drugs,"
    "but x starts from 0.142857143 for the case of 7 drugs"
    x_dose = np.linspace(0.142857143, 1.0, 7)
    x_dose_new = np.linspace(0.142857143, 1.0, 1000)
    Ydose50_pred = []
    IC50_pred = []
    AUC_pred = []
    Emax_pred = []
    Y_pred_interp_all = []
    std_upper_interp_all = []
    std_lower_interp_all = []
    for i in range(Yval_curve.shape[0]):
        y_resp = m_pred_curve[i, :].copy()
        std_upper = y_resp + 2*np.sqrt(v_pred_curve[i, :])
        std_lower = y_resp - 2 * np.sqrt(v_pred_curve[i, :])
        f = interp1d(x_dose, y_resp)
        #f2 = interp1d(x_dose, y_resp, kind='cubic')
        y_resp_interp = pchip_interpolate(x_dose, y_resp, x_dose_new)
        std_upper_interp = pchip_interpolate(x_dose, std_upper, x_dose_new)
        std_lower_interp = pchip_interpolate(x_dose, std_lower, x_dose_new)

        #y_resp_interp = f2(x_dose_new)
        Y_pred_interp_all.append(y_resp_interp)
        std_upper_interp_all.append(std_upper_interp)
        std_lower_interp_all.append(std_lower_interp)
        AUC_pred.append(metrics.auc(x_dose_new, y_resp_interp))
        Emax_pred.append(y_resp_interp[-1])

        res1 = y_resp_interp < 0.507
        res2 = y_resp_interp > 0.493
        res_aux = np.where(res1 & res2)[0]
        if (res1 & res2).sum()>0:
            res_IC50 = np.arange(res_aux[0],res_aux[0]+ res_aux.shape[0])==res_aux
            res_aux = res_aux[res_IC50].copy()
        else:
            res_aux = res1 & res2

        if (res1 & res2).sum() > 0:
            Ydose50_pred.append(y_resp_interp[res_aux].mean())
            IC50_pred.append(x_dose_new[res_aux].mean())
        elif y_resp_interp[-1] < 0.5:
            for dose_j in range(x_dose_new.shape[0]):
                if (y_resp_interp[dose_j] < 0.5):
                    break
            Ydose50_pred.append(y_resp_interp[dose_j])
            aux_IC50 = x_dose_new[dose_j]  # it has to be a float not an array to avoid bug
            IC50_pred.append(aux_IC50)
        else:
            Ydose50_pred.append(0.5)
            IC50_pred.append(1.5)

    Ydose50_pred = np.array(Ydose50_pred)
    IC50_pred = np.array(IC50_pred)[:,None]
    AUC_pred = np.array(AUC_pred)[:, None]
    Emax_pred = np.array(Emax_pred)[:, None]

    posy = 1
    plt.figure(Nfold+nsplits+2)
    plt.plot(x_dose_new, Y_pred_interp_all[posy])
    plt.plot(x_dose_new, std_upper_interp_all[posy],'b--')
    plt.plot(x_dose_new, std_lower_interp_all[posy], 'b--')
    plt.plot(x_dose, Yval_curve[posy, :], '.')
    plt.plot(IC50_pred[posy], Ydose50_pred[posy], 'rx')
    plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_pred[posy], 'r')  # Plot a horizontal line as Emax
    plt.plot(x_dose_new, np.ones_like(x_dose_new) * Emax_val[posy], 'r')  # Plot a horizontal line as Emax
    plt.title(f"AUC = {AUC_pred[posy]}")
    print(AUC_pred[posy])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Emax_abs = np.mean(np.abs(Emax_val - Emax_pred))
    AUC_abs = np.mean(np.abs(AUC_val - AUC_pred))
    IC50_MSE = np.mean((IC50_val - IC50_pred) ** 2)
    IC50_abs = np.mean(np.abs(IC50_val - IC50_pred))
    MSE_curves = np.mean((m_pred_curve - Yval_curve) ** 2, 1)
    AllPred_MSE = np.mean((m_pred_curve - Yval_curve) ** 2)
    per_Dose_MAE = np.mean(np.abs(m_pred_curve - Yval_curve), 0)
    curve_MAE = np.mean(np.abs(m_pred_curve - Yval_curve))
    print("IC50 MSE:", IC50_MSE)
    print("AUC MAE:", AUC_abs)
    print("Emax MAE:", Emax_abs)
    Med_MSE = np.median(MSE_curves)
    Mean_MSE = np.mean(MSE_curves)
    print("Med_MSE:", Med_MSE)
    print("Mean_MSE:", Mean_MSE)
    print("All Predictions MSE:", AllPred_MSE)
    print("MAE per Dose:",per_Dose_MAE)
    print("MAE all curves:",curve_MAE)

    from scipy.stats import spearmanr

    pos_Actual_IC50 = IC50_val != 1.5
    pos_No_IC50 = IC50_val == 1.5

    IC50_Resp_MAE = np.mean(np.abs(IC50_val[pos_Actual_IC50]- IC50_pred[pos_Actual_IC50]))
    IC50_NoResp_MAE = np.mean(np.abs(IC50_val[pos_No_IC50] - IC50_pred[pos_No_IC50]))

    IC50_Resp_r2 = r2_score(IC50_val[pos_Actual_IC50],IC50_pred[pos_Actual_IC50])
    IC50_NoResp_r2 = r2_score(IC50_val[pos_No_IC50],IC50_pred[pos_No_IC50])

    print("MAE IC50 Reponsive:",IC50_Resp_MAE)
    print("MAE IC50 Non-Reponsive:", IC50_NoResp_MAE)
    print("R2 IC50 Reponsive:", IC50_Resp_r2)
    print("R2 IC50 Non-Reponsive:", IC50_NoResp_r2)

    print("Yval shape",Yval.shape)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    return AUC_abs, Emax_abs, IC50_abs, IC50_Resp_MAE, IC50_NoResp_MAE, IC50_Resp_r2,IC50_NoResp_r2, curve_MAE,per_Dose_MAE