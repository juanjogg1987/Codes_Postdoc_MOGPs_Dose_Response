
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Select M: Mutation, Gene Copy Number (GCMP: CellModelPassport or GDM: DepMap)  and Expression (ERMA or EDM: DepMap),
#Also, select
Which_data_output = "NP24"
Which_data_type1 = "EDM"
Which_data_type2 = "GDM"

if Which_data_output == "NP24":
    data_drugs_NP24 = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Drug Screen Datasets/NP24_Drug_Screen.csv")

    "We will use mainly the DepMap_ID to identify the cell lines due to be more consistent with not many NaN data"
    data_drugs_NP24 = data_drugs_NP24[~data_drugs_NP24["DepMap_ID"].isna()] #Here we get rid of the DepMap_ID rows with NaN or Missing ID
    data_drugs_NP24 = data_drugs_NP24.drop(['COSMICID'], axis=1).reset_index()

    types_of_cancer = ["BREAST","LUNG","SKIN","OVARY","ENDOMETRIUM","BONE","HAEMATOPOIETIC",
                       "SOFT_TISSUE","PANCREAS","STOMACH","CENTRAL_NERVOUS_SYSTEM","PLEURA",
                       "LARGE_INTESTINE","OESOPHAGUS","PROSTATE","LIVER","URINARY_TRACT",
                       "AUTONOMIC_GANGLIA","THYROID","KIDNEY","UPPER_AERODIGESTIVE_TRACT",
                       "SALIVARY_GLAND","BILIARY_TRACT"]

    dic_index_cancer = {"BREAST":[], "LUNG":[], "SKIN":[], "OVARY":[], "ENDOMETRIUM":[], "BONE":[], "HAEMATOPOIETIC":[],
                       "SOFT_TISSUE":[], "PANCREAS":[], "STOMACH":[], "CENTRAL_NERVOUS_SYSTEM":[], "PLEURA":[],
                       "LARGE_INTESTINE":[], "OESOPHAGUS":[], "PROSTATE":[], "LIVER":[], "URINARY_TRACT":[],
                       "AUTONOMIC_GANGLIA":[], "THYROID":[], "KIDNEY":[], "UPPER_AERODIGESTIVE_TRACT":[],
                       "SALIVARY_GLAND":[], "BILIARY_TRACT":[],"UNKNOWN":[]}

    #index_cancer =[-1]*types_of_cancer.__len__()
    for i in range(data_drugs_NP24.shape[0]):
        flag_in = 0
        for count,typecancer in enumerate(types_of_cancer):
            if(typecancer in data_drugs_NP24["CCLE_Name"][i]):
                #print(i)
                flag_in = 1
                dic_index_cancer[typecancer].append(i)
        if flag_in == 0:
            print(f"In Pos {i} a new name:", data_drugs_NP24["CCLE_Name"][i])
            dic_index_cancer["UNKNOWN"].append(i)
    count_check = 0
    for i in range(types_of_cancer.__len__()):
        count_check = count_check + dic_index_cancer[types_of_cancer[i]].__len__()
    count_check = count_check + dic_index_cancer["UNKNOWN"].__len__()

    print(f"Count_check = {count_check} and real lenth = {data_drugs_NP24.shape}")

    # for i in range(data_drugs_NP24.shape[0]):
    #     #plt.close('all')
    #     sigmoid_curve = np.array([float(i) for i in data_drugs_NP24["Activity Data (median)"][i].split(",")])
    #     plt.figure(i)
    #     #plt.plot(1/(1+np.exp(-sigmoid_curve)))
    #     plt.plot(sigmoid_curve)

    data_drugs_NP24 = data_drugs_NP24.sort_values("DepMap_ID").reset_index()

if Which_data_type1 == "M" or Which_data_type2 == "M":
    #data_drugs_NP24 = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Drug Screen Datasets/NP24_Drug_Screen.csv")
    #data_mutation_CMP21_Dec = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/CMP21_Dec.csv")
    #data_mutation_CMP21_Nov = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/CMP21_Nov.csv")

    "The data below is in the folder DepMap"
    data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_18Q4.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_19Q1.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_19Q2.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_19Q3.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_19Q4.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_20Q1.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_20Q4.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_21Q1.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_21Q2.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_21Q3.csv")
    #data_mutation = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Mutation Datasets/DepMap/Mutation_21Q4.csv")

    count_high_nan = 0
    count_low_nan = 0
    for colum_name in data_mutation.columns:
        num_of_nan = data_mutation[colum_name].isna().sum()
        #print(f"The column_name = {colum_name} Exist")
        #print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > 0:
            count_high_nan = count_high_nan + 1
        else:
            count_low_nan = count_low_nan + 1

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

if Which_data_type1 == "GCMP" or Which_data_type2 == "CGMP":  #Gene Copy Model Passport
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/Cell Model Passport/GCN_CMP_18.csv"); NumCode = 24511
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/Cell Model Passport/GCN_CMP_GISTIC_19.csv"); NumCode = 34970
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/Cell Model Passport/GCN_CMP_PICNIC_19.csv"); NumCode = 25881
    data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN18.csv"); NumCode = 25881

    count_high_nan = 0
    count_low_nan = 0
    for i in range(1,NumCode):
        string_num = str(0)*(5-str(i).__len__()) + str(i)
        colum_name = "SIDG"+string_num
        if colum_name in data_gene.columns:
            num_of_nan = data_gene[colum_name].isna().sum()
            #print(f"The column_name = {colum_name} does EXIST EXIST")
            #print(f"With a Number of NaN = {num_of_nan}")
            if num_of_nan > 100:
                count_high_nan = count_high_nan + 1
            else:
                count_low_nan = count_low_nan + 1
        else:
            print(f"The colum_name = {colum_name} does NOT exist")
    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")
if Which_data_type1 == "GDM" or Which_data_type2 == "GDM":
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN18.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN19.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN20.csv")
    data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN21.csv")
    data_gene = data_gene[~data_gene["DepMap_ID"].isna()]  # Here we get rid of the DepMap_ID rows with NaN or Missing ID
    data_gene = data_gene.drop(['Unnamed: 0'], axis=1)
    data_gene = data_gene.drop(['COSMICID'], axis=1)

    column_names_to_delete = []
    Nrows = data_gene.shape[0]
    count_high_nan = 0
    count_low_nan = 0
    for colum_name in data_gene.columns:
        num_of_nan = data_gene[colum_name].isna().sum()
        #print(f"The column_name = {colum_name} Exist")
        #print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > int(Nrows*0.05):  #if the 5% of rows is NaN in the column we delete the column below
            count_high_nan = count_high_nan + 1
            column_names_to_delete.append(colum_name)
        else:
            count_low_nan = count_low_nan + 1

    if column_names_to_delete.__len__()>0: #Here we delete the column if more than 5% rows are NaN.
        for col_del in column_names_to_delete:
            data_gene = data_gene.drop([col_del], axis=1)
            print(f"The Column={col_del} was deleted")

    data_gene = data_gene.dropna()  #Here we drop all the rows with NaN
    data_gene = data_gene.sort_values("DepMap_ID").reset_index()

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

if Which_data_type1 == "ERMA" or Which_data_type2 == "ERMA":
    data_exp = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/RMA_proc_basalEXP.csv")

    count_high_nan = 0
    count_low_nan = 0
    for colum_name in data_exp.columns:
        num_of_nan = data_exp[colum_name].isna().sum()
        #print(f"The column_name = {colum_name} Exist")
        #print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > 0:
            count_high_nan = count_high_nan + 1
        else:
            count_low_nan = count_low_nan + 1

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

if Which_data_type1 == "EDM" or Which_data_type2 == "EDM":
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP18_TPM.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP19.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP19_Q1_TPM.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP20.csv")
    data_exp = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP21.csv")

    data_exp = data_exp[~data_exp["DepMap_ID"].isna()]  # Here we get rid of the DepMap_ID rows with NaN or Missing ID
    data_exp = data_exp.drop(['COSMICID'], axis=1)

    column_names_to_delete = []
    Nrows = data_exp.shape[0]
    count_high_nan = 0
    count_low_nan = 0
    for colum_name in data_exp.columns:
        num_of_nan = data_exp[colum_name].isna().sum()
        # print(f"The column_name = {colum_name} Exist")
        # print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > int(Nrows * 0.05):  # if the 5% of rows is NaN in the column we delete the column below
            count_high_nan = count_high_nan + 1
            column_names_to_delete.append(colum_name)
        else:
            count_low_nan = count_low_nan + 1

    if column_names_to_delete.__len__() > 0:  # Here we delete the column if more than 5% rows are NaN.
        for col_del in column_names_to_delete:
            data_exp = data_exp.drop([col_del], axis=1)
            print(f"The Column={col_del} was deleted")

    data_exp = data_exp.dropna()  # Here we drop all the rows with NaN
    data_exp = data_exp.sort_values("DepMap_ID").reset_index()

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# "This section is to concatenate the Compounds fo Drugs and Genomics Data"
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
# """Creating Additional data frame with the genomics"""
# for i in range(data_gene.shape[0]):
#     Num_duplicate = (data_gene["DepMap_ID"][i] == data_drugs_NP24["DepMap_ID"]).sum()
#     if Num_duplicate>0:
#         df_aux = data_gene[data_gene.columns[2:]][i:i+1] #Here we take the data from column 3 until the end
#         df_aux = pd.concat([df_aux]*Num_duplicate)
#         if i>0:
#             df_concat = pd.concat([df_concat, df_aux]) #Here we add the genomics
#         else:
#             df_concat = df_aux



# """Creating Additional data frame with the Compounds of Drugs"""
#
# data_chemo = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Chemoinformatic Data/NP24_Descriptors_NewDrugNames.csv")
#
# for i in range(data_chemo.shape[0]):
#     Num_duplicate = (data_chemo["cmpdname"][i] == data_drugs_NP24["Compound"]).sum()
#     if Num_duplicate>0:
#         df_aux = data_chemo[data_chemo.columns[0:]][i:i+1]
#         df_aux = pd.concat([df_aux]*Num_duplicate)
#         if i>0:
#             df_concat = pd.concat([df_concat, df_aux]) #Here we add the genomics
#         else:
#             df_concat = df_aux

"""Concatenating Expression and Copy Number"""

Num_data = data_exp.shape[0]
if data_gene.shape[0]>data_exp.shape[0]:
    Num_data = data_gene.shape[0]

# for i in range(Num_data):
#     logic_q =  data_gene["DepMap_ID"][i] == data_drugs_NP24["DepMap_ID"]
#     if Num_duplicate>0:
#         df_aux = data_chemo[data_chemo.columns[0:]][i:i+1]
#         df_aux = pd.concat([df_aux]*Num_duplicate)
#         if i>0:
#             df_concat = pd.concat([df_concat, df_aux]) #Here we add the genomics
#         else:
#             df_concat = df_aux