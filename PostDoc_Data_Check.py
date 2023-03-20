
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Select M: Mutation, Gene Copy Number (GCMP: CellModelPassport or GDM: DepMap)  and Expression (ERMA or EDM: DepMap),
#Also, select
Which_data_type = "M"

if Which_data_type is "M":
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
        print(f"The column_name = {colum_name} Exist")
        print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > 0:
            count_high_nan = count_high_nan + 1
        else:
            count_low_nan = count_low_nan + 1

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

elif Which_data_type is "GCMP":
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
            print(f"The column_name = {colum_name} does EXIST EXIST")
            print(f"With a Number of NaN = {num_of_nan}")
            if num_of_nan > 100:
                count_high_nan = count_high_nan + 1
            else:
                count_low_nan = count_low_nan + 1
        else:
            print(f"The colum_name = {colum_name} does NOT exist")
    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")
elif Which_data_type is "GDM":
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN18.csv")
    data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN19.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN20.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Gene copy number Datasets/DepMap/GCN21.csv")

    count_high_nan = 0
    count_low_nan = 0
    for colum_name in data_gene.columns:
        num_of_nan = data_gene[colum_name].isna().sum()
        print(f"The column_name = {colum_name} Exist")
        print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > 0:
            count_high_nan = count_high_nan + 1
        else:
            count_low_nan = count_low_nan + 1

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

elif Which_data_type is "ERMA":
    data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/RMA_proc_basalEXP.csv")

    count_high_nan = 0
    count_low_nan = 0
    for colum_name in data_gene.columns:
        num_of_nan = data_gene[colum_name].isna().sum()
        print(f"The column_name = {colum_name} Exist")
        print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > 0:
            count_high_nan = count_high_nan + 1
        else:
            count_low_nan = count_low_nan + 1

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

elif Which_data_type is "EDM":
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP18_TPM.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP19.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP19_Q1_TPM.csv")
    #data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP20.csv")
    data_gene = pd.read_csv("/media/juanjo/Elements/Postdoc_Data/Datasets_Postdoc/Expression Datasets/DepMap/EXP21.csv")

    count_high_nan = 0
    count_low_nan = 0
    for colum_name in data_gene.columns:
        num_of_nan = data_gene[colum_name].isna().sum()
        print(f"The column_name = {colum_name} Exist")
        print(f"With a Number of NaN = {num_of_nan}")
        if num_of_nan > 22:
            count_high_nan = count_high_nan + 1
        else:
            count_low_nan = count_low_nan + 1

    print(f"Number of Data low NaN = {count_low_nan}\nNumber of Data High NaN = {count_high_nan}")

