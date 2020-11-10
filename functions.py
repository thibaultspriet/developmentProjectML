
import pandas as pd
import numpy as np
import os

#NETOYAGE DE LA TABLE
#Permet de tester si il reste des valeurs manquantes
def test(data):
    data_na=data.isna()
    for k in data_na:
        for c in data_na[k]:
            if c==True:
                return "Il reste des valeurs manquantes"
    return "Toutes les valeurs manquantes ont été remplacées"


#Permet de remplacer les valeurs manquantes par 
#la moyenne des colonnes dont les valeurs sont des float ou des int
def clear_data_Float_Int(data,k,int_or_float):
        moy=data[k].mean()
        data_na=data[k].isna()
        if int_or_float==np.int:
            if moy-np.floor(moy)<0.5:
                    moy=int(moy)
            else:
                    moy=int(moy + 1)
        for value in range(len(data_na)):
            if data_na[value]:
                data.at[value,k]=moy

                
#Permet de remplacer les valeurs manquantes par
#la valeur du string qui est le plus présente des colonnes dont les valeurs ne sont pas des nombres 
def clear_data_String(data,k):
    list_value={}
    data_na=data[k].isna()
    for value in range(len(data_na)):
        if not data_na[value]:
            if data[k][value] not in list_value:
                list_value[data[k][value]]=0
            else:
                list_value[data[k][value]]+=1
    moy,Max="",0
    for value in list_value:
        if list_value[value]>Max:
            Max,moy=list_value[value],value
    
    for value in range(len(data)):
        if data_na[value]:
            data.at[value,k]=moy

#La fonction qui prend en argument les fichiers et qui remplace les valeurs manquantes
def clean_file(file):
    data = pd.read_csv(file)
    data_types=data.dtypes
    for index in data:
        if data_types[index]==np.object:
            clear_data_String(data,index)
        else:
            if data_types[index]==np.int:
                clear_data_Float_Int(data,index,np.int)
            else:
                clear_data_Float_Int(data,index,np.float)
    return data,  str(test(data))+"\nsur : " + str(file) + "\n"
    
data_b,test_banknote=clean_file('data_banknote_authentication.txt')
print(test_banknote)
data_k,test_kidney=clean_file('kidney_disease.csv')
print(test_kidney)

def normalize_data(data):
    for index in data:
        print(index)
        if data_types[index]==np.float:
            m , v= data[index].mean(),data[index].var()
            for value in range(len(data[index])):
                a=data[index][value]
                data.at[index,value]=(a-m)/v
                print(value)
                

#NORMALISATION DU CODE
                
#Permet de vérifier qu'on a normalisé et centré la table
def test_normalize(data):
    data_mean,data_var=data.mean(),data.std()
    data_types=data.dtypes
    for index in data:
        if data_types[index]==np.float:
            if abs(data[index].mean())>1e-10:
                return "La table n'est pas centrée"
            if np.abs(data[index].std()-1)>1e-10:
                return "La table n'est pas normalisée"
    return "La table est normalisée\n"

#Normalise et centre la table
def normalize_data(data):
    data_mean,data_var=data.mean(),data.std()
    data_types=data.dtypes
    for index in data:
        if data_types[index]==np.float:
            data[index]=(data[index]-data_mean[index])/data_var[index]
    return data_mean,data_var,data

data_b_mean,data_b_var,data_b=normalize_data(data_b)
print(test_normalize(data_b))
data_k_mean,data_k_var,data_k=normalize_data(data_k)
print(test_normalize(data_b))


