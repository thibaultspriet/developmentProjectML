import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

###############
# Clean data
###############

#NETOYAGE DE LA TABLE
#Permet de tester si il reste des valeurs manquantes
def testMissingValue(data):
    data_na=data.isna()
    for k in data_na:
        for c in data_na[k]:
            if c==True:
                print("Il reste des valeurs manquantes")
                return False
    print("Toutes les valeurs manquantes ont été remplacées")
    return True


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
def clear_data_String(data,k,data_na):
    list_value={}
    data_na=data_na[k]
    for value in range(len(data_na)):
        if data_na[value]:
            if data[k][value] not in list_value:
                list_value[data[k][value]]=0
            else:
                list_value[data[k][value]]+=1
    moy,Max=data[k][0],0
    for value in list_value:
        if list_value[value]>Max:
            Max,moy=list_value[value],value
    for value in range(len(data)):
        if not data_na[value]:
            data.at[value,k]=moy

#La fonction qui prend en argument les fichiers et qui remplace les valeurs manquantes
def clean_file(file):
    data = pd.read_csv(file)
    data_na=data.notna()
    data_types=data.dtypes
    for k in data:
        for c in range(len(data_na[k])):
            #Il est possible que certains string aient des \t ou des " ", il faut les enlever
            if type(data[k][c])==str:
                data.at[c,k]=data[k][c].replace(" ","")
                data.at[c,k]=data[k][c].replace("\t","")
                #Si un des NaN avait ce genre de caractères alors ils n'étaient pas repérés et comptaient 
                #Pour une valeur: On modifie donc la table data_na pour des valeurs en string
                if data[k][c]=="?":
                    data_na.at[c,k]=False
    for index in data:
        if data_types[index]==np.object:
            clear_data_String(data,index,data_na)
        else:
            if data_types[index]==np.int:
                clear_data_Float_Int(data,index,np.int)
            else:
                clear_data_Float_Int(data,index,np.float)
    return data,testMissingValue(data)

print('data_banknote_authentication.txt')
data_b,test_banknote=clean_file('data_banknote_authentication.txt')

print('kidney_disease.csv')
data_k,test_kidney=clean_file('kidney_disease.csv')
                

    
    
#NORMALISATION DU CODE                
#Permet de vérifier qu'on a normalisé et centré la table
def test_normalize(data):
    data_mean,data_var=data.mean(),data.std()
    data_types=data.dtypes
    for index in data:
        if data_types[index]==np.float:
            if abs(data[index].mean())>1e-10:
                print("La table n'est pas centrée")
                return False
            if np.abs(data[index].std()-1)>1e-10:
                print("La table n'est pas normalisée")
                return False
    print("La table est normalisée")
    return True

#Normalise et centre la table
def normalize_data(data):
    data_mean,data_var=data.mean(),data.std()
    data_types=data.dtypes
    for index in data:
        if data_types[index]==np.float:
            data[index]=(data[index]-data_mean[index])/data_var[index]
    return data_mean,data_var,data,test_normalize(data)

print('\ndata_banknote_authentication.txt')
data_b_mean,data_b_var,data_b,test_nb=normalize_data(data_b)

print('kidney_disease.csv')
data_k_mean,data_k_var,data_k,test_nk=normalize_data(data_k)

#ON REMPLACE LES STRING PAR DES INTS
def replace_by_Int(data):
    data_types=data.dtypes
    for k in data:
        if data_types[k]==np.object:
            list_value={}
            data_na=data[k].isna()
            number=0
            for value in range(len(data_na)):
                if not data_na[value]:
                    if data[k][value] not in list_value:
                        list_value[data[k][value]]=number
                        number+=1
                    data.at[value,k]=list_value[data[k][value]]
            data[k] = data[k].astype(int)


replace_by_Int(data_k)
print("\ntypes de la table\n : ",data_k.dtypes)
replace_by_Int(data_b)
print("\ntypes de la table : ",data_b.dtypes)


###############
# End clean data
###############


###############
# Split data
###############

# Author : SPRIET Thibault
def train_test(X,y,test_size,crossValidation,cross_size=None):
    """Split the datasst in two subsets train and test
    Parameters:
    ----------
    X : DataFrame
    y : DataFrame
    test_size : float
        between 0 and 1
    crossValidation : bool
    cross_size : float,optional

    Returns:
    --------
    X_train : DataFrame
    X_test : DataFrame
    y_train : DataFrame
    y_test : DataFrame
    """

    [X_train, X_test, y_train, y_test] = train_test_split(X,y,test_size=test_size)
    if crossValidation:
        if cross_size == None :
            raise ValueError("cross_size must be specified when crossValidation is True")
        print(f'X_train size : {X_train.shape[0]} ; pourcentage : {cross_size/(1-test_size)}')
        [X_train, X_cross, y_train, y_cross] = train_test_split(X_train,y_train,test_size=cross_size/(1-test_size))
        return [X_train,X_cross,X_test,y_train,y_cross,y_test]
    return [X_train,X_test,y_train,y_test]

###############
# End split data
###############


