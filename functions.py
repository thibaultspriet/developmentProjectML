import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score, ShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import statistics as s

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.cluster import KMeans

import mimetypes

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
    return True

#Normalise et centre la table
def normalize_data(data):
    data_mean,data_var=data.mean(),data.std()
    data_types=data.dtypes
    for index in data:
        if data_types[index]==np.float:
            data[index]=(data[index]-data_mean[index])/data_var[index]
    return data,test_normalize(data)

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


#La fonction qui prend en argument les fichiers et qui remplace les valeurs manquantes
def clean_file(file):
    print('---START CLEANING : ',file,'---')
    #Ici si ce n'est pas un fichier csv on suppose qu'il n'y a pas forcément
    #les noms de colonnes il faut donc éciter que la première ligne devienne
    #le header
    if file[-3:] != 'csv':
        data = pd.read_csv(file,header=None)
    else:
        data=pd.read_csv(file)
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
    if not testMissingValue(data):
            return data
    print("Toutes les valeurs manquantes ont été remplacées")
    data,test_normalize=normalize_data(data)
    if not test_normalize:
            return data_mean,data_var,data
    print("La table est normalisée")
    replace_by_Int(data)
    print('---END CLEANING :',file,'---\n')
    return data

#data_k = clean_file('kidney_disease.csv')
#data_b = clean_file('data_banknote_authentication.txt')

#data_k.to_csv('kidney_disease_cleaned.csv')
#data_b.to_csv('data_banknote_cleaned.csv')

# Author : Thibault Spriet
def getDataLabels(data):
    """Gives the data and labels in two different df

    Parameters
    ----------
    data : DataFrame
        Cleaned data (labels in the last column)

    Returns
    -------
    tuple(DataFrame)
        the data and the labels
    """
    if "id" in data.columns:
        data = data.drop("id",axis=1)
    labels = data.iloc[:,-1]
    dataValue = data.iloc[:,:data.shape[1]-1]
    return dataValue,labels



###############
# End clean data
###############


###############
# Split data
###############

# Author : SPRIET Thibault
def train_test(X,y,test_size,crossValidation, cross_size=None):
    """Split the dataset in two subsets train and test
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


###############
# PCA
###############

def PCA(X_train, X_test) :
    '''Applies the pca algorithm on the dataset.
    Parameters:
    ----------
    X_train : DataFrame
    X_test : DataFrame

    Returns:
    --------
    X_train_PCA : DataFrame
    X_test_PCA : DataFrame
    '''
    X = np.concatenate((X_train, X_test))

    pca = PCA(n_components = 'mle') # we let the algorithm decide of the optimal number of components
    pca.fit(X)
    X = pca.transform(X)

    X_train_PCA = X[:len(X_train)]
    X_test_PCA = X[len(X_train):]
    return(X_train_PCA, X_test_PCA)


###############
# End PCA
###############


###############
# Train models
###############

#SVM classifiers
# Author : SPRIET Thibault
def trainSVMlinear(X,y):
    """train an SVM classifier

    Parameters
    ----------
    X : Array
        dimension : n_samples x n_feature
    y : Array
        dimension : N_samples x 1. Labels

    Returns
    -------
    object
        the train classifier
    """
    clf = SVC(kernel="linear")
    clf.fit(X,y)
    return clf

def trainSVMpoly(X,y,degree,gamma,r):
    clf = SVC(kernel="poly",degree = degree,gamma = gamma,coef0 = r)
    clf.fit(X,y)
    return clf

def trainSVMrbf(X,y,gamma):
    clf = SVC(kernel="rbf",gamma = gamma)
    clf.fit(X,y)
    return clf

def trainSVMsigmoid(X,y,gamma):
    clf = SVC(kernel="sigmoid",gamma = gamma)
    clf.fit(X,y)
    return clf

# Author : Thibault Spriet
def trainSVM(X,y,kernel="linear"):
    """train an SVM classifier

    Parameters
    ----------
    X : DataFrame
        data features
    y : Serie
        labels
    kernel : string
        accepted : linear, poly, rbf, sigmoid

    Returns
    -------
    Object
        trained classifier
    """
    clf = SVC(kernel=kernel)
    clf.fit(X,y)
    return clf



#Logistic Regression

def trainLogisticRegression(X,y) :
    clf = LogisticRegression()
    clf.fit(X, y)
    return(clf)



# Fonction simplificatrice
def maxl(l) :
    max = l[0]
    i = 0
    for k in range(len(l)) :
        if l[k] > max :
            max = l[k]
            i = k
    return(i)

# Decision forest
def trainDecisionForest(X,y,n_trees) :
    # Optimization of the depth of the trees using cross-validation

    # Define the cvp (cross-validation procedure)
    cvp = ShuffleSplit(n_splits=1000, test_size=1/3, train_size=2/3)

    # Define the max depths between 1 and 10
    n_depths = 10
    depths = np.linspace(1, 10, n_depths)

    # Loop on the max_depth parameter and compute accuracy
    tab_accuracy_tree = np.zeros(n_depths)
    for i in range(n_depths):
        class_tree = DecisionTreeClassifier(max_depth=depths[i])
        tab_accuracy_tree[i] = s.median(cross_val_score(class_tree, X, y, scoring='accuracy', cv=cvp))
    #plot(depths, tab_accuracy_tree)

    opt = maxl(tab_accuracy_tree) + 1 # depth for which we get the maximum accuracy

    # Train Decision forest :
    class_forest = RandomForestClassifier(n_estimators=n_trees, max_depth=3)
    class_forest.fit(X, y)
    return(class_forest)


# Ada Boost classifier
def trainAdaBoost(X,y,n_trees):
    class_ada = AdaBoostClassifier(n_estimators=n_trees)
    class_ada.fit(X,y)
    return(class_ada)


###############
# End train models
###############

###############
# Test models
###############

# Author : SPRIET Thibault
def testSVM(SVMclf,X_test):
    """Test the SVM classifier

    Parameters
    ----------
    SVMclf : object
        the trained classifier
    X_test : Array
        n_samples x n_feature

    Returns
    -------
    Array
        classification labels
    """
    return SVMclf.predict(X_test)


def testLogReg(LRclf, X_test) :
    """ Test the Logistic Regression

    Parameters
    ----------
    LRclf : object
        the trained classifier
    X_test : Array
        n_samples x n_features

    Returns
    -------
    Array
        classification labels
    """
    return(LRclf.predict(X_test))


def testDecisionForest(DFclf, X_test) :
    return(DFclf.predict(X_test))


def testAdaBoost(ABclf, X_test) :
    return(ABclf.predict(X_test))


def testKmeans(X_train, y_train, X_test) :
    """ We apply the kmeans algorithm on the whole dataset, and we look at the repartition of the train part in the clusters. We then classify X_test thanks to the clusters and their caracteristics."""

    # Kmeans algorithm on the whole dataset
    X = np.concatenate((X_train, X_test))
    kmeans = KMeans(n_clusters=2).fit(X)
    lab = kmeans.labels_

    # We look at the repartition of X_train in the clusters
    c0 = 0
    c1 = 0
    for k in range(len(X_train)) :
        if lab[k] == 0 :
            c0 += np.array(y_train)[k]
        else :
            c1 += np.array(y_train)[k]
    if c0 > c1 : # if we got more y_train = 0 in the cluster 1
        c0 = 1
        c1 = 0
    else : # if we got more y_train = 0 in the cluster 0
        c0 = 0
        c1 = 1

    # Classification of X_test
    y_test = np.zeros(len(X_test))
    i = 0
    for k in range(len(X_train),len(X)) :
        if lab[k] == 0 :
            y_test[i] = c0
        else :
            y_test[i] = c1
        i += 1
    return(y_test)


###############
# End test models
###############

###############
# Cross Validation
###############

def CrossValidation(X,y,degree,gamma,r,cv):
    """Find the best classifier with a cross validation

    Parameters
    ----------
    X : Array
        dimension : n_samples x n_feature
    y : Array
        dimension : N_samples x 1. Labels
    degree,gamma,r : parameters of the kernels
    cv : a cv-crossed validation
        (fit a model and compute the score 5 consecutive times (with different splits each time))

    Returns
    -------
    Best_mean : float
        the best mean
    Best_std : float
        the std of the best classifier
    Best_Classifier : string
        the svm classifier with the best mean

    """
    linear_clf = cross_val_score(trainSVMlinear(X,y), X, y, cv=cv,scoring='f1_macro')
    poly_clf = cross_val_score(trainSVMpoly(X,y,degree,gamma,r), X, y, cv=cv,scoring='f1_macro')
    rbf_clf = cross_val_score(trainSVMrbf(X,y,gamma), X, y, cv=cv,scoring='f1_macro')
    sigmoid_clf = cross_val_score(trainSVMsigmoid(X,y,gamma), X, y, cv=cv,scoring='f1_macro')
    Mean = [linear_clf.mean(),poly_clf.mean(),rbf_clf.mean(),sigmoid_clf.mean()]
    Std = [linear_clf.std(),poly_clf.std(),rbf_clf.std(),sigmoid_clf.std()]
    Classifier = ['linear','poly','rbf','sigmoid']
    Best_mean = 0
    Best_std = 0
    Best_classifier = Classifier[0]
    for k in range(len(Mean)):
        if Mean[k]>Best_mean:
            Best_mean = Mean[k]
            Best_std = Std[k]
            Best_Classifier = Classifier[k]
    print("The best classifier is %0.2f : Accuracy: %0.2f (+/- %0.2f)" % (Best_Classifier, Best_mean, Best_std * 2))
    return Best_mean, Best_std, Best_Classifier

# Author : Thibault Spriet
def crossValidationProcedure(clf,X,y,parameter,values):
    """return best trained classifier

    Parameters
    ----------
    clf : Object
        classifier
    X : DataFrame
        Cross validation data
    y : Serie
        Cross validation labels
    parameter : string
        name of parameter to tune
    values : list
        allowed values of the parameter

    Returns
    -------
    Object
        classifier
    """
    cvp = ShuffleSplit(n_splits=1000, test_size=1/3, train_size=2/3)

    # Loop on the max_depth parameter and compute median RMSE
    tab_measure = np.zeros(len(values))
    for i in range(len(values)):
        params = {parameter : values[i]}
        classifier = clf.set_params(**params)
        tab_measure[i] = np.median(cross_val_score(classifier, X, y, scoring='accuracy', cv=cvp))
    index = np.argmax(tab_measure)
    print(f'best {parameter} = {values[index]}')
    return clf.set_params(**{parameter:values[index]}).fit(X,y)



###############
# End Cross Validation
###############



###############
# Validate model
###############

def confusionMatrix(y,y_predicted,title=None):
    """plot the confusion matrix

    Parameters
    ----------
    y : Array
        true labels
    y_predicted : Array
        predicted labels
    title : string, optional
        title of the matrix, by default None

    Returns
    -------
    subplot
        matrix confusion
    """
    conf = confusion_matrix(y, y_predicted)
    ax = sns.heatmap(conf,annot=True)
    ax.set_title(title)
    return ax

def validateModel(y,y_pred):
    """Print relevant pieces of information to evaluate a model

    Parameters
    ----------
    y : Array
        true labels
    y_pred : Array
        predicted labels
    """
    acc = accuracy_score(y,y_pred,normalize=True)*100
    rec = recall_score(y,y_pred, average = 'binary') * 100
    f1 = f1_score(y,y_pred)
    print(f'Your model has an accuracy of : {acc}%\nYour model has a recall of : {rec}%\nYour model\'s F1 score = {f1} ')


###############
# End Validate model
###############