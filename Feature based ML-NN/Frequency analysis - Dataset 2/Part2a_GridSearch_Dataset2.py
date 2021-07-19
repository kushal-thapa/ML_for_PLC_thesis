# Importing all required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as TREE

# Frequency Dataset :Dataset 2
# Uploading the dataset
X_full = np.load('Dataset2_freq-analysis_feature_based_dataset_comb_1A_comp.npy')
y_full = np.load('Label_freq-analysis_label_comb_1A_comp.npy')

PLC_freq = [690,750,810,870, 930, 990, 1050, 1110, 1170, 1230, 1290, 1350, 
               1410, 1470, 1530, 1590, 1650, 1710, 1770, 1830, 1890, 1950, 2010] 

#%%
# Case 1
for i in range(len(PLC_freq)):
    print(PLC_freq[i])
    start = int(800*i)
    stop = int(800*(i+1))
    X = X_full[start:stop,:]
    y = y_full[start:stop]
    
    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1, stratify=y)

    # Different classifiers
    clf1 = LR(max_iter=3000,random_state=1)
    clf2 = SVC(random_state=1,kernel='rbf') #,gamma=0.1)
    clf3 = TREE(random_state=1)

    # Pipes for different classifiers
    pipe1 = make_pipeline(StandardScaler(),clf1)
    pipe2 = make_pipeline(StandardScaler(),clf2)
    pipe3 = make_pipeline(StandardScaler(),clf3)

    # Grid Search
    #******************************************************************************
    
    # Logistic Regression
    param_range_lr = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    solver_range_lr = ['lbfgs', 'newton-cg', 'liblinear'] 
    param_grid_lr = [{'logisticregression__C': param_range_lr, 'logisticregression__solver': solver_range_lr}] 
    lr_gs = GridSearchCV(estimator=pipe1,param_grid=param_grid_lr,scoring='accuracy',cv=10)
    
    # Fitting and scores
    lr_gs = lr_gs.fit(X_train, y_train)
    LR_train_accuracy = lr_gs.best_score_
    LR_test_accuracy = lr_gs.score(X_test,y_test)
    LR_best_parameters = lr_gs.best_params_
    print('Training best score LR:', LR_train_accuracy)
    print('Best parameters LR:', LR_best_parameters)
    print('Test score LR:', LR_test_accuracy)
    
    #******************************************************************************
    
    # SVM
    param_range_svm = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid_svm = [{'svc__C': param_range_svm, 'svc__gamma': param_range_svm}]
    svm_gs = GridSearchCV(estimator=pipe2,param_grid=param_grid_svm,scoring='accuracy',cv=10)
    
    # Fitting and scores
    svm_gs = svm_gs.fit(X_train, y_train)
    SVM_train_accuracy = svm_gs.best_score_
    SVM_test_accuracy =svm_gs.score(X_test,y_test)
    SVM_best_parameters =svm_gs.best_params_
    print('Training best score LR:', SVM_train_accuracy)
    print('Best parameters LR:', SVM_best_parameters)
    print('Test score LR:', SVM_test_accuracy)
    
    #******************************************************************************

    # Decision Tree
    param_range_tree = [1,2,3,4,5,6,7,8,9,10,None]
    split_range_tree = [1.0,2,3,4,5,6,7]
    param_grid_tree = [{'decisiontreeclassifier__max_depth': param_range_tree, 
                        'decisiontreeclassifier__min_samples_split': split_range_tree}] 
    tree_gs = GridSearchCV(estimator=pipe3,param_grid=param_grid_tree,scoring='accuracy',cv=10)
    
    # Fitting and scores
    tree_gs = tree_gs.fit(X_train, y_train)
    TREE_train_accuracy = tree_gs.best_score_
    TREE_test_accuracy = tree_gs.score(X_test,y_test)
    TREE_best_parameters = tree_gs.best_params_
    print('Training best score LR:', TREE_train_accuracy)
    print('Best parameters LR:', TREE_best_parameters)
    print('Test score LR:', TREE_test_accuracy)
    
# =============================================================================
#     # Writing the results to a text file
#     dataset = 'Dataset 2: Case1'
#     file = open("Dataset2_comp_GS_Results_Case1.txt", "a+")
#     file.write('%s,%i,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%s \n' %(dataset,PLC_freq[i],LR_train_accuracy,LR_test_accuracy,SVM_train_accuracy,SVM_test_accuracy,
#                                          TREE_train_accuracy,TREE_test_accuracy,LR_best_parameters,SVM_best_parameters,TREE_best_parameters))
#     file.close()
# =============================================================================
    
#%%

# Case 2 and 3
subset_idx_start = int(3*800)
subset_idx_end = int(18*800)

dataset = X_full[subset_idx_start:subset_idx_end]
label = y_full[subset_idx_start:subset_idx_end]

# Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3,random_state=1, stratify=label)

# Different classifiers
clf1 = LR(max_iter=3000,random_state=1)
clf2 = SVC(random_state=1,kernel='rbf') 
clf3 = TREE(random_state=1)

# Pipes for different classifiers
pipe1 = make_pipeline(StandardScaler(),clf1)
pipe2 = make_pipeline(StandardScaler(),clf2)
pipe3 = make_pipeline(StandardScaler(),clf3)
# 
# Grid Search
#*****************************************************************************

# Logistic Regression
param_range_lr = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
solver_range_lr = ['lbfgs', 'newton-cg', 'liblinear'] 
param_grid_lr = [{'logisticregression__C': param_range_lr, 'logisticregression__solver': solver_range_lr}] 
lr_gs = GridSearchCV(estimator=pipe1,param_grid=param_grid_lr,scoring='accuracy',cv=10)
#     
# Fitting and score
lr_gs = lr_gs.fit(X_train, y_train)

print('Training best score LR:', lr_gs.best_score_)
print('Best parameters LR:', lr_gs.best_params_)
print('Test score LR:', lr_gs.score(X_test, y_test))

#*****************************************************************************
# SVM
param_range_svm = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid_svm = [{'svc__C': param_range_svm, 'svc__gamma': param_range_svm}]
svm_gs = GridSearchCV(estimator=pipe2,param_grid=param_grid_svm,scoring='accuracy',cv=10)
    
# Fitting and scores
svm_gs = svm_gs.fit(X_train[0:800] ,y_train[0:800])
print('Training best score SVM:', svm_gs.best_score_)
print('Best parameters SVM:', svm_gs.best_params_)
print('Test score SVM:', svm_gs.score(X_test, y_test))

#******************************************************************************

# Decision Tree
param_range_tree = [1,2,3,4,5,6,7,8,9,10,None]
split_range_tree = [1.0,2,3,4,5,6,7]
param_grid_tree = [{'decisiontreeclassifier__max_depth': param_range_tree, 
                        'decisiontreeclassifier__min_samples_split': split_range_tree}] 
tree_gs = GridSearchCV(estimator=pipe3,param_grid=param_grid_tree,scoring='accuracy',cv=10)

# Fitting and scores
tree_gs = tree_gs.fit(X_train[0:800], y_train[0:800])
print('Training best score TREE:', tree_gs.best_score_)
print('Best parameters TREE:', tree_gs.best_params_)
print('Test score TREE:', tree_gs.score(X_test, y_test))
