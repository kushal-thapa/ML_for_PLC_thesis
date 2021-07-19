
# Importing all required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as TREE

# Amplitude Dataset (Dataset 1)
# Uploading the dataset, creating dataframe (using pandas)
file = np.load('Dataset1_amp-analysis_feature_based_dataset_comb_1170Hz.npy')

X_full = file
y_full = np.load('Label-amp-analysis_label_comb_1170Hz_new.npy')
amp = [10,20,50,100,250,500,750,1000]

for i in range(len(amp)):
    print(amp[i])
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
#     dataset = 'Dataset 1'
#     file = open("Dataset1_GS_Results.txt", "a+")
#     file.write('%s,%i,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%s \n' %(dataset,amp[i],LR_train_accuracy,LR_test_accuracy,SVM_train_accuracy,SVM_test_accuracy,
#                                          TREE_train_accuracy,TREE_test_accuracy,LR_best_parameters,SVM_best_parameters,TREE_best_parameters))
#     file.close()
# =============================================================================





