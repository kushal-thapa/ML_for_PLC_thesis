# Importing all required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as TREE
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Frequency Dataset (Dataset 2)
dataset_full = np.load('Dataset2_freq-analysis_feature_based_dataset_comb_1A_comp.npy')
label = np.load('Label_freq-analysis_label_comb_1A_comp.npy')
PLC_freq = [690,750,810,870, 930, 990, 1050, 1110, 1170, 1230, 1290, 1350, 
               1410, 1470, 1530, 1590, 1650, 1710, 1770, 1830, 1890, 1950, 2010] 

subset_idx_start = int(3*800)
subset_idx_end = int(18*800)

dataset = dataset_full[subset_idx_start:subset_idx_end]
label = label[subset_idx_start:subset_idx_end]

# Case 2
dataset_id = 'Dataset 2'
case = 'Case 2'

for i in range(10):
    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3,random_state=i, stratify=label)

    # Different classifiers
    clf1 = LR(max_iter=3000,random_state=1,C=10,solver='lbfgs')
    clf2 = SVC(random_state=1,kernel='rbf',C=10,gamma=0.01)
    clf3 = TREE(random_state=1,max_depth=10,min_samples_split=2)
    
    # Pipes for different classifiers
    pipe1 = make_pipeline(StandardScaler(),clf1)
    pipe2 = make_pipeline(StandardScaler(),clf2)
    pipe3 = make_pipeline(StandardScaler(),clf3)
    
    lr_gs = pipe1.fit(X_train, y_train)
    svm_gs = pipe2.fit(X_train, y_train) 
    tree_gs = pipe3.fit(X_train, y_train)
    # 
    # Training and testing 
    LR_train_accuracy = lr_gs.score(X_train,y_train)
    LR_test_accuracy = lr_gs.score(X_test,y_test)
    SVM_train_accuracy = svm_gs.score(X_train,y_train)
    SVM_test_accuracy =svm_gs.score(X_test,y_test)
    TREE_train_accuracy = tree_gs.score(X_train,y_train)
    TREE_test_accuracy = tree_gs.score(X_test,y_test)
    
    # precision, recall, and F1
    LR_yhat_classes = lr_gs.predict(X_test)
    LR_precision = precision_score(y_test, LR_yhat_classes)
    LR_recall = recall_score(y_test, LR_yhat_classes)
    LR_f1 = f1_score(y_test, LR_yhat_classes)
    LR_confmat = np.reshape(confusion_matrix(y_true=y_test, y_pred=LR_yhat_classes),(1,4))
    
    SVM_yhat_classes = svm_gs.predict(X_test)
    SVM_precision = precision_score(y_test, SVM_yhat_classes)
    SVM_recall = recall_score(y_test, SVM_yhat_classes)
    SVM_f1 = f1_score(y_test, SVM_yhat_classes)
    SVM_confmat = np.reshape(confusion_matrix(y_true=y_test, y_pred=SVM_yhat_classes),(1,4))
    
    TREE_yhat_classes = tree_gs.predict(X_test)
    TREE_precision = precision_score(y_test, TREE_yhat_classes)
    TREE_recall = recall_score(y_test, TREE_yhat_classes)
    TREE_f1 = f1_score(y_test, TREE_yhat_classes)
    TREE_confmat = np.reshape(confusion_matrix(y_true=y_test, y_pred=TREE_yhat_classes),(1,4))
    
    print(i)
    print('Training score LR:', LR_train_accuracy)
    print('Test score LR:', LR_test_accuracy)
    print('Training score SVM:', SVM_train_accuracy)
    print('Test score SVM:', SVM_test_accuracy)
    print('Training score TREE:', TREE_train_accuracy)
    print('Test score TREE:', TREE_test_accuracy)
    print('LR precision:', LR_precision)
    print('LR recall:', LR_recall)
    print('LR f1:', LR_f1)
    print('LR confusion matrix:', LR_confmat)
    print('SVM precision:', SVM_precision)
    print('SVM recall:', SVM_recall)
    print('SVM f1:', SVM_f1)
    print('SVM confusion matrix:', SVM_confmat)
    print('TREE precision:', TREE_precision)
    print('TREE recall:', TREE_recall)
    print('TREE f1:', TREE_f1)
    print('TREE confusion matrix:', TREE_confmat)
        
# =============================================================================
#     # Writing the results to a text file
#     file = open("Dataset2_old_Case2_Results_Train_Test_final.txt", "a+")
#     file.write('%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%s \n' %(dataset_id,LR_train_accuracy,LR_test_accuracy,SVM_train_accuracy,SVM_test_accuracy,
#                 TREE_train_accuracy,TREE_test_accuracy,LR_precision,LR_recall,LR_f1,SVM_precision,SVM_recall,SVM_f1,
#                 TREE_precision,TREE_recall,TREE_f1,LR_confmat,SVM_confmat,TREE_confmat))
#     file.close()
# =============================================================================