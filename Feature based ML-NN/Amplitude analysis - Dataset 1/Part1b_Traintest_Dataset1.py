import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as TREE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Amplitude Dataset (Dataset 1)
dataset = np.load('Dataset1_amp-analysis_feature_based_dataset_comb_1170Hz.npy')
label = np.load('Label-amp-analysis_label_comb_1170Hz_new.npy')
amp = [10,20,50,100,250,500,750,1000]

dataset_id = 'Dataset1'
print(dataset_id)

# Hyperparameter values from Grid search results
LR_C = [0.0001,0.0001,0.0001,1,0.1,0.01,1,0.0001]
LR_solver = ['lbfgs','liblinear','lbfgs','lbfgs','lbfgs','lbfgs','lbfgs','liblinear']
#SVM_C = [100,10,100,10,10,1,1000,1]    # from Grid Search
SVM_C = [1,10,10,10,10,1,1000,1]        # Tweaked manually to optimize
#SVM_gamma = [0.001,0.1,0.0001,0.01,0.001,0.01,0.0001,0.001]
SVM_gamma = [0.001,0.0001,0.0001,0.001,0.001,0.01,0.0001,0.001]
#TREE_max_depth = [10,1,1,3,1,1,5,1]
TREE_max_depth = [3,1,1,3,1,1,5,1]
TREE_min_samples_split = [2,1.0,1.0,2,1.0,1.0,7,1.0]

 
# Training and testing 
for j in range(10):
    for i in range(len(amp)):
     
        start = int(800*i)
        stop = int(800*(i+1))
        X = dataset[start:stop,:]
        y = label[start:stop]
        
        # Splitting data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=j, stratify=y)
        
        # Different classifiers
        clf1 = LR(max_iter=3000,random_state=j,C=LR_C[i],solver=LR_solver[i])
        clf2 = SVC(random_state=j,kernel='rbf',C=SVM_C[i],gamma=SVM_gamma[i])
        clf3 = TREE(random_state=j,max_depth=TREE_max_depth[i],min_samples_split=TREE_min_samples_split[i])
    
        # Pipes for different classifiers
        pipe1 = make_pipeline(StandardScaler(),clf1)
        pipe2 = make_pipeline(StandardScaler(),clf2)
        pipe3 = make_pipeline(StandardScaler(),clf3)
            
        lr_gs = pipe1.fit(X_train, y_train)
        svm_gs = pipe2.fit(X_train, y_train) 
        tree_gs = pipe3.fit(X_train, y_train)    
        
        # Training and testing accuracy and loss
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
        
        print(j)
        print(amp[i])
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
#         # Writing the results to a text file
#         file = open("Dataset1_TrainTest_Results_final.txt", "a+")
#         file.write('%s,%i,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%s \n' %(dataset_id,amp[i],LR_train_accuracy,LR_test_accuracy,SVM_train_accuracy,SVM_test_accuracy,
#                     TREE_train_accuracy,TREE_test_accuracy,LR_precision,LR_recall,LR_f1,SVM_precision,SVM_recall,SVM_f1,
#                     TREE_precision,TREE_recall,TREE_f1,LR_confmat,SVM_confmat,TREE_confmat))
# =============================================================================
        
