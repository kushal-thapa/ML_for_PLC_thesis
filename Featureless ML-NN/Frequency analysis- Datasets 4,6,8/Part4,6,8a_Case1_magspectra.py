import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


mag_spectra = np.load('Dataset4_freq-analysis_mag_spectra_comb_1A_comp.npy')
label = np.load('Label_freq-analysis_label_comb_1A_comp.npy')

freq = [690,750,810,870,930,990,1050,1110,1170,1230,1290,1350,1410,1470,1530,1590,1650,1710,1770,1830,1890,1950,2010]

dataset_id = 'Datasets 4'
print(dataset_id)

# =============================================================================
# # Scaling data
# mag_spectra_scaled = preprocessing.scale(mag_spectra.flatten())
# mag_spectra_new = np.reshape(mag_spectra_scaled,mag_spectra.shape)
# =============================================================================

dataset = mag_spectra
m = 'mag_spectra'


for i in range(10):    
    for n in range(len(freq)):
        # Splitting dataset into training and testing sets
        start = int(800*n)
        stop = int(800*(n+1))
        X = dataset[start:stop,:]
        y = label[start:stop]
        
        # Splitting data for training and testing
        train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.3,random_state=i+2, stratify=y)
        
        # mean centering and standardization
        mean_vals = np.mean(train_images,axis=0)
        std_val = np.std(train_images)
        train_images = (train_images-mean_vals)/std_val
        test_images = (test_images-mean_vals)/std_val
        
        # Creating a model
        model = tf.keras.Sequential()
           
        # Neural network 
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        
        # Fitting the model with training data
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) #,beta_1=0.9,epsilon=1e-07)
        model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_images, train_labels, epochs=50, batch_size=16, validation_data=(test_images,test_labels))
    
        training_acc = history.history['acc']
        training_loss = history.history['loss']
        validation_acc = history.history['val_acc']
        validation_loss = history.history['val_loss']
        
        # predict probabilities for test set
        yhat_probs = model.predict(test_images)
        # predict crisp classes for test set
        yhat_classes = model.predict_classes(test_images)
        
        precision = precision_score(test_labels, yhat_classes)
        recall = recall_score(test_labels, yhat_classes)
        f1 = f1_score(test_labels, yhat_classes)
        confmat = np.reshape(confusion_matrix(y_true=test_labels, y_pred=yhat_classes),(1,4))
             
        # Testing accuracy and loss
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Iteration:',i)
        print('dataset:',m)
        print('PLC Freq',freq[n])
        print('Testing accuracy:', test_acc)
        print('Testing loss:', test_loss)
        print('Precision: %f' % precision)
        print('Recall: %f' % recall)
        print('F1 score: %f' % f1)  
        print('Confusion matrix',confmat)
        
# =============================================================================
#         # Learning and loss curve
#         plt.figure()
#         plt.plot(history.history['acc'],'r',linewidth=4,label='train acc')
#         plt.plot(history.history['val_acc'],'b--',linewidth=4,label='test acc')
#         plt.ylabel('accuracy',fontsize=24)
#         plt.xlabel('epoch',fontsize=24)
#         plt.xticks(fontsize=20)
#         plt.yticks(fontsize=20)
#         plt.legend(loc='center right',fontsize=20)
#         plt.show()
#         
#         plt.figure()
#         plt.plot(history.history['loss'],'r',linewidth=4,label='train loss')
#         plt.plot(history.history['val_loss'],'b--',linewidth=4,label='test loss')
#         #plt.title('Learning and loss curve')
#         plt.ylabel('loss',fontsize=24)
#         plt.xlabel('epoch',fontsize=24)
#         plt.xticks(fontsize=20)
#         plt.yticks(fontsize=20)
#         plt.legend(loc='center right',fontsize=20)
#         plt.show()
# =============================================================================

        
# =============================================================================
#         # Writing the results to a text file
#         file = open("Dataset4_comp_Case1_Results_magspectra_final.txt", "a+")
#         file.write('%s,%i,%.5f,%.5f,%s,%s,%s,%s,%.5f,%.5f,%.5f,%s \n' %(m,freq[n],test_acc,test_loss,training_acc,training_loss,validation_acc,validation_loss,
#                                              precision,recall,f1,confmat))
#         file.close()
# =============================================================================
        
        

