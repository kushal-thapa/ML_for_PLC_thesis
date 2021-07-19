
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Case 3: Training with multiple frequency, testing with a seperate one
dataset_id = 'Datasets 4, 6 & 8'
case = 'Case 3'

# Loading the dataset files
mag_spectra = np.load('Dataset4_freq-analysis_mag_spectra_comb_1A_comp.npy')
label = np.load('Label_freq-analysis_label_comb_1A_comp.npy')

# =============================================================================
# # Scaling data
# mag_spectra_scaled = preprocessing.scale(mag_spectra.flatten())
# mag_spectra_new = np.reshape(mag_spectra_scaled,mag_spectra.shape)
# =============================================================================

#dataset = np.reshape(mag_spectra,(17600,3,1024,1))
m = 'mag_spectra'
freq = [690,750,810,870,930,990,1050,1110,1170,1230,1290,1350,1410,1470,1530,1590,1650,1710,1770,1830,1890,1950,2010]
freq_subset = [870,930,990,1050,1110,1170,1230,1290,1350,1410,1470,1530,1590,1650,1710]

subset_idx_start = int(3*800)
subset_idx_end = int(18*800)

dataset_rec = mag_spectra[subset_idx_start:subset_idx_end]
label = label[subset_idx_start:subset_idx_end]
m = 'mag_spectra'

# mean centering and standardization
mean_vals = np.mean(dataset_rec,axis=0)
std_val = np.std(dataset_rec)
dataset = (dataset_rec-mean_vals)/std_val


for i in range(10):
    for n in range(len(freq_subset)):
            # Splitting dataset into training and testing sets
            test_start_idx = int(n*800)
            test_end_idx = int((n+1)*800)
            test_images = dataset[test_start_idx:test_end_idx]
            train_images = np.concatenate((dataset[0:test_start_idx],dataset[test_end_idx:-1]),axis=0)
            test_labels = label[test_start_idx:test_end_idx]
            train_labels = np.concatenate((label[0:test_start_idx],label[test_end_idx:-1]),axis=0)
               
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
            history = model.fit(train_images, train_labels, epochs=20, batch_size=16, validation_data=(test_images,test_labels))
        
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
            print('PLC Freq',freq_subset[n])
            print('Testing accuracy:', test_acc)
            print('Testing loss:', test_loss)
            print('Precision: %f' % precision)
            print('Recall: %f' % recall)
            print('F1 score: %f' % f1)  
            print('Confusion matrix',confmat)
            
# =============================================================================
#             # Writing the results to a text file
#             file = open("Datasets4,6,8_Case3_magspectra_Results_final.txt", "a+")
#             file.write('%s,%i,%.5f,%.5f,%s,%s,%s,%s,%.5f,%.5f,%.5f,%s \n' %(m,freq_subset[n],test_acc,test_loss,training_acc,training_loss,validation_acc,validation_loss,
#                                                  precision,recall,f1,confmat))
#             file.close()
# =============================================================================
        
# =============================================================================
#             # Learning and loss curve
#             plt.figure()
#             plt.plot(history.history['acc'],'r',linewidth=4,label='train acc')
#             plt.plot(history.history['val_acc'],'b--',linewidth=4,label='test acc')
#             plt.ylabel('accuracy',fontsize=24)
#             plt.xlabel('epoch',fontsize=24)
#             plt.xticks(fontsize=20)
#             plt.yticks(fontsize=20)
#             plt.legend(loc='center right',fontsize=20)
#             plt.show()
#             
#             plt.figure()
#             plt.plot(history.history['loss'],'r',linewidth=4,label='train loss')
#             plt.plot(history.history['val_loss'],'b--',linewidth=4,label='test loss')
#             #plt.title('Learning and loss curve')
#             plt.ylabel('loss',fontsize=24)
#             plt.xlabel('epoch',fontsize=24)
#             plt.xticks(fontsize=20)
#             plt.yticks(fontsize=20)
#             plt.legend(loc='center right',fontsize=20)
#             plt.show()
# =============================================================================
