import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# Amplitude Dataset (Datasets 3,5,7)
time_series = np.load('Dataset7_amp-analysis_time_series_dataset_comb_1170Hz.npy')
label = np.load('Label_amp-analysis_label_comb_1170Hz_new.npy')
m='time_series'

amp = [10,20,50,100,250,500,750,1000]

dataset_id = 'Datasets 3, 5 & 7'
print(dataset_id)

# Scaling data
time_series_scaled = preprocessing.scale(time_series.flatten())
time_series_new = np.reshape(time_series_scaled,time_series.shape)


dataset = time_series_new

for i in range(1):    
    for n in range(len(amp)):
        if n==7:
             # Splitting dataset into training and testing sets
             start = int(800*n)
             stop = int(800*(n+1))
             X = dataset[start:stop,:]
             y = label[start:stop]
             
             # Splitting data for training and testing
             train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.3,random_state=1, stratify=y)
             
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
         
             training_acc = history.history['accuracy']
             training_loss = history.history['loss']
             validation_acc = history.history['val_accuracy']
             validation_loss = history.history['val_loss']
             
            # predict probabilities for test set
             yhat_probs = model.predict(test_images)
             # predict crisp classes for test set
             yhat_classes = model.predict_classes(test_images)
             
             precision = precision_score(test_labels, yhat_classes)
             recall = recall_score(test_labels, yhat_classes)
             f1 = f1_score(test_labels, yhat_classes)
             confmat = confusion_matrix(y_true=test_labels, y_pred=yhat_classes)
                  
             # Testing accuracy and loss
             test_loss, test_acc = model.evaluate(test_images, test_labels)
             print('Iteration:',i)
             print('dataset:',m)
             print('PLC_freq',amp[n])
             print('Testing accuracy:', test_acc)
             print('Testing loss:', test_loss)
             print('Precision: %f' % precision)
             print('Recall: %f' % recall)
             print('F1 score: %f' % f1)  
             print('Confusion matrix',confmat)
                 
# =============================================================================
#              # Writing the results to a text file
#              file = open("Datasets3,5,7_TrainTest_timeseries_Results_final.txt", "a+")
#              file.write('%s,%i,%.5f,%.5f,%s,%s,%s,%s,%.5f,%.5f,%.5f,%s \n' %(m,amp[n],test_acc,test_loss,training_acc,training_loss,validation_acc,validation_loss,
#                                                   precision,recall,f1,confmat))
#              file.close()
# =============================================================================
         
