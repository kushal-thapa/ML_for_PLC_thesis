import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# Amplitude Dataset (Datasets 3,5,7)
real_imag_spectra = np.load('Dataset5_amp-analysis_real_imag_spectra_comb_1170Hz.npy')
label = np.load('Label_amp-analysis_label_comb_1170Hz_new.npy')

amp = [10,20,50,100,250,500,750,1000]

dataset_id = 'Datasets 3, 5 & 7'
print(dataset_id)

# Scaling data
real_imag_spectra_scaled = preprocessing.scale(real_imag_spectra.flatten())
real_imag_spectra_new = np.reshape(real_imag_spectra_scaled,real_imag_spectra.shape)

dataset = real_imag_spectra_new
m= 'real_imag_spectra'

for i in range(9):    
   for n in range(len(amp)):
        # Splitting dataset into training and testing sets
        start = int(800*n)
        stop = int(800*(n+1))
        X = dataset[start:stop,:]
        y = label[start:stop]
        
        # Splitting data for training and testing
        train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.3,random_state=i, stratify=y)
        
        # Creating a model
        model = tf.keras.Sequential()
    
        # Convolution layers
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(3, 1024, 2)))
        model.add(tf.keras.layers.MaxPooling2D((1, 3)))
        
        # Neural network 
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        model.summary()
        
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
        print('PLC Amp:',amp[n])
        print('Testing accuracy:', test_acc)
        print('Testing loss:', test_loss)
        print('Precision: %f' % precision)
        print('Recall: %f' % recall)
        print('F1 score: %f' % f1)  
        print('Confusion matrix',confmat)
            
# =============================================================================
#         # Writing the results to a text file
#         file = open("Datasets3,5,7_TrainTest_Results_final.txt", "a+")
#         file.write('%s,%i,%.5f,%.5f,%s,%s,%s,%s,%.5f,%.5f,%.5f,%s \n' %(m,amp[n],test_acc,test_loss,training_acc,training_loss,validation_acc,validation_loss,
#                                              precision,recall,f1,confmat))
#         file.close()
# =============================================================================
        
     

