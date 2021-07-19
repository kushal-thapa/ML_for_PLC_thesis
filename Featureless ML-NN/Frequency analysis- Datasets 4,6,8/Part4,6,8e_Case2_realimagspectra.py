import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


real_imag_spectra = np.load('Dataset6_freq-analysis_real_imag_spectra_comb_1A_comp.npy')
label = np.load('Label_freq-analysis_label_comb_1A_comp.npy')

freq = [690,750,810,870,930,990,1050,1110,1170,1230,1290,1350,1410,1470,1530,1590,1650,1710,1770,1830,1890,1950,2010]

dataset_id = 'Dataset 6'
print(dataset_id)

subset_idx_start = int(3*800)
subset_idx_end = int(18*800)

dataset_rec = real_imag_spectra[subset_idx_start:subset_idx_end]
label = label[subset_idx_start:subset_idx_end]
m = 'real_imag_spectra'

# Scaling data
real_imag_spectra_scaled = preprocessing.scale(dataset_rec.flatten())
dataset = np.reshape(real_imag_spectra_scaled,dataset_rec.shape)

for i in range(10):    
    # Splitting data for training and testing
    train_images, test_images, train_labels, test_labels = train_test_split(dataset, label, test_size=0.3,random_state=i, stratify=label)
    
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
    print('Testing accuracy:', test_acc)
    print('Testing loss:', test_loss)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)  
    print('Confusion matrix',confmat)
        
# =============================================================================
#     # Writing the results to a text file
#     file = open("Dataset6_comp_Case2_Results_realimagspectra_final.txt", "a+")
#     file.write('%s,%.5f,%.5f,%s,%s,%s,%s,%.5f,%.5f,%.5f,%s \n' %(m,test_acc,test_loss,training_acc,training_loss,validation_acc,validation_loss,
#                                          precision,recall,f1,confmat))
#     file.close()
# =============================================================================
    
# =============================================================================
#     # Learning and loss curve
#     plt.figure()
#     plt.plot(history.history['acc'],'r',linewidth=4,label='train acc')
#     plt.plot(history.history['val_acc'],'b--',linewidth=4,label='test acc')
#     plt.ylabel('accuracy',fontsize=24)
#     plt.xlabel('epoch',fontsize=24)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.legend(loc='center right',fontsize=20)
#     plt.show()
#     
#     plt.figure()
#     plt.plot(history.history['loss'],'r',linewidth=4,label='train loss')
#     plt.plot(history.history['val_loss'],'b--',linewidth=4,label='test loss')
#     #plt.title('Learning and loss curve')
#     plt.ylabel('loss',fontsize=24)
#     plt.xlabel('epoch',fontsize=24)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.legend(loc='center right',fontsize=20)
#     plt.show()
# =============================================================================

