import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('classic')

# Uploading the NN dataset
df_raw = pd.read_csv('Dataset1_TrainTest_NN_Results_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_NN = df_raw.dropna(axis='index')

NN_test_acc = np.array(df_NN[2])
NN_train_acc = (np.array(df_NN[53].str.replace(']',''))).astype(float)

# Uploading the dataset, creating dataframe (using pandas)
column_names= ['Dataset','PLC Amplitude(mA)','LR train acc','LR test acc','SVM train acc','SVM test acc','TREE train acc','TREE test acc',
               'LR_precision','LR_recall','LR_f1','SVM_precision','SVM_recall','SVM_f1','TREE_precision','TREE_recall',
               'TREE_f1','LR_confmat','SVM_confmat','TREE_confmat']
df = pd.read_csv('Dataset1_TrainTest_ML_Results_final.txt',sep=',', names=column_names, engine='python',index_col=False)
df['NN train acc'] = NN_train_acc
df['NN test acc'] = NN_test_acc

X = np.unique(np.array(df['PLC Amplitude(mA)']))

# Test Accuracy
plt.figure()
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['LR test acc'],linewidth=4,color='r',linestyle='--',label='Logistic Regression',ci=95)
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['SVM test acc'],linewidth=4,color='b',linestyle='-',label='SVM',ci=95)
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['TREE test acc'],linewidth=8,color='g',linestyle=':',label='Decision Tree',ci=95)
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['NN test acc'],linewidth=8,color='k',linestyle='-.',label='Neural Network',ci=95)
plt.xlabel('PLC signal amplitude (mA) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0,1,21),fontsize=20)
plt.ylim(0.4,1.05)
plt.legend(loc='center right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

df_zoom = df[df['PLC Amplitude(mA)']>=250]
X_zoom = np.unique(np.array(df_zoom['PLC Amplitude(mA)']))
plt.figure()
sns.lineplot(x=df_zoom['PLC Amplitude(mA)'],y=df_zoom['LR test acc'],linewidth=4,color='r',linestyle='--',ci=95)
sns.lineplot(x=df_zoom['PLC Amplitude(mA)'],y=df_zoom['SVM test acc'],linewidth=4,color='b',linestyle='-',ci=95)
sns.lineplot(x=df_zoom['PLC Amplitude(mA)'],y=df_zoom['TREE test acc'],linewidth=8,color='g',linestyle=':',ci=95)
sns.lineplot(x=df_zoom['PLC Amplitude(mA)'],y=df_zoom['NN test acc'],linewidth=8,color='k',linestyle='-.',ci=95)
#plt.xlabel('PLC signal amplitude (mA) ',fontsize=24)
#plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X_zoom,fontsize=20)
plt.yticks(np.linspace(0.9,1,11),fontsize=20)
plt.ylim(0.9,1)
#plt.legend(loc='center right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

# Train and test accuracy subplots
fig,ax = plt.subplots(4,1,sharex=True,sharey=True)

sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['LR train acc'],linewidth=4,color='r',linestyle='-',label='Logistic Regression Training',ci=95,ax=ax[0])
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['LR test acc'],linewidth=4,color='r',linestyle='--',label='Logistic Regression Testing',ci=95,ax=ax[0])
ax[0].set_ylabel('')
ax[0].set_xticks(X)
ax[0].tick_params(axis='both', which='both', labelsize=20)
ax[0].legend(loc='lower right',fontsize=20)
ax[0].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['SVM test acc'],linewidth=4,color='b',linestyle='-',label='SVM training',ci=95,ax=ax[1])
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['SVM train acc'],linewidth=4,color='b',linestyle='--',label='SVM testing',ci=95,ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_xticks(X)
ax[1].tick_params(axis='both', which='both', labelsize=20)
ax[1].legend(loc='lower right',fontsize=20)
ax[1].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['TREE test acc'],linewidth=4,color='g',linestyle='-',label='Decision Tree training',ci=95,ax=ax[2])
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['TREE train acc'],linewidth=4,color='g',linestyle='--',label='Decision Tree testing',ci=95,ax=ax[2])
ax[2].set_ylabel('')
ax[2].set_xticks(X)
ax[2].tick_params(axis='both', which='both', labelsize=20)
ax[2].legend(loc='lower right',fontsize=20)
ax[2].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['NN test acc'],linewidth=4,color='k',linestyle='-',label='Neural Network training',ci=95,ax=ax[3])
sns.lineplot(x=df['PLC Amplitude(mA)'],y=df['NN train acc'],linewidth=4,color='k',linestyle='--',label='Neural Network testing',ci=95,ax=ax[3])
ax[3].set_ylabel('')
ax[3].set_xticks(X)
ax[3].set_xlabel('PLC signal amplitude (mA)',fontsize=24)
ax[3].tick_params(axis='both', which='both', labelsize=20)
ax[3].legend(loc='lower right',fontsize=20)
ax[3].grid(linestyle=':',linewidth=2,alpha=0.5)



