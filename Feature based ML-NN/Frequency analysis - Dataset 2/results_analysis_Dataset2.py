
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('classic')
# =================================================================================================================================
#%%
# Case 1
# Uploading the NN dataset
df_NN = pd.read_csv('Dataset2_comp_TrainTest_NN_Results_Case1_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))

NN_test_acc = np.array(df_NN[2])
NN_train_acc = (np.array(df_NN[53].str.replace(']',''))).astype(float)

# Uploading the dataset, creating dataframe (using pandas)
column_names= ['Dataset','PLC Freq(Hz)','LR train acc','LR test acc','SVM train acc','SVM test acc','TREE train acc','TREE test acc',
               'LR_precision','LR_recall','LR_f1','SVM_precision','SVM_recall','SVM_f1','TREE_precision','TREE_recall',
               'TREE_f1','LR_confmat','SVM_confmat','TREE_confmat']
df = pd.read_csv('Dataset2_comp_TrainTest_ML_Results_Case1_final.txt',sep=',', names=column_names, engine='python',index_col=False)
df['NN train acc'] = NN_train_acc
df['NN test acc'] = NN_test_acc

X = np.unique(np.array(df['PLC Freq(Hz)']))

# Accuracy
plt.figure()
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['LR test acc'],linewidth=4,color='r',linestyle='--',label='Logistic Regression',ci=95)#,err_style='bars')
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['SVM test acc'],linewidth=4,color='b',linestyle='-',label='SVM',ci=95)#,err_style='bars')
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['TREE test acc'],linewidth=8,color='g',linestyle=':',label='Decision Tree',ci=95)#,err_style='bars')
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['NN test acc'],linewidth=8,color='k',linestyle='-.',label='NN',ci=95)#,err_style='bars')
plt.xlabel('PLC Frequency (Hz) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0.5,1,21),fontsize=20)
plt.ylim(0.5,1)
plt.xlim(680,2020)
plt.legend(loc='upper right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

df_zoom = df[(df['PLC Freq(Hz)']>=930) & (df['PLC Freq(Hz)']<=1650)]
X_zoom = np.unique(np.array(df_zoom['PLC Freq(Hz)']))
plt.figure()
sns.lineplot(x=df_zoom['PLC Freq(Hz)'],y=df_zoom['LR test acc'],linewidth=4,color='r',linestyle='--',ci=95)
sns.lineplot(x=df_zoom['PLC Freq(Hz)'],y=df_zoom['SVM test acc'],linewidth=4,color='b',linestyle='-',ci=95)
sns.lineplot(x=df_zoom['PLC Freq(Hz)'],y=df_zoom['TREE test acc'],linewidth=8,color='g',linestyle=':',ci=95)
sns.lineplot(x=df_zoom['PLC Freq(Hz)'],y=df_zoom['NN test acc'],linewidth=8,color='k',linestyle='-.',ci=95)
plt.xticks(X_zoom,fontsize=20)
plt.yticks(np.linspace(0.9,1,11),fontsize=20)
plt.ylim(0.9,1)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

#---------------------------------------------------------------------------------------------------------------------------------------

# Train and test accuracy subplots
fig,ax = plt.subplots(4,1,sharex=True,sharey=True)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['LR train acc'],linewidth=4,color='r',linestyle='-',label='Logistic Regression Training',ci=95,ax=ax[0])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['LR test acc'],linewidth=4,color='r',linestyle='--',label='Logistic Regression Testing',ci=95,ax=ax[0])
ax[0].set_ylabel('')
ax[0].set_xticks(X)
ax[0].tick_params(axis='both', which='both', labelsize=20)
ax[0].legend(loc='lower center',fontsize=20)
ax[0].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['SVM test acc'],linewidth=4,color='b',linestyle='-',label='SVM training',ci=95,ax=ax[1])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['SVM train acc'],linewidth=4,color='b',linestyle='--',label='SVM testing',ci=95,ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_xticks(X)
ax[1].tick_params(axis='both', which='both', labelsize=20)
ax[1].legend(loc='lower center',fontsize=20)
ax[1].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['TREE test acc'],linewidth=4,color='g',linestyle='-',label='Decision Tree training',ci=95,ax=ax[2])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['TREE train acc'],linewidth=4,color='g',linestyle='--',label='Decision Tree testing',ci=95,ax=ax[2])
ax[2].set_ylabel('')
ax[2].set_xticks(X)
ax[2].tick_params(axis='both', which='both', labelsize=20)
ax[2].legend(loc='lower center',fontsize=20)
ax[2].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['NN test acc'],linewidth=4,color='k',linestyle='-',label='Neural Network training',ci=95,ax=ax[3])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['NN train acc'],linewidth=4,color='k',linestyle='--',label='Neural Network testing',ci=95,ax=ax[3])
ax[3].set_ylabel('')
ax[3].set_xticks(X)
ax[3].set_xlabel('PLC Frequency (Hz)',fontsize=24)
ax[3].tick_params(axis='both', which='both', labelsize=20)
ax[3].legend(loc='lower center',fontsize=20)
ax[3].grid(linestyle=':',linewidth=2,alpha=0.5)

#%%
# Case 2

# Uploading the NN dataset
df_raw = pd.read_csv('Dataset2_comp_TrainTest_NN_Results_Case2_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,207)))
df_NN = df_raw.dropna(axis='index')

NN_test_acc = np.array(df_NN[1])
NN_train_acc = (np.array(df_NN[52].str.replace(']',''))).astype(float)
NN_precision = np.array(df_NN[203])
NN_recall = np.array(df_NN[204])
NN_f1 = np.array(df_NN[205])

# Uploading the dataset, creating dataframe (using pandas)
column_names= ['Dataset','LR train acc','LR test acc','SVM train acc','SVM test acc','TREE train acc','TREE test acc',
               'LR_precision','LR_recall','LR_f1','SVM_precision','SVM_recall','SVM_f1','TREE_precision','TREE_recall',
               'TREE_f1','LR_confmat','SVM_confmat','TREE_confmat']
df = pd.read_csv('Dataset2_comp_TrainTest_ML_Results_Case2_final.txt',sep=',', names=column_names, engine='python',index_col=False)

df['NN train acc'] = NN_train_acc
df['NN test acc'] = NN_test_acc
df['NN_precision'] = NN_precision
df['NN_recall'] = NN_recall
df['NN_f1'] = NN_f1

train_df = df[['LR train acc','LR test acc','SVM train acc','SVM test acc','TREE train acc','TREE test acc','NN train acc','NN test acc']].copy()
plt.figure()
sns.boxplot(data=train_df,linewidth=2,fliersize=6)
plt.ylabel('Accuracy',fontsize=24)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

print('Logistic Regression:')
print('Train Score = %.5f +/- %.5f' %(np.mean(df['LR train acc']),np.std(df['LR train acc'])))
print('Test Score = %.5f +/- %.5f' %(np.mean(df['LR test acc']),np.std(df['LR test acc'])))
print('Precision = %.5f +/- %.5f' %(np.mean(df['LR_precision']),np.std(df['LR_precision'])))
print('Recall = %.5f +/- %.5f' %(np.mean(df['LR_recall']),np.std(df['LR_recall'])))
print('F1 score = %.5f +/- %.5f' %(np.mean(df['LR_f1']),np.std(df['LR_f1'])))
print('===========================================================================')
print('SVM:')
print('Train Score = %.5f +/- %.5f' %(np.mean(df['SVM train acc']),np.std(df['SVM train acc'])))
print('Test Score = %.5f +/- %.5f' %(np.mean(df['SVM test acc']),np.std(df['SVM test acc'])))
print('Precision = %.5f +/- %.5f' %(np.mean(df['SVM_precision']),np.std(df['SVM_precision'])))
print('Recall = %.5f +/- %.5f' %(np.mean(df['SVM_recall']),np.std(df['SVM_recall'])))
print('F1 score = %.5f +/- %.5f' %(np.mean(df['SVM_f1']),np.std(df['SVM_f1'])))
print('===========================================================================')
print('Decision Tree:')
print('Train Score = %.5f +/- %.5f' %(np.mean(df['TREE train acc']),np.std(df['TREE train acc'])))
print('Test Score = %.5f +/- %.5f' %(np.mean(df['TREE test acc']),np.std(df['TREE test acc'])))
print('Precision = %.5f +/- %.5f' %(np.mean(df['TREE_precision']),np.std(df['TREE_precision'])))
print('Recall = %.5f +/- %.5f' %(np.mean(df['TREE_recall']),np.std(df['TREE_recall'])))
print('F1 score = %.5f +/- %.5f' %(np.mean(df['TREE_f1']),np.std(df['TREE_f1'])))
print('===========================================================================')
print('Neural Network:')
print('Train Score = %.5f +/- %.5f' %(np.mean(df['NN train acc']),np.std(df['NN train acc'])))
print('Test Score = %.5f +/- %.5f' %(np.mean(df['NN test acc']),np.std(df['NN test acc'])))
print('Precision = %.5f +/- %.5f' %(np.mean(df['NN_precision']),np.std(df['NN_precision'])))
print('Recall = %.5f +/- %.5f' %(np.mean(df['NN_recall']),np.std(df['NN_recall'])))
print('F1 score = %.5f +/- %.5f' %(np.mean(df['NN_f1']),np.std(df['NN_f1'])))

# =================================================================================================================================
#%%
# Case 3

# Uploading the NN dataset
df_raw = pd.read_csv('Dataset2_old_TrainTest_NN_Results_Case3_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_NN = df_raw.dropna(axis='index')

NN_test_acc = np.array(df_NN[2])
NN_train_acc = (np.array(df_NN[53].str.replace(']',''))).astype(float)

# Uploading the dataset, creating dataframe (using pandas)
column_names= ['Dataset','PLC Freq(Hz)','LR train acc','LR test acc','SVM train acc','SVM test acc','TREE train acc','TREE test acc',
               'LR_precision','LR_recall','LR_f1','SVM_precision','SVM_recall','SVM_f1','TREE_precision','TREE_recall',
               'TREE_f1','LR_confmat','SVM_confmat','TREE_confmat']
df_full = pd.read_csv('Dataset2_comp_TrainTest_ML_Results_Case3_final.txt',sep=',', names=column_names, engine='python',index_col=False)
df = df_full.head(n=45)

df['NN train acc'] = NN_train_acc
df['NN test acc'] = NN_test_acc

X = np.unique(np.array(df['PLC Freq(Hz)']))

# Accuracy
plt.figure()
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['LR test acc'],linewidth=4,color='r',linestyle='--',label='Logistic Regression',ci=95)#,err_style='bars')
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['SVM test acc'],linewidth=4,color='b',linestyle='-',label='SVM',ci=95)#,err_style='bars')
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['TREE test acc'],linewidth=8,color='g',linestyle=':',label='Decision Tree',ci=95)#,err_style='bars')
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['NN test acc'],linewidth=8,color='k',linestyle='-.',label='NN',ci=95)#,err_style='bars')
plt.xlabel('PLC Frequency (Hz) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0.4,1,23),fontsize=20)
plt.ylim(0.4,1)
plt.xlim(860,1720)
plt.legend(loc='lower right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

#---------------------------------------------------------------------------------------------------------------------------------------
# Train and test accuracy subplots
fig,ax = plt.subplots(4,1,sharex=True,sharey=True)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['LR train acc'],linewidth=4,color='r',linestyle='-',label='Logistic Regression Training',ci=95,ax=ax[0])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['LR test acc'],linewidth=4,color='r',linestyle='--',label='Logistic Regression Testing',ci=95,ax=ax[0])
ax[0].set_ylabel('')
ax[0].set_xticks(X)
ax[0].tick_params(axis='both', which='both', labelsize=20)
ax[0].legend(loc='lower right',fontsize=15)
ax[0].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['SVM test acc'],linewidth=4,color='b',linestyle='-',label='SVM training',ci=95,ax=ax[1])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['SVM train acc'],linewidth=4,color='b',linestyle='--',label='SVM testing',ci=95,ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_xticks(X)
ax[1].tick_params(axis='both', which='both', labelsize=20)
ax[1].legend(loc='lower right',fontsize=15)
ax[1].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['TREE test acc'],linewidth=4,color='g',linestyle='-',label='Decision Tree training',ci=95,ax=ax[2])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['TREE train acc'],linewidth=4,color='g',linestyle='--',label='Decision Tree testing',ci=95,ax=ax[2])
ax[2].set_ylabel('')
ax[2].set_xticks(X)
ax[2].tick_params(axis='both', which='both', labelsize=20)
ax[2].legend(loc='lower right',fontsize=15)
ax[2].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df['PLC Freq(Hz)'],y=df['NN test acc'],linewidth=4,color='k',linestyle='-',label='Neural Network training',ci=95,ax=ax[3])
sns.lineplot(x=df['PLC Freq(Hz)'],y=df['NN train acc'],linewidth=4,color='k',linestyle='--',label='Neural Network testing',ci=95,ax=ax[3])
ax[3].set_ylabel('')
ax[3].set_xticks(X)
ax[3].set_xlabel('PLC Frequency (Hz)',fontsize=24)
ax[3].tick_params(axis='both', which='both', labelsize=20)
ax[3].legend(loc='lower right',fontsize=15)
ax[3].grid(linestyle=':',linewidth=2,alpha=0.5)

