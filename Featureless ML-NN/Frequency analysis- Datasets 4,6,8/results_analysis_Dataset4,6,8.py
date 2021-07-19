import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('classic')
# =================================================================================================================================
#%%
# Case 1
# Uploading the dataset, creating dataframe (using pandas)
df_raw = pd.read_csv('Dataset4,6,8_comp_Case1_Results_combined_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df = df_raw

df[53] = df[53].str.replace(']','')
df[53] = pd.to_numeric(df[53])

df_timeseries = df[df[0] == 'time_series']
df_magspec = df[df[0] == 'mag_spectra']
df_recspec = df[df[0] == 'real_imag_spectra']

X=np.unique(np.array(df_magspec[1]))

# Accuracy
plt.figure()
sns.lineplot(x=df_magspec[1],y=df_magspec[2],linewidth=4,color='r',linestyle='--',label='Magnitude spectrogram',ci=95)
sns.lineplot(x=df_recspec[1],y=df_recspec[2],linewidth=4,color='b',linestyle='-',label='Rectangular spectrogram',ci=95)
sns.lineplot(x=df_timeseries[1],y=df_timeseries[2],linewidth=8,color='g',linestyle=':',label='Time-series',ci=95)
plt.xlabel('PLC signal frequency (Hz) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0.4,1,21),fontsize=20)
plt.ylim(0.4,1.001)
plt.xlim(680,2020)
plt.legend(loc='upper right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

# Precision, Recall, F1
# Plotting only F1 here
plt.figure()
sns.lineplot(x=df_magspec[1],y=df_magspec[206],linewidth=4,color='r',label='Magnitude spectrogram',ci=95)
sns.lineplot(x=df_recspec[1],y=df_recspec[206],linewidth=4,color='b',label='Rectangular spectrogram',ci=95)
sns.lineplot(x=df_timeseries[1],y=df_timeseries[206],linewidth=4,color='g',label='Time-series',ci=95)
plt.xlabel('PLC signal frequency (Hz) ',fontsize=24)
plt.ylabel('F1 Score',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0.4,1,21),fontsize=20)
plt.ylim(0.4,1.001)
plt.xlim(680,2020)
plt.legend(loc='upper right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

# Train and test accuracy subplots
fig,ax = plt.subplots(3,1,sharex=True,sharey=True)

sns.lineplot(x=df_magspec[1],y=df_magspec[53],linewidth=4,color='r',linestyle='-',label='Magnitude spec training',ci=95,ax=ax[0])
sns.lineplot(x=df_magspec[1],y=df_magspec[2],linewidth=4,color='r',linestyle='--',label='Magnitude spec testing',ci=95,ax=ax[0])
ax[0].set_ylabel('')
ax[0].set_ylim(0.4,1.05)
ax[0].set_xticks(X)
ax[0].tick_params(axis='both', which='both', labelsize=20)
ax[0].legend(loc='lower center',fontsize=20)
ax[0].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df_recspec[1],y=df_recspec[53],linewidth=4,color='b',linestyle='-',label='Rectangular spec training',ci=95,ax=ax[1])
sns.lineplot(x=df_recspec[1],y=df_recspec[2],linewidth=4,color='b',linestyle='--',label='Rectangular spec testing',ci=95,ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_ylim(0.4,1.05)
ax[1].set_xticks(X)
ax[1].tick_params(axis='both', which='both', labelsize=20)
ax[1].legend(loc='lower center',fontsize=20)
ax[1].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df_timeseries[1],y=df_timeseries[53],linewidth=4,color='g',linestyle='-',label='Time series training',ci=95,ax=ax[2])
sns.lineplot(x=df_timeseries[1],y=df_timeseries[2],linewidth=4,color='g',linestyle='--',label='Time series testing',ci=95,ax=ax[2])
ax[2].set_ylabel('')
ax[2].set_ylim(0.4,1.05)
ax[2].set_xticks(X)
ax[2].tick_params(axis='both', which='both', labelsize=20)
ax[2].legend(loc='upper right',fontsize=20)
ax[2].grid(linestyle=':',linewidth=2,alpha=0.5)

# =================================================================================================================================
#%%
# Case 2

# Uploading the NN dataset
df_raw = pd.read_csv('Dataset4,6,8_comp_Case2_Results_combined_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,207)))
df = df_raw
col_with_strings = [3,52,53,102,103,152,153,202]
for col in col_with_strings:
    df[col] = df[col].str.replace('[','')
    df[col] = df[col].str.replace(']','')
    df[col] = pd.to_numeric(df[col])

df_timeseries = df[df[0] == 'time_series']
df_magspec = df[df[0] == 'mag_spectra']
df_recspec = df[df[0] == 'real_imag_spectra']

TS_train_acc = df_timeseries[52]
MS_train_acc = df_magspec[52]
RS_train_acc = df_recspec[52]

TS_test_acc = df_timeseries[1]
MS_test_acc = df_magspec[1]
RS_test_acc = df_recspec[1]

TS_precision = df_timeseries[203]
MS_precision = df_magspec[203]
RS_precision = df_recspec[203]

TS_recall = df_timeseries[204]
MS_recall = df_magspec[204]
RS_recall = df_recspec[204]

TS_F1 = df_timeseries[205]
MS_F1 = df_magspec[205]
RS_F1 = df_recspec[205]

print('Time series:')
print('Train Score = %.5f +/- %.5f' %(np.mean(TS_train_acc),np.std(TS_train_acc)))
print('Test Score = %.5f +/- %.5f' %(np.mean(TS_test_acc),np.std(TS_test_acc)))
print('Precision = %.5f +/- %.5f' %(np.mean(TS_precision),np.std(TS_precision)))
print('Recall = %.5f +/- %.5f' %(np.mean(TS_recall),np.std(TS_recall)))
print('F1 score = %.5f +/- %.5f' %(np.mean(TS_F1),np.std(TS_F1)))
print('===========================================================================')
print('Magnitude spectrogram:')
print('Train Score = %.5f +/- %.5f' %(np.mean(MS_train_acc),np.std(MS_train_acc)))
print('Test Score = %.5f +/- %.5f' %(np.mean(MS_test_acc),np.std(MS_test_acc)))
print('Precision = %.5f +/- %.5f' %(np.mean(MS_precision),np.std(MS_precision)))
print('Recall = %.5f +/- %.5f' %(np.mean(MS_recall),np.std(MS_recall)))
print('F1 score = %.5f +/- %.5f' %(np.mean(MS_F1),np.std(MS_F1)))
print('===========================================================================')
print('Rectangular spectrogram:')
print('Train Score = %.5f +/- %.5f' %(np.mean(RS_train_acc),np.std(RS_train_acc)))
print('Test Score = %.5f +/- %.5f' %(np.mean(RS_test_acc),np.std(RS_test_acc)))
print('Precision = %.5f +/- %.5f' %(np.mean(RS_precision),np.std(RS_precision)))
print('Recall = %.5f +/- %.5f' %(np.mean(RS_recall),np.std(RS_recall)))
print('F1 score = %.5f +/- %.5f' %(np.mean(RS_F1),np.std(RS_F1)))

# =================================================================================================================================
#%%
# Case 3

# Uploading the NN dataset
df_raw = pd.read_csv('Dataset4,6,8_comp_Case3_Results_combined_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df = df_raw

df[23] = df[23].str.replace(']','')
df[23] = pd.to_numeric(df[23])

df_timeseries = df[df[0] == 'time_series']
df_magspec = df[df[0] == 'mag_spectra']
df_recspec = df[df[0] == 'real_imag_spectra']

X=np.unique(np.array(df_magspec[1]))

# Accuracy
plt.figure()
sns.lineplot(x=df_magspec[1],y=df_magspec[2],linewidth=4,color='r',linestyle='--',label='Magnitude spectrogram',ci=95)
sns.lineplot(x=df_recspec[1],y=df_recspec[2],linewidth=4,color='b',linestyle='-',label='Rectangular spectrogram',ci=95)
sns.lineplot(x=df_timeseries[1],y=df_timeseries[2],linewidth=4,color='g',linestyle=':',label='Time-series',ci=95)
plt.xlabel('PLC signal frequency (Hz) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0.4,1,21),fontsize=20)
plt.ylim(0.4,1.001)
plt.xlim(860,1720)
plt.legend(loc='upper right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)


# Train and test accuracy subplots
fig,ax = plt.subplots(3,1,sharex=True,sharey=True)

sns.lineplot(x=df_magspec[1],y=df_magspec[23],linewidth=4,color='r',linestyle='-',label='Magnitude spec training',ci=95,ax=ax[0])
sns.lineplot(x=df_magspec[1],y=df_magspec[2],linewidth=4,color='r',linestyle='--',label='Magnitude spec testing',ci=95,ax=ax[0])
ax[0].set_ylabel('')
ax[0].set_ylim(0.0,1.05)
ax[0].set_xticks(X)
ax[0].tick_params(axis='both', which='both', labelsize=20)
ax[0].legend(loc='lower center',fontsize=20)
ax[0].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df_recspec[1],y=df_recspec[23],linewidth=4,color='b',linestyle='-',label='Rectangular spec training',ci=95,ax=ax[1])
sns.lineplot(x=df_recspec[1],y=df_recspec[2],linewidth=4,color='b',linestyle='--',label='Rectangular spec testing',ci=95,ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_ylim(0.0,1.05)
ax[1].set_xticks(X)
ax[1].tick_params(axis='both', which='both', labelsize=20)
ax[1].legend(loc='lower center',fontsize=20)
ax[1].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df_timeseries[1],y=df_timeseries[23],linewidth=4,color='g',linestyle='-',label='Time series training',ci=95,ax=ax[2])
sns.lineplot(x=df_timeseries[1],y=df_timeseries[2],linewidth=4,color='g',linestyle='--',label='Time series testing',ci=95,ax=ax[2])
ax[2].set_ylabel('')
ax[2].set_xlabel('PLC Frequency(Hz)',fontsize=24)
ax[2].set_ylim(0.0,1.05)
ax[2].set_xticks(X)
ax[2].tick_params(axis='both', which='both', labelsize=20)
ax[2].legend(loc='lower center',fontsize=20)
ax[2].grid(linestyle=':',linewidth=2,alpha=0.5)
