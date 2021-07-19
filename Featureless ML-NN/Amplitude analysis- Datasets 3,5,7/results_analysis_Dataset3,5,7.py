import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('classic')

# Uploading the dataset, creating dataframe (using pandas)
df_raw = pd.read_csv('Datasets3,5,7_TrainTest_Results_combined_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_raw_dropna = df_raw.dropna(axis='index')
df_raw_dropna_confmat = df_raw_dropna.iloc[:,:-1]
df = df_raw_dropna_confmat

df[53] = df[53].str.replace(']','')
df[53] = pd.to_numeric(df[53])
 
df_timeseries = df[df[0] == 'time_series']
df_magspec = df[df[0] == 'mag_spectra']
df_recspec = df[df[0] == 'real_imag_spectra']

X=np.unique(np.array(df_magspec[1]))

# Accuracy
plt.figure()
sns.lineplot(x=df_magspec[1],y=df_magspec[2],linewidth=4,color='r',linestyle='--',label='Magnitude spectrogram',ci=95)
sns.lineplot(x=df_recspec[1],y=df_recspec[2],linewidth=4,color='b',label='Rectangular spectrogram',ci=95)
sns.lineplot(x=df_timeseries[1],y=df_timeseries[2],linewidth=8,color='g',linestyle=':',label='Time-series',ci=95)
plt.xlabel('PLC signal amplitude (mA) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0,1,21),fontsize=20)
plt.ylim(0.4,1.05)
plt.legend(loc='center right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

df_zoom = df[df[1]>=250]
X_zoom = np.unique(np.array(df_zoom[1]))
plt.figure()
sns.lineplot(x=df_magspec[1],y=df_magspec[206],linewidth=4,color='r',linestyle='--',label='Magnitude spectrogram',ci=95)
sns.lineplot(x=df_recspec[1],y=df_recspec[206],linewidth=4,color='b',label='Rectangular spectrogram',ci=95)
sns.lineplot(x=df_timeseries[1],y=df_timeseries[206],linewidth=8,color='g',linestyle=':',label='Time-series',ci=95)
plt.xticks(X_zoom,fontsize=20)
plt.xlim(250,1000)
plt.yticks(np.linspace(0.9,1,11),fontsize=20)
plt.ylim(0.9,1.0)
plt.legend(loc='lower right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)


# Precision, Recall, F1
# Plotting only F1 here
plt.figure()
sns.lineplot(x=df_magspec[1],y=df_magspec[206],linewidth=4,color='r',linestyle='--',label='Magnitude spectrogram',ci=95)
sns.lineplot(x=df_recspec[1],y=df_recspec[206],linewidth=4,color='b',label='Rectangular spectrogram',ci=95)
sns.lineplot(x=df_timeseries[1],y=df_timeseries[206],linewidth=8,color='g',linestyle=':',label='Time-series',ci=95)
plt.xlabel('PLC signal amplitude (mA) ',fontsize=24)
plt.ylabel('F1 score',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0,1,21),fontsize=20)
plt.ylim(0.3,1.05)
plt.legend(loc='lower right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)


# Train and test accuracy subplots
fig,ax = plt.subplots(3,1,sharex=True,sharey=True)

sns.lineplot(x=df_magspec[1],y=df_magspec[53],linewidth=4,color='r',linestyle='-',label='Magnitude spec Training',ci=95,ax=ax[0])
sns.lineplot(x=df_magspec[1],y=df_magspec[2],linewidth=4,color='r',linestyle='--',label='Magnitude spec Testing',ci=95,ax=ax[0])
ax[0].set_ylabel('')
ax[0].set_ylim(0.4,1.05)
ax[0].set_xticks(X)
ax[0].tick_params(axis='both', which='both', labelsize=20)
ax[0].legend(loc='lower right',fontsize=20)
ax[0].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df_recspec[1],y=df_recspec[53],linewidth=4,color='b',linestyle='-',label='Rectangular spec training',ci=95,ax=ax[1])
sns.lineplot(x=df_recspec[1],y=df_recspec[2],linewidth=4,color='b',linestyle='--',label='Rectangular spec testing',ci=95,ax=ax[1])
ax[1].set_ylabel('')
ax[0].set_ylim(0.4,1.05)
ax[1].set_xticks(X)
ax[1].tick_params(axis='both', which='both', labelsize=20)
ax[1].legend(loc='lower right',fontsize=20)
ax[1].grid(linestyle=':',linewidth=2,alpha=0.5)

sns.lineplot(x=df_timeseries[1],y=df_timeseries[53],linewidth=4,color='g',linestyle='-',label='Time series training',ci=95,ax=ax[2])
sns.lineplot(x=df_timeseries[1],y=df_timeseries[2],linewidth=4,color='g',linestyle='--',label='Time series testing',ci=95,ax=ax[2])
ax[2].set_ylabel('')
ax[0].set_ylim(0.4,1.05)
ax[2].set_xticks(X)
ax[2].tick_params(axis='both', which='both', labelsize=20)
ax[2].legend(loc='upper right',fontsize=20)
ax[2].grid(linestyle=':',linewidth=2,alpha=0.5)

