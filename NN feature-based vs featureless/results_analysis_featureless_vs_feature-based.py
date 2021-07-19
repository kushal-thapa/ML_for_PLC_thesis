import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('classic')
#%%
# Amplitude analysis
# Uploading the dataset, creating dataframe (using pandas)
df_raw_featurebased = pd.read_csv('Amp_Dataset1_TrainTest_NN_Results_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_feature_dataset = df_raw_featurebased[df_raw_featurebased[0] == 'feature_dataset']

df_raw_featureless = pd.read_csv('Amp_Datasets3,5,7_TrainTest_magspectra_Results_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_featureless_dataset = df_raw_featureless[df_raw_featureless[0] == 'mag_spectra']

X = np.unique(np.array(df_feature_dataset[1]))

# Accuracy
plt.figure()
sns.lineplot(x=df_feature_dataset[1],y=df_feature_dataset[2],linewidth=4,color='r',linestyle='--',label='Feature-based NN',ci=100)
sns.lineplot(x=df_featureless_dataset[1],y=df_featureless_dataset[2],linewidth=4,color='b',label='Featureless NN',ci=100)
plt.xlabel('PLC signal amplitude (mA) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0,1,21),fontsize=20)
plt.ylim(0.4,1.05)
plt.legend(loc='center right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

df_zoom = df_feature_dataset[df_feature_dataset[1]>=250]
df_zoom_featureless = df_featureless_dataset[df_featureless_dataset[1]>=250]
X_zoom = np.unique(np.array(df_zoom[1]))
plt.figure()
sns.lineplot(x=df_zoom[1],y=df_zoom[2],linewidth=4,color='r',linestyle='--',ci=95)
sns.lineplot(x=df_zoom_featureless[1],y=df_zoom_featureless[2],linewidth=4,color='b',linestyle='-',ci=95)
plt.xticks(X_zoom,fontsize=20)
plt.yticks(np.linspace(0.9,1,11),fontsize=20)
plt.ylim(0.9,1)
#plt.legend(loc='center right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

#%%
# Frequency analysis
# Case 1
# Uploading the dataset, creating dataframe (using pandas)
df_raw_featurebased = pd.read_csv('Freq_Case1_Dataset2_comp_TrainTest_NN_Results_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_feature_dataset = df_raw_featurebased[df_raw_featurebased[0] == 'feature_dataset']

df_raw_featureless = pd.read_csv('Freq_Case1_Dataset4_comp_Results_magspectra_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_featureless_dataset = df_raw_featureless[df_raw_featureless[0] == 'mag_spectra']

X = np.unique(np.array(df_feature_dataset[1]))

# Accuracy
plt.figure()
sns.lineplot(x=df_feature_dataset[1],y=df_feature_dataset[2],linewidth=4,color='r',linestyle='--',label='Feature-based NN',ci=95)
sns.lineplot(x=df_featureless_dataset[1],y=df_featureless_dataset[2],linewidth=4,color='b',label='Featureless NN',ci=95)
plt.xlabel('PLC signal frequency (Hz) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0,1,21),fontsize=20)
plt.ylim(0.4,1.05)
plt.xlim(680,2020)
plt.legend(loc='upper right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

df_zoom = df_feature_dataset[(df_feature_dataset[1]>=930) & (df_feature_dataset[1]<=1650)]
df_zoom_featureless = df_featureless_dataset[(df_feature_dataset[1]>=930) & (df_feature_dataset[1]<=1650)]
X_zoom = np.unique(np.array(df_zoom[1]))
plt.figure()
sns.lineplot(x=df_zoom[1],y=df_zoom[2],linewidth=4,color='r',linestyle='--',ci=95)
sns.lineplot(x=df_zoom_featureless[1],y=df_zoom_featureless[2],linewidth=4,color='b',linestyle='-',ci=95)
plt.xticks(X_zoom,fontsize=20)
plt.yticks(np.linspace(0.9,1,11),fontsize=20)
plt.ylim(0.9,1)
plt.xlim(930,1650)
#plt.legend(loc='center right',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)

#%%
# Case 3

# Uploading the dataset, creating dataframe (using pandas)
df_raw_featurebased = pd.read_csv('Freq_Case3_Dataset2_comp_TrainTest_NN_Results_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_feature_dataset = df_raw_featurebased[df_raw_featurebased[0] == 'feature_dataset']

df_raw_featureless = pd.read_csv('Freq_Case3_Dataset4_comp_Results_magspectra_final.txt',sep=',', engine='python',index_col=False,names=list(range(0,208)))
df_featureless_dataset = df_raw_featureless[df_raw_featureless[0] == 'mag_spectra']

X = np.unique(np.array(df_feature_dataset[1]))

# Accuracy
plt.figure()
sns.lineplot(x=df_feature_dataset[1],y=df_feature_dataset[2],linewidth=4,color='r',linestyle='--',label='Feature-based NN',ci=100)
sns.lineplot(x=df_featureless_dataset[1],y=df_featureless_dataset[2],linewidth=4,color='b',label='Featureless NN',ci=100)
plt.xlabel('PLC signal amplitude (mA) ',fontsize=24)
plt.ylabel('Test accuracy',fontsize=24)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0,1,21),fontsize=20)
plt.ylim(0.4,1.05)
plt.xlim(860,1720)
plt.legend(loc='upper left',fontsize=20)
plt.grid(linestyle=':',linewidth=2,alpha=0.5)