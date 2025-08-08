from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from scipy.io import loadmat
import torch

config ={
"font.family":'serif',
"mathtext.fontset":'stix',
"font.serif":['simsun'],
}
rcParams.update(config)

real=np.load('feature_r_1-4_Obs2.npy')
n_s=len(real)
imag=np.load('feature_i_1-4_Obs2.npy')
n_t=len(imag)

# feature=np.concatenate([real[0:105*1,:],real[105*1:164*2,:],real[164*2:164*9,:],real[164*9:1637,:]],axis=0)
tsne=TSNE(perplexity=40,n_components=2,n_iter=5000,init='pca')
embs=tsne.fit_transform(imag)

# np.save('embs.npy',embs)
embs_1=embs[0:1090,:]
embs_2=embs[1090:,:]

fig=plt.figure()
ax=plt.gca()

plt.scatter(embs_1[:,0],embs_1[:,1],s=4,label='Representation of Normal degradation stage')
plt.scatter(embs_2[:,0],embs_2[:,1],s=4,label='Representation of Normal degradation stage')
plt.xlabel('1st PC',fontproperties = 'Times New Roman')
# plt.xlabel('第一主成分',fontproperties = 'Times New Roman')
plt.ylabel('2nd PC',fontproperties = 'Times New Roman')
plt.xticks(fontproperties = 'Times New Roman')
plt.yticks(fontproperties = 'Times New Roman')
plt.title('Bearing 1-4 (IMAG)',fontproperties = 'Times New Roman')
# plt.title('Bearing 1-4 (虚部)',fontproperties = 'Times New Roman')
plt.legend(loc='lower right',prop={"family": "Times New Roman"})

xminorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(xminorLocator)
yminorLocator = MultipleLocator(5)
ax.yaxis.set_minor_locator(yminorLocator)
plt.tight_layout()
plt.show()
fig.savefig(r'C:\Users\caoyu\Desktop\1-4_i_Obs2.svg', transparent=True)
# embs_1=embs[:105*1,:]
# embs_2=embs[105*1:164*2,:]
# embs_3=embs[164*2:164*9,:]
# embs_4=embs[164*9:1637,:]

# fig=plt.figure()
# ax=plt.gca()
# plt.scatter(embs_1[:,0],embs_1[:,1],s=4,label='Real feature in Period I of Bearing3-2')
# plt.scatter(embs_2[:,0],embs_2[:,1],s=4,label='Real feature in Period II of Bearing3-2')
# plt.scatter(embs_3[:,0],embs_3[:,1],s=4,label='Real feature in Period III of Bearing3-2')
# plt.scatter(embs_4[:,0],embs_4[:,1],s=4,label='Real feature in Period IV of Bearing3-2')
# plt.xticks(fontproperties = 'Times New Roman')
# plt.yticks(fontproperties = 'Times New Roman')
# plt.legend(loc='upper right',prop={"family": "Times New Roman"})

# xminorLocator = MultipleLocator(5)
# ax.xaxis.set_minor_locator(xminorLocator)
# yminorLocator = MultipleLocator(5)
# ax.yaxis.set_minor_locator(yminorLocator)
# plt.show()
# # fig.savefig('3-2real.svg', transparent=True)

# feature_r=np.concatenate([real[0:1090,:],real[1090:,:]],axis=0)
# tsne=TSNE(perplexity=40,n_components=2,n_iter=5000,init='pca')
# embs=tsne.fit_transform(real)

# # np.save('embs.npy',embs)

# embs_1=embs[0:1090,:]
# embs_2=embs[1090:,:]


# # embs_3=embs[240:1430,:]
# # embs_4=embs[1430:1505,:]
# # embs_5=embs[1505:1601,:]
# # embs_6=embs[1601:,:]

# fig=plt.figure()
# ax=plt.gca()

# plt.scatter(embs_1[:,0],embs_1[:,1],s=4,label='一般退化阶段的特征表示')
# plt.scatter(embs_2[:,0],embs_2[:,1],s=4,label='快速退化阶段的特征表示')
# # plt.scatter(embs_1[:,0],embs_1[:,1],s=4,label='早期退化阶段的特征表示')
# # plt.scatter(embs_2[:,0],embs_2[:,1],s=4,label='早期跑合阶段的特征表示')
# # plt.scatter(embs_3[:,0],embs_3[:,1],s=4,label='一般退化阶段的特征表示')
# # plt.scatter(embs_4[:,0],embs_4[:,1],s=4,label='快速退化阶段的特征表示')
# # plt.scatter(embs_5[:,0],embs_5[:,1],s=4,label='自愈阶段的特征表示')
# # plt.scatter(embs_6[:,0],embs_6[:,1],s=4,label='完全失效阶段的特征表示')
# plt.xlabel('第一主成分')
# # plt.xlabel('第一主成分',fontproperties = 'Times New Roman')
# plt.ylabel('第二主成分')
# plt.xticks(fontproperties = 'Times New Roman')
# plt.yticks(fontproperties = 'Times New Roman')
# plt.title('Bearing 1-4 (实部)')
# # plt.title('Bearing 1-4 (虚部)',fontproperties = 'Times New Roman')
# plt.legend(loc='lower right')

# xminorLocator = MultipleLocator(5)
# ax.xaxis.set_minor_locator(xminorLocator)
# yminorLocator = MultipleLocator(5)
# ax.yaxis.set_minor_locator(yminorLocator)
# plt.tight_layout()
# plt.show()
# fig.savefig(r'C:\Users\caoyu\Desktop\1-4_r_Obs2.svg', transparent=True)