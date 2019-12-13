import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import matplotlib.colors as cl

data=pd.read_csv('C:\\Users\HP\Desktop\Ebooks\Work\IT Projects\Data Science\Python\Data Viz Project\Economist_Assignment_Data.csv')
xdata=np.array(data)

z=xdata[:,5].copy()
for x in range(0,173):
	if z[x]=='SSA':
		z[x]=0
	elif z[x]=='Americas':
		z[x]=1
	elif z[x]== 'MENA':
		z[x]=2
	elif z[x]=='Asia Pacific':
		z[x]=3
	elif z[x]=='East EU Cemt Asia':
		z[x]=4
	else : z[x]=5


cMap=cl.ListedColormap(['Purple','Blue','Green','Yellow','Orange','Red'])
a=plt.scatter(data.CPI,data.HDI,c=z,cmap=cMap,alpha=0.6)
plt.title('Corruption and Human Development')
plt.xlabel('Corruption Perception Index')
plt.ylabel('Human Development Index')
cbar=plt.colorbar()
cbar.set_label('Region', rotation = 90)
cbar.ax.set_yticklabels(['SSA','Americas','MENA','Asia Pacific','East EU cemt Asia','EU W. Europe',' ',' '])
plt.show()

