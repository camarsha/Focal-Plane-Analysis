"""
Simple script to help visualize a fit between channel number and rho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs



#data to read in and fit parameters first
if __name__== "__main__":
    filename = '/home/caleb/Research/Longland/Experiments/23Na_24Mg/DataAnalysis/10_deg_data/calibration.dat'
    parms = np.array([0.0000000000381,0.0000000524,0.003616,76.7508]) #spanc values highest order term to lowest
    x_0 = 2491.1033 #spanc reference channel 

    
else:
    filename = raw_input("What is the filename? ") 
    x_0 = float(raw_input("What is the central channel? "))
    parms = []
    while True:
        val = raw_input("Starting with the highest order term enter fit parameters ")
        if val=='':break
        parms.append(float(val))
#now we have the fit function



    
#read in data

data = pd.read_csv(filename,delim_whitespace=True) #assumes we are given rho+err and channel+err



def fit(x,x_0,parms):
    x = x-x_0
    return np.polyval(parms,x)

#define interval for fit
stop = data.max()["Channel"]
start = data.min()["Channel"]
x = np.linspace(start,stop,num=100)
y = fit(x,x_0,parms)

r = []#list of residuals
#residuals
for i in data.index:
    z = (fit(data["Channel"][i],x_0,parms)-data["Rho"][i])
    r.append(z)

#plot data
#make the zero line for residuals
guide_line = np.zeros(x.size)

fig = plt.figure(1)
g1 = fig.add_axes((.1,.3,.8,.6))
g1.set_title(r'$10^{\circ}$ Fit',fontsize=20)
plt.errorbar(data["Channel"],data["Rho"],data["Channel_Err"],data["Rho_Err"],marker='o',linestyle='None')
plt.plot(x,y)
g1.set_xticklabels([])
g1.set_ylabel(r'$\rho}$',fontsize=20)
g2=fig.add_axes((.1,.1,.8,.2))   
plt.errorbar(data["Channel"],r,marker='o',linestyle='None')
labels = g2.get_yticks().tolist()
labels.pop()
g2.set_yticklabels(labels)
g2.set_ylabel('Residuals',fontsize=16)
#plt.plot(x,guide_line,'k--')
plt.axhline(y=0,linewidth=2, color='k',linestyle='--')
g2.set_xlabel('Channel Number',fontsize=16)
[i.set_linewidth(3.0) for i in g2.spines.itervalues()]
[i.set_color('k') for i in g2.spines.itervalues()]
[i.set_linestyle('--') for i in g2.spines.itervalues()]
plt.show()



