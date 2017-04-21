"""
short module to give a R-like fit+residual plot function  
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

def residual_plot(datax,datay,err,fit_funtion):
    
    res = datay-fit_funtion(datax) #calculate the residuals
    
    #generate fit line
    x = np.linspace(min(datax)-100,max(datax)+100,10000) #10000 points for line 
    y = fit_funtion(x)

    #create the the grid that the fits and residuals will be show in.
    grid = gridspec.GridSpec(2, 1,
                             width_ratios=[1,1],
                             height_ratios=[4,1],
                             wspace=0.0, hspace=0.0) 
    #first the trend line
    ax0 = plt.subplot(grid[0]) #pick the larger area plot for fit plot
    ax0.set_xticklabels([]) #going to use a common axis so remove the x labels
    plt.ylabel(r'$\rho$ (cm)',fontsize=20)
    plt.errorbar(datax,datay,yerr=err,marker='o',linestyle='None') #plot data with just errors and points
    plt.plot(x,y) #fit line
    plt.title('Fit and Residuals') #go ahead and add title so it is on the top subplot
    #next the residuals
    ax1 = plt.subplot(grid[1]) #switch to smaller subplot
    plt.errorbar(datax,res,yerr=err,marker='o',linestyle='None') #again just points no lines
    plt.ylabel('Residuals',fontsize=20) 
    plt.xlabel('Channel #',fontsize=20)
    plt.axhline(y=0,linewidth=2, color='k',linestyle='--') #draw a horizontal line at 0 to guide the eye
    #these three list comprehensions set the dashed border of the residual subplot
    [j.set_linewidth(3.0) for j in ax1.spines.itervalues()]
    [j.set_color('k') for j in ax1.spines.itervalues()]
    [j.set_linestyle('--') for j in ax1.spines.itervalues()]


    
