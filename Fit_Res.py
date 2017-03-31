import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbs
import pandas as pd

#plots data points with error bars with a fit line and shows residuals
class FitPlot:

    def __init__(self,datax,datay,xerr,yerr,fitx,fity):
        
        #rest o' this crap is for the plotting
        self.fig = plt.figure(1)
        self.g1 = self.fig.add_axes((.1,.3,.8,.6))
        plt.errorbar(datax,datay,xerr,yerr,marker='o',linestyle='None')
        plt.plot(datax,fitline)
        
        self.g2 = self.fig.add_axes((.1,.1,.8,.2))
        



