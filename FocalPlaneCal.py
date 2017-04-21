import re
import numpy as np
from scipy import optimize
import pandas as pd
from Residual_Plot import residual_plot
import matplotlib.pyplot as plt
"""
Program is intended for use with TUNL Enge Splitpole.
Using NNDC 2012 mass evaluation for masses as Q-Values.

Caleb Marshall, TUNL/NCSU, 2017
"""

#convert to MeV/c^2
u_convert = 931.49409

#read in the mass table with the format provided
mass_table = pd.read_csv('pretty.mas12',sep='\s+') #I altered the file itself to make it easier to parse

#once again the chi_square objective function
def chi_square(poly,x_rho,x_channel,unc):
    poly = np.poly1d(poly)
    x_theory = poly(x_channel)
    temp = .5*((x_theory-x_rho)/unc)**2.0
    return np.sum(temp)
 
 
#gather all the nuclei data into a class         
class Nuclei():

    def __init__(self, name):
        #parse the input string
        self.A = int(re.split('\D+',name)[0]) 
        self.El = re.split('\d+',name)[1]
        self.get_mass_charge()
        
        
    #searches the mass table for the given isotope
    def get_mass_charge(self,table=mass_table):
        self.m = table[(table.El== self.El) & (table.A==self.A)]['Mass'].values*u_convert
        self.dm = table[(table.El == self.El) & (table.A==self.A)]['Mass_Unc'].values*u_convert
        self.Z = table[(table.El == self.El) & (table.A==self.A)]['Z'].values
        
    #just if you want to check data quickly
    def __call__(self):
        print 'Nuclei is '+str(self.A)+str(self.El)+'\n'
        print 'Mass =',self.m,'+/-',self.dm 


#class that handles all the kinematics
class Reaction():

    def __init__(self,a,A,b,B,B_field,E_lab,E_lab_unc):
        """
        Parse reaction names,looks up there masses, and calculates Q-value(MeV) 
        E_lab = MeV
        B_field = Tesla
        """
        self.a = Nuclei(a)
        self.A = Nuclei(A)
        self.b = Nuclei(b)
        self.B = Nuclei(B)
        self.Q = ((self.a.m+self.A.m)-(self.b.m+self.B.m))
        self.dQ = np.sqrt(self.a.dm**2+self.A.dm**2+self.b.dm**2+self.B.dm**2) #using quadrature
        self.B_field = B_field #mag field
        self.q = self.b.Z #charge of projectile
        self.E_lab = E_lab
        self.E_lab_unc = E_lab_unc
        
        
        
    def calc_energy(self,E_level,theta,E_level_unc):
        
        #Based on Equation C.5 & C.6 for Christian's book.
        theta = theta*(np.pi/180.0) #to radians 
        r = (np.sqrt(self.a.m*self.b.m*self.E_lab)/(self.b.m+self.B.m))*np.cos(theta)
        s = (self.E_lab*(self.B.m-self.a.m)+self.B.m*(self.Q-E_level))/(self.b.m+self.B.m)
        Eb =  (r + np.sqrt(r**2.0+s))**2.0 #We only care about positive solutions
        
        #Check if solution is real.
        if (np.isnan(Eb)):
            print "Reaction is energetically forbidden!!"
            return 0.0

        #going with the assumption(for now) that the uncertainty is dominated by Q values and beam
        Eb_unc = np.sqrt(self.dQ**2+self.E_lab_unc**2+E_level_unc**2)
        
        
        #Return the positive solutions squared. 
        return Eb,Eb_unc

    
        

class Focal_Plane_Fit():

    def __init__(self):

        self.reactions = {}
        self.theta = float(raw_input('What is the lab angle? \n'))#making the assumption this points are all from the same theta 
        #points is a list of dictionaries with rho,channel entry structure. Each of those has a value/uncertainty component.
        self.points = []
        self.fits = {}
        
    def add_reaction(self):
        #take user input for reaction
        a = str(raw_input('Enter projectile \n'))
        A = str(raw_input('Enter target \n'))
        b = str(raw_input('Enter detected particle \n'))
        B = str(raw_input('Enter residual particle \n'))
        B_field = float(raw_input('What was the B field setting? \n'))
        E_lab = float(raw_input('Beam energy? \n'))
        E_lab_unc = float(raw_input('Beam energy uncertainty? \n'))
        print 'Reaction',len(self.reactions.keys()),'has been defined as '+a+' + '+A+' -> '+B+' + '+b
        print 'E_beam =',E_lab,'+/- MeV',E_lab_unc,'With B-Field',B_field,'T' 
        self.reactions[len(self.reactions.keys())] = Reaction(a,A,b,B,B_field,E_lab,E_lab_unc) #keys for dictionary go from 0,..,n 

        
    def add_point(self):
        reaction = int(raw_input('Which reaction(0...n)? \n'))
        channel = float(raw_input('Enter the peak channel number. \n'))
        channel_unc = float(raw_input('What is the centroid uncertainty? \n'))
        level = float(raw_input('Enter the peak energy (MeV). \n'))
        level_unc = float(raw_input('Enter the peak uncertainty (MeV). \n'))
        rho, rho_unc = self.calc_rho(reaction,level,level_unc)
        rho = {'value':rho,'unc':rho_unc}
        channel = {'value':channel,'unc':channel_unc}
        point = {'rho':rho,'channel':channel}
        self.points.append(point)
        
    def calc_rho(self,reaction,E_level,E_level_unc):
        reaction = self.reactions[reaction] #just for short hand
        Eb,Eb_unc = reaction.calc_energy(E_level,self.theta,E_level_unc)
        p = np.sqrt(2.0*reaction.b.m*Eb+Eb**2.0) #"Relativistic" but Eb is not calculated that way  
        rho = p/(reaction.b.Z*reaction.B_field)*.33356 #enge's paper not sure about this
        rho_unc = np.sqrt((reaction.b.m/p*1.0/reaction.B_field)**2.0*Eb_unc**2.0 +
                          (Eb/p*1.0/reaction.B_field)**2.0*reaction.b.dm**2.0)
        print rho,rho_unc
        return rho,rho_unc
     
    #chi square fit 
    def fit(self,order=1):
        N = len(self.points) # Number of data points
        if N > (order+1): #check to see if we have n+2 points where n is the fit order
            print "Using a fit of order",order
            x_rho = np.zeros(N) #just decided to start with arrays. maybe dumb
            x_channel = np.zeros(N)
            x_unc = np.zeros(N)
            coeff = np.ones(order+1) #these are the n+1 fit coefficients for the polynomial fit
            for i,point in enumerate(self.points):
                #collect the different values needed for chi_square fit
                x_rho[i] = point['rho']['value']
                x_channel[i] = point['channel']['value']
                x_unc[i] = np.sqrt(point['rho']['unc']**2)
                
            #now scale channel points
            channel_mu = np.sum(x_channel)/float(N) #scale will be average of all calibration peaks
            x_channel = x_channel-channel_mu
            sol = optimize.minimize(chi_square,coeff,args=(x_rho,x_channel,x_unc),method='Nelder-Mead')
            #tell myself what I did
            chi2 = sol.fun/(N-(order+1))
            print "Chi Square is", chi2
            print "Fit parameters are (from highest order term to lowest)",sol.x
            self.fits[order] = np.poly1d(sol.x) #add to dictionary the polynomial object
            print "Fit stored in member variable fits[%d]" %order
            #create the a plot showing the fit and its residuals
            residual_plot(x_channel,x_rho,x_unc,self.fits[order])
            plt.text(max(x_channel-30),max(x_rho),r"$\chi^2=$",fontsize=20) #just print chi_square value on plot
            plt.show()
            #right now automatically doing higher order fits
            if order < 3:
                self.fit(order=(order+1))
                
        else:
            print "Not enough points to preform fit of order",order,"or higher"
            
    #finally given channel number use a fit to give energy         
    def peak_energy(self): 
        channel = float(raw_input("Enter channel number."))
        channer_unc = float(raw_input("Enter channel uncertainty."))
        if len(self.fits()) > 1:
            fit = int(raw_input("Which fit?"))
        else:
            fit = 1
        print "Calibrated energy is:",self.fits[fit](channel)
