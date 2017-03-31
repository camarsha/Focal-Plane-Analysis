import re
import numpy as np
from scipy import optimize
import pandas as pd

"""
Program is intended for use with TUNL Enge Splitpole.
Using NNDC 2012 mass evaluation for masses as Q-Values.

Caleb Marshall 2017
"""

#read in the mass table with the format provided
mass_table = pd.read_csv('pretty.mas12',sep='\s+') #I altered the file itself to make it easier to parse


#gather all the nuclei data into a class         
class Nuclei():

    def __init__(self, name):
        #parse the input string
        self.A = int(re.split('\D+',name)[0]) 
        self.El = re.split('\d+',name)[1]
        self.m,self.dm = self.get_mass()
        
    #searches the mass table for the given isotope
    def get_mass(self,table=mass_table):
        m = table[(table.El== self.El) & (table.A==self.A)]['Mass'].values/1000.0 #convert to MeV from KeV
        dm = table[(table.El == self.El) & (table.A==self.A)]['Mass_Unc'].values/1000.0
        
        return m,dm

    #just if you want to check data quickly
    def __call__(self):
        print 'Nuclei is '+str(self.A)+str(self.El)+'\n'
        print 'Mass =',self.m,'+/-',self.dm 



class Reaction():

    def __init__(self,a,A,b,B):
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
        
    def calc_energy(self,E_lab,theta,E_level=0.0,E_lab_err=0.0):
        
        #Based on Equation C.5 & C.6 for Christian's book.
        theta = theta*(np.pi/180.0) #to radians 
        r = (np.sqrt(self.a.m*self.b.m*E_lab)/(self.b.m+self.B.m))*np.cos(theta)
        s = (E_lab*(self.B.m-self.a.m)+self.B.m*(self.Q-E_level))/(self.b.m+self.B.m)
        Eb =  (r + np.sqrt(r**2.0+s))**2.0 #We only care about positive solutions since theta < 90
        
        #Check if solution is real.
        if (np.isnan(root_Eb)):
            print "Reaction is energetically forbidden!!"
            return 0.0
        
        #Return the positive solutions squared. 
        return Eb
        
        

class Focal_Plane_Fit():

    def __init__(self):

        self.reactions = {}

    def add_reaction(self):
        #take user input for reaction
        a = str(raw_input('Enter projectile \n'))
        A = str(raw_input('Enter target \n'))
        b = str(raw_input('Enter detected particle \n'))
        B = str(raw_input('Enter residual particle \n'))
        print 'Reaction',len(self.reactions.keys()),'has been defined as '+a+' + '+A+' -> '+B+' + '+b 
        self.reactions[len(self.reactions.keys())] = Reaction(a,A,b,B) #keys for dictionary go from 0,..,n 

        
    # def add_point(self):

    #def calc_rho(self,E_level,B_field,E_level_Err,B_field_Err):

#Define the reaction of interest and other parameters here d+var stands for uncertainty
E_beam = 19.917 #MeV 
dE_beam = .005 #Mev




