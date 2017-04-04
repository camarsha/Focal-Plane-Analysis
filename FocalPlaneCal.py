import re
import numpy as np
from scipy import optimize
import pandas as pd

"""
Program is intended for use with TUNL Enge Splitpole.
Using NNDC 2012 mass evaluation for masses as Q-Values.

Caleb Marshall 2017
"""

#convert to MeV/c^2
u_convert = 931.494095

#read in the mass table with the format provided
mass_table = pd.read_csv('pretty.mas12',sep='\s+') #I altered the file itself to make it easier to parse



#gather all the nuclei data into a class         
class Nuclei():

    def __init__(self, name):
        #parse the input string
        self.A = int(re.split('\D+',name)[0]) 
        self.El = re.split('\d+',name)[1]
        self.get_mass_charge()
        
        
    #searches the mass table for the given isotope
    def get_mass_charge(self,table=mass_table):
        self.m = table[(table.El== self.El) & (table.A==self.A)]['Mass'].values*u_convert #convert to MeV from KeV
        self.dm = table[(table.El == self.El) & (table.A==self.A)]['Mass_Unc'].values*u_convert
        self.Z = table[(table.El == self.El) & (table.A==self.A)]['Z'].values
        
    #just if you want to check data quickly
    def __call__(self):
        print 'Nuclei is '+str(self.A)+str(self.El)+'\n'
        print 'Mass =',self.m,'+/-',self.dm 



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
        self.points = []
        
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
    def fit(self):
        pass
        
    

#Define the reaction of interest and other parameters here d+var stands for uncertainty
E_beam = 19.917 #MeV 
dE_beam = .005 #Mev




