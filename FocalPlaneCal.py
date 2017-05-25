import re
import numpy as np
from scipy import optimize
import pandas as pd
from Residual_Plot import residual_plot
import matplotlib.pyplot as plt
import pymc as pm
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

#converts a standard deviation from a normal distribution to a log-normal sigma value
#mu is normal distribution mean, sig is its s.d
def log_normal_sigma(mu,sig):
    log_sig = np.sqrt(np.log(1.+(sig/mu)**2.)) #equation can be found on Wikipedia
    return log_sig

#converts the normal distribution mean to the log-normal mu
def log_normal_mu(mu,sig):
    log_mu = np.log(mu/(np.sqrt(1.+(sig/mu)**2.)))
    return log_mu

# for all values with uncertainty
def measured_value(x,dx):
    return { 'value' : x,
             'unc'   : dx}
               

#gather all the nuclei data into a class         
class Nuclei():

    def __init__(self, name):
        #parse the input string
        self.A = int(re.split('\D+',name)[0]) 
        self.El = re.split('\d+',name)[1]
        self.get_mass_charge()
        
        
    #searches the mass table for the given isotope
    def get_mass_charge(self,table=mass_table):
        m = table[(table.El== self.El) & (table.A==self.A)]['Mass'].values*u_convert
        dm = table[(table.El == self.El) & (table.A==self.A)]['Mass_Unc'].values*u_convert
        self.m = measured_value(m,dm)
        self.Z = table[(table.El == self.El) & (table.A==self.A)]['Z'].values
        
    #just if you want to check data quickly
    def __call__(self):
        print 'Nuclei is '+str(self.A)+str(self.El)+'\n'
        print 'Mass =',self.m,'+/-',self.dm 


#class that handles all the kinematics
class Reaction():

    def __init__(self,a,A,b,B,B_field,E_lab,E_lab_unc,theta):
        """
        Parse reaction names,looks up there masses, and calculates Q-value(MeV) 
        E_lab = MeV
        B_field = Tesla
        """
        self.a = Nuclei(a)
        self.A = Nuclei(A)
        self.b = Nuclei(b)
        self.B = Nuclei(B)
        __Q = ((self.a.m['value']+self.A.m['value'])-(self.b.m['value']+self.B.m['value']))
        __dQ = np.sqrt(self.a.m['unc']**2+self.A.m['unc']**2+self.b.m['unc']**2+self.B.m['unc']**2) #using quadrature
        self.Q = measured_value(__Q,__dQ)
        self.B_field = B_field #mag field
        self.q = self.b.Z #charge of projectile
        self.E_lab = measured_value(E_lab,E_lab_unc)
        self.theta = theta
        
        
        

class Focal_Plane_Fit():

    def __init__(self):

        self.reactions = {} 
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
        theta = float(raw_input('What is the lab angle? \n'))#making the assumption this points are all from the same theta
        self.reactions[len(self.reactions.keys())] = Reaction(a,A,b,B,B_field,E_lab,E_lab_unc,theta) #keys for dictionary go from 0,..,n 
        print 'Reaction',(len(self.reactions.keys())-1),'has been defined as '+a+' + '+A+' -> '+B+' + '+b
        print 'E_beam =',E_lab,'+/- MeV',E_lab_unc,'With B-Field',B_field,'T' 
        
    def add_point(self,reaction,level,level_unc,channel,channel_unc):
        rho,rho_unc,rho_trace = self.calc_rho(reaction,level,level_unc) #get rho and uncertainty
        rho = measured_value(rho,rho_unc) #convert to dict
        channel = measured_value(channel,channel_unc) 
        point = {'rho':rho,'channel':channel,'trace':rho_trace} 
        self.points.append(point)
        
    #add a calibration point which includes rho and an associated channel value    
    def input_point(self):
        reaction = int(raw_input('Which reaction(0...n)? \n'))
        channel = float(raw_input('Enter the peak channel number. \n'))
        channel_unc = float(raw_input('What is the centroid uncertainty? \n'))
        level = float(raw_input('Enter the peak energy (MeV). \n'))
        level_unc = float(raw_input('Enter the peak uncertainty (MeV). \n'))
        self.add_point(reaction,level,level_unc,channel,channel_unc)
    
        
    #added Monte Carlo error propagation for rho     
    def calc_rho(self,reaction,E_level,E_level_unc,steps=10000,burn=1000):
        reaction = self.reactions[reaction] #just for short hand pick out reaction
        E_level = pm.Normal('E_level',E_level,(1.0/E_level_unc)**2)#setup normal distribution for E_level
        reaction_variables = vars(reaction) #get variables from reaction
        normals = {} # dictionary for all quantities in our model
        normals["E_level"] = E_level #go ahead and add energy level
        #define distributions, all normal for now
        for var in reaction_variables:
            #test to see if value has uncertainty
            if type(reaction_variables[var]) == dict:                
                temp_mu = reaction_variables[var]['value']
                temp_sigma = reaction_variables[var]['unc']
                temp_normal = pm.Normal(var,temp_mu,(1.0/temp_sigma)**2) #tau = 1/sigma^2 var is name of variable
                normals[var] = temp_normal
            #special cases for masses    
            elif isinstance(reaction_variables[var],Nuclei):
                temp = reaction_variables[var].m
                temp_mu = temp['value']
                temp_sigma = temp['unc']
                temp_normal = pm.Normal(var,temp_mu,(1.0/temp_sigma)**2) 
                normals[var] = temp_normal
            else:
                normals[var] = reaction_variables[var] #these are just constant parameters
        #using pymc2 to give uncertainty on rho. This function accepts the sampled points and returns the rho value
        @pm.deterministic
        def rho_func(a=normals["a"], A=normals["A"], b=normals["b"], B=normals["B"],
                     E_level=normals["E_level"], theta=normals["theta"], q=normals["q"],
                     B_field=normals["B_field"],E_lab = normals["E_lab"],Q=normals["Q"]):
            #Based on Equation C.5 & C.6 for Christian's book.
            theta = theta*(np.pi/180.0) #to radians 
            r = (np.sqrt(a*b*E_lab)/(b+B))*np.cos(theta) 
            s = (E_lab*(B-a)+B*(Q-E_level))/(b+B)
            Eb =  (r + np.sqrt(r**2.0+s))**2.0 #We only care about positive solutions

            #Check if solution is real.
            if (np.isnan(Eb)):
                print "Reaction is energetically forbidden!!"
                return 0.0 

            p = np.sqrt(2.0*b*Eb+Eb**2.0) #"Relativistic" but Eb is not calculated that way  
            rho = p/(q*B_field)*.33356 #enge's paper not sure about this

            #Return the positive solutions squared. 
            return rho

        #define model
        model_variables = [i for i in normals.values() if isinstance(i,pm.Normal)] #get all the normal distributions
        model_variables.append(rho_func)
        model = pm.Model(model_variables)
        mcmc = pm.MCMC(model) #prepare the sampler
        mcmc.sample(steps,burn) #50000 samples with a 10000 burn in
        trace = mcmc.trace('rho_func')
        print 
        print "Sampling done, here are the stats"
        print "The 95% confidence interval is",mcmc.stats()['rho_func']['95% HPD interval']
        print "Mean:", mcmc.stats()['rho_func']['mean']
        print "Standard Deviation:", mcmc.stats()['rho_func']['standard deviation']
        return mcmc.stats()['rho_func']['mean'],mcmc.stats()['rho_func']['standard deviation'],trace #the stats we want
        
     
    #chi square fit will be good for quick cal 
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

    #now to try Bayesian fitting method          
    def bay_fit(self,order):
        #standard test to see if there are enough points
        N = len(self.points)
        if order < N-1:
            print "Not enough points to preform fit of order",order,"or higher"
        else:
            #get data
            x_obs = np.asarray([ele['channel']['value'] for ele in self.points])
            x_unc = np.asarray([ele['channel']['unc'] for ele in self.points])
            y_obs = np.asarray([ele['rho']['value'] for ele in self.points])
            y_unc = np.asarray([ele['rho']['unc'] for ele in self.points])
            #lets define priors for the polynomial terms
            priors = []
            for i in xrange(order+1):
                priors.append(pm.Uniform(str(i),0,100)) #uniform priors from 0 to 100

            #x and y likelihoods
            x = pm.Normal('x',x_obs,(1.0/x_unc)**2.0)
            y = pm.Normal('y',y_obs,(1.0/y_unc)**2.0)

          
    
    #finally given channel number use a fit to give energy         
    def peak_energy(self): 
        channel = float(raw_input("Enter channel number."))
        channer_unc = float(raw_input("Enter channel uncertainty."))
        if len(self.fits()) > 1:
            fit = int(raw_input("Which fit?"))
        else:
            fit = 1
        print "Calibrated energy is:",self.fits[fit](channel)

    #simple function to read in a file with calibration points and preform a fit on them    
    def read_calibration(self,cal_file,reaction=None):
        #can just pick, or ask for user input for reaction
        if type(reaction) != int:
            reaction = int(raw_input('Which reaction(0...n)? \n'))
        data = pd.read_csv(cal_file,sep='\s+')
        for i in data.index:
            level = data["level"][i]
            level_unc = data["level_unc"][i]
            channel = data["channel"][i]
            channel_unc = data["channel_unc"][i]
            self.add_point(reaction,level,level_unc,channel,channel_unc)
        self.fit()
