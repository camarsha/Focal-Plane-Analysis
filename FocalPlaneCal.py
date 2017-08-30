import re
import numpy as np
from scipy import optimize
import pandas as pd
from Residual_Plot import residual_plot
import matplotlib.pyplot as plt
import pymc as pm
import string


"""
Program is intended for use with TUNL Enge Splitpole.
Using NNDC 2012 mass evaluation for masses as Q-Values.

Caleb Marshall, TUNL/NCSU, 2017
"""

# convert to MeV/c^2
u_convert = 931.4940954

# read in the mass table with the format provided
mass_table = pd.read_csv('pretty.mas12', sep='\s+')  # Pretty file.


# once again the chi_square objective function
def chi_square(poly, rho, channel, unc):
    poly = np.poly1d(poly)
    theory = poly(channel)
    temp = ((theory-rho)/unc)**2.0
    return np.sum(temp)


# for all values with uncertainty
def measured_value(x, dx):
    return {'value': x,
            'unc': dx}


# gather all the nuclei data into a class
class Nuclei():

    def __init__(self, name):
        # parse the input string
        self.name = name
        self.A = int(re.split('\D+', name)[0])
        self.El = re.split('\d+', name)[1]
        self.get_mass_charge()
       
    # searches the mass table for the given isotope
    def get_mass_charge(self, table=mass_table):
        m = table[(table.El == self.El) &
                  (table.A == self.A)]['Mass'].values*u_convert
        dm = table[(table.El == self.El)
                   & (table.A == self.A)]['Mass_Unc'].values*u_convert
        self.m = measured_value(m, dm)
        self.Z = table[(table.El == self.El) &
                       (table.A == self.A)]['Z'].values

    # just if you want to check data quickly
    def __call__(self):
        print 'Nuclei is '+str(self.A)+str(self.El)+'\n'
        print 'Mass =', self.m['value'], '+/-', self.m['unc']


# class that handles all the kinematics
class Reaction():

    def __init__(self, a, A, b, B, B_field,
                 E_lab, E_lab_unc, theta, mass_unc=False):
        """
        Parse reaction names,looks up there masses, and calculates Q-value(MeV)
        E_lab = MeV
        B_field = Tesla
        """
        self.a = Nuclei(a)
        self.A = Nuclei(A)
        self.b = Nuclei(b)
        self.B = Nuclei(B)
        __Q = ((self.a.m['value'] + self.A.m['value']) -
               (self.b.m['value'] + self.B.m['value']))
        __dQ = np.sqrt(self.a.m['unc']**2+self.A.m['unc']**2 +
                       self.b.m['unc']**2 + self.B.m['unc']**2)  # using quadrature
        self.Q = measured_value(__Q, __dQ)
        self.B_field = B_field  # mag field
        self.q = self.b.Z  # charge of projectile
        if E_lab_unc:
            self.E_lab = measured_value(E_lab, E_lab_unc)
        else:
            self.E_lab = E_lab
        self.theta = theta
        if not mass_unc:
            self.a.m['unc'] = 0.0
            self.A.m['unc'] = 0.0
            self.b.m['unc'] = 0.0
            self.B.m['unc'] = 0.0
            
    def name(self):
        print self.a.name+' + '+self.A.name+' -> '+self.B.name+' + '+self.b.name


class Focal_Plane_Fit():

    def __init__(self):

        self.reactions = {}
        # Points is a list of dictionaries with rho,channel entry structure.
        # Each of those has a value/uncertainty component.
        self.points = []
        self.fits = {}
        self.fits_bay = {}
        self.output_peaks = []

    def add_reaction(self):
        # take user input for reaction
        a = str(raw_input('Enter projectile \n'))
        A = str(raw_input('Enter target \n'))
        b = str(raw_input('Enter detected particle \n'))
        B = str(raw_input('Enter residual particle \n'))
        B_field = float(raw_input('What was the B field setting? \n'))
        E_lab = float(raw_input('Beam energy? \n'))
        E_lab_unc = float(raw_input('Beam energy uncertainty? \n'))
        theta = float(raw_input('What is the lab angle? \n')) # making the assumption this points are all from the same theta
        self.reactions[len(self.reactions.keys())] = Reaction(a, A, b, B,
                                                              B_field, E_lab, E_lab_unc, theta) #keys for dictionary go from 0,..,n 
        print 'Reaction', (len(self.reactions.keys())-1),'has been defined as '+a+' + '+A+' -> '+B+' + '+b
        print 'E_beam =', E_lab,'+/- MeV', E_lab_unc, 'With B-Field', B_field,'T' 

    def add_point(self, reaction, level,
                  level_unc, channel, channel_unc):
        rho, rho_unc, rho_trace = self.calc_rho(reaction,level,level_unc)  # get rho and uncertainty
        rho = measured_value(rho, rho_unc)  # convert to dict
        channel = measured_value(channel, channel_unc)
        point = {'rho': rho, 'channel': channel, 'trace': rho_trace}
        self.points.append(point)

    # add a calibration point which includes rho and an associated channel value
    def input_point(self):
        reaction = int(raw_input('Which reaction(0...n)? \n'))
        channel = float(raw_input('Enter the peak channel number. \n'))
        channel_unc = float(raw_input('What is the centroid uncertainty? \n'))
        level = float(raw_input('Enter the peak energy (MeV). \n'))
        level_unc = float(raw_input('Enter the peak uncertainty (MeV). \n'))
        self.add_point(reaction, level, level_unc, channel, channel_unc)

    def create_distributions(self,reaction):
        reaction_variables = vars(reaction) #get variables from reaction
        normals = {} # dictionary for all quantities in our model
        #define distributions, all normal for now
        #this loop is pulling variables directly from the reaction.__dict__ method 
        for var in reaction_variables:
            #test to see if value has uncertainty
            if type(reaction_variables[var]) == dict:                
                temp_mu = reaction_variables[var]['value']
                temp_sigma = reaction_variables[var]['unc']
                temp_normal = pm.Normal(var,temp_mu,(1.0/temp_sigma)**2.0) #tau = 1/sigma^2 var is name of variable
                normals[var] = temp_normal
            #special cases for masses    
            elif isinstance(reaction_variables[var],Nuclei):
                temp = reaction_variables[var].m
                temp_mu = temp['value']
                temp_sigma = temp['unc']
                if temp_sigma == 0.0:
                    normals[var] = temp_mu
                else:    
                    temp_normal = pm.Normal(var,temp_mu,(1.0/temp_sigma)**2.0) 
                    normals[var] = temp_normal
            else:
                normals[var] = reaction_variables[var] #these are just constant parameters
                
        return normals
    
    #added Monte Carlo error propagation for rho     
    def calc_rho(self,reaction,E_level,E_level_unc,steps=11000,burn=1000):
        reaction = self.reactions[reaction] #just for short hand pick out reaction
        #print reaction.name() #just make sure you know which reaction is being calculated
        E_level = pm.Normal('E_level',E_level,(1.0/E_level_unc)**2.0)#setup normal distribution for E_level
        normals = self.create_distributions(reaction)
        normals["E_level"] = E_level #go ahead and add energy level
        #using pymc2 to give uncertainty on rho. This function accepts the sampled points and returns the rho value
               
               
        @pm.deterministic
        def rho_func(a=normals["a"], A=normals["A"], b=normals["b"], B=normals["B"],
                     E_level=normals["E_level"], theta=normals["theta"], q=normals["q"],
                     B_field=normals["B_field"],E_lab = normals["E_lab"],Q=normals["Q"]):
            # formalism taken from http://skisickness.com/2010/04/25/
            theta = theta*(np.pi/180.0) #to radians 
            s = (A+a+E_lab)**2.0-(2*a*(E_lab)+(E_lab)**2.0) #relativistic invariant
            pcm = np.sqrt(((s-a**2.0-A**2.0)**2.0-(4.*a**2.0*A**2.0))/(4.*s)) # com p
            chi = np.log((pcm+np.sqrt(A**2.0+pcm**2.0))/A) # rapidity
            p_prime = np.sqrt(((s-b**2.0-(B+E_level)**2.0)**2.0-(4.*b**2.0*(B+E_level)**2.0))/(4.*s)) # com p of products
            # now solve for momentum of b, terms are just for ease  
            term1 = np.sqrt(b**2.0+p_prime**2.0)*np.sinh(chi)*np.cos(theta) 
            term2 = np.cosh(chi)*np.sqrt(p_prime**2.0-(b**2.0*np.sin(theta)**2.0*np.sinh(chi)**2.0))
            term3 = 1.0+np.sin(theta)**2.0*np.sinh(chi)**2.0
            p = ((term1+term2)/term3)
            rho = (.3335641*p)/(q*B_field) # from enge's paper 

            return rho

        #define model
        model_variables = [i for i in normals.values() if isinstance(i,pm.Normal)] #get all the normal distributions
        model_variables.append(rho_func)
        model = pm.Model(model_variables)
        mcmc = pm.MCMC(model) #prepare the sampler
        mcmc.sample(steps,burn,progress_bar=False) #50000 samples with a 10000 burn in
        trace = mcmc.trace('rho_func')
        print 
        print "Sampling done, here are the stats"
        print "The 95% confidence interval is",mcmc.stats()['rho_func']['95% HPD interval']
        print "Mean:", mcmc.stats()['rho_func']['mean']
        print "Standard Deviation:", mcmc.stats()['rho_func']['standard deviation']
        return mcmc.stats()['rho_func']['mean'],mcmc.stats()['rho_func']['standard deviation'],trace #the stats we want
        
     
    #chi square fit will be good for quick cal 
    def fit(self,order=2,plot=True):
        N = len(self.points) # Number of data points
        if N > (order+1): #check to see if we have n+2 points where n is the fit order
            print "Using a fit of order",order
            x_rho = np.zeros(N) #just decided to start with arrays. maybe dumb
            x_channel = np.zeros(N)
            x_unc = np.zeros(N) #rho unc
            x_channel_unc = np.zeros(N) #channel unc
            coeff = np.ones(order+1) #these are the n+1 fit coefficients for the polynomial fit
            for i,point in enumerate(self.points):
                #collect the different values needed for chi_square fit
                x_rho[i] = point['rho']['value']
                x_channel[i] = point['channel']['value']
                x_unc[i] = point['rho']['unc']
                x_channel_unc = point['channel']['unc']
                
            #now scale channel points
            channel_mu = np.sum(x_channel)/float(N) #scale will be average of all calibration peaks
            x_channel_scaled = x_channel-channel_mu
            #create bounds
            abound = lambda x:(-100.0,100.0) #creates a tuple
            bounds = [abound(x) for x in xrange(order+1)] #list of tuples
            #differential_evolution method, much faster than basin hopping with nelder-mead and seems to get same answers
            sol = optimize.differential_evolution(chi_square,bounds,maxiter=100000,args=(x_rho,x_channel_scaled,x_unc))
            #tell myself what I did
            chi2 = sol.fun/(N-(order+1))
            print "Chi Square is", chi2
            print "Fit parameters are (from highest order term to lowest)",sol.x
            #now adjust uncertainty to do fit again
            x_unc[:] = self.adjust_unc(np.poly1d(sol.x),x_channel_scaled,x_rho,x_channel_unc,x_unc)
            print "Now doing adjusted unc fit"
            sol = optimize.differential_evolution(chi_square,bounds,maxiter=100000,args=(x_rho,x_channel_scaled,x_unc))
            chi2 = sol.fun/(N-(order+1))
            print "Chi Square is", chi2
            print "Adjusted fit parameters are (from highest order term to lowest)",sol.x
            self.fits[order] = np.poly1d(sol.x) #add to dictionary the polynomial object
            print "Fit stored in member variable fits[%d]" %order
            #create the a plot showing the fit and its residuals
            if plot:
                residual_plot(x_channel_scaled,x_rho,x_unc,self.fits[order])
                plt.show()            
        else:
            print "Not enough points to preform fit of order",order,"or higher"

    #this is based on description in spanc on how they estimate uncertainty
    @staticmethod
    def adjust_unc(poly,x,y,x_unc,y_unc):
        dpoly = np.polyder(poly) #compute derivative
        dpoly = dpoly(x) #evaluate
        new_unc = np.sqrt((dpoly*x_unc)**2.0+y_unc**2.0) #add in quadrature
        return new_unc
    
    #now to try Bayesian fitting method          
    def bay_fit(self,order=2,trace_plot=False,plot=True,
                iterations=100000,burn=50000,thin=10):
        
        #get data
        x_obs = np.asarray([ele['channel']['value'] for ele in self.points])
        channel_mu = np.sum(x_obs)/float(len(x_obs))
        x_scaled = x_obs - channel_mu
        x_unc = np.asarray([ele['channel']['unc'] for ele in self.points])
        #y data scale to make unc on roughly same scale as x unc
        y_obs = np.asarray([ele['rho']['value'] for ele in self.points])
        y_unc = np.asarray([ele['rho']['unc'] for ele in self.points])
        
        letters = string.ascii_uppercase #all upper case letters for prior names
        priors = {letters[i]:pm.Normal(letters[i],mu=0.0,tau=.1) for i in xrange(order+1)} #Normal priors sd = 10
        sorted_priors = np.asarray([j for i,j in sorted(priors.items())])
        sorted_priors[-1] = pm.Normal(letters[order],mu=0.0,tau=.01)
     
        #x uncertainties 
        x = pm.Normal('x',x_scaled,(1.0/x_unc)**2.0,size=len(x_scaled))
        
        #use regular fit to initialize scale of parameters
        self.fit(order=order,plot=False)  
        
        @pm.deterministic
        def Npoly(x=x,priors=sorted_priors):
            total = np.poly1d(priors)(x)
            return total
        
        y_fit = pm.Normal('y',mu=Npoly,tau=(1.0/y_unc)**2.0,value=y_obs,observed=True) #generates fitted posterior
        
        # set model up
        model = pm.Model(y_fit, Npoly, [sorted_priors, x])
        mcmc = pm.MCMC(model)

        # initialize values
        for i in xrange(order+1):
            sorted_priors[i].value = self.fits[order][order-i]
        
        #start sampling
        mcmc.sample(iter=iterations,burn=burn,thin=thin)
        print

        #getting an "adjusted" chi square value based on .5*[chi+C]
        N = ((iterations-burn)/thin)#reduced chi square needs to account for data,samples,and constraints
        nu = (len(x_obs)-(order+1))#dof
        chi = np.sum(((y_fit.get_value()[:,None]-Npoly.trace().T)/y_unc[:,None])**2.0)/N #normal chi square term
        C = np.sum(((x_scaled[:,None]-x.trace().T)/x_unc[:,None])**2.0)/N #contribution from jitter in x
        print r"\Chi is:", chi/nu
        print "C is:", C/len(x_obs)
        print "Adjusted chi squared after sampling is:", .5*(chi/nu+C/len(x_obs))
        
        #gives traces for diagnostic purposes
        if trace_plot:
            pm.Matplot.plot(mcmc)
            plt.show()
   
        Fit = []
        
        for ele in sorted(priors.keys()):
            coeff,coeff_unc = mcmc.stats()[ele]['mean'],mcmc.stats()[ele]['standard deviation']
            print "Mean for "+ele+" is",coeff,"+/-",coeff_unc
            Fit.append(measured_value(coeff,coeff_unc))
            
        if plot:
            fit = np.poly1d([ele['value'] for ele in Fit])
            residual_plot(x_scaled,y_obs,y_unc,fit,xerr=x_unc)
            plt.show()

        Fit.append(channel_mu) #give offset 
        self.fits_bay['Order_'+str(order)] = Fit[:] #store polynomial parameters

    def input_peak(self):
        channel = float(raw_input("Enter channel number."))
        channel_unc = float(raw_input("Enter channel uncertainty."))
        channel = measured_value(channel,channel_unc)
        self.peak_energy(channel)
        
    #finally given channel number use a fit to give energy         
    def peak_energy(self,reaction,channel,fit_order=2):
        if type(channel) == dict:
            Reaction = self.reactions[reaction]
            normals = self.create_distributions(Reaction) #get reaction variables as distributions
            #calc rho from normal distributions in polynomial fit
            coeff = []
            letters = string.ascii_uppercase
            i = 0
            for ele in self.fits_bay['Order_'+str(fit_order)]:
                if type(ele) is dict:
                    mu = ele['value']
                    sigma = ele['unc'] 
                    coeff.append(pm.Normal(letters[-i-1],mu=mu,tau=(1.0/sigma)**2.0)) #doing letters in reverse Z,Y,X to avoid conflicts in names
                    i = i + 1
                else:
                    x_mu = ele #exception for average mu because it is a scalar that is exact
                    
            #channel normal distribution
            Channel = pm.Normal('Channel',mu=(channel['value']-x_mu),tau=(1.0/channel['unc'])**2.0)

            
                    
            print "Calculating the fitted rho value and Energy..."

            @pm.deterministic
            def calc_rho(channel=Channel,coeff=coeff):
                value = np.poly1d(coeff)(channel)
                return value

            #now to do the kinematics backwards to get E_level
            @pm.deterministic
            def E_level(rho=calc_rho,A=normals["A"],B=normals["B"],
                        a=normals["a"],b=normals["b"],E_lab=normals["E_lab"],
                        theta=normals["theta"], q=normals["q"],
                        B_field=normals["B_field"],Q=normals["Q"]):

                theta = theta*(np.pi/180.0) #to radians
                #r = (np.sqrt(a*b*E_lab))/(b+B)*np.cos(theta)  
                pb = (B_field*rho*q)/.3335641 #momentum of b
                pa = np.sqrt(2.0*a*E_lab+E_lab**2.0)
                pB = np.sqrt(pa**2.0+pb**2.0-(2.0*pa*pb*np.cos(theta))) # conservation gives you this
                #now do conservation of Energy
                E_tot = E_lab+a+A # what we start out with
                Eb = np.sqrt(pb**2.0+b**2.0)
                EB = np.sqrt(pB**2.0+B**2.0)
                Ex = E_tot - Eb - EB
                return Ex
             
            model = pm.Model([calc_rho,E_level],coeff,Channel)
            mcmc = pm.MCMC(model)
            mcmc.sample(11000,1000)
            print
            mu =  mcmc.stats()['calc_rho']['mean']
            sig = mcmc.stats()['calc_rho']['standard deviation']
            E = mcmc.stats()['E_level']['mean']
            E_sig = mcmc.stats()['E_level']['standard deviation']
            print "Rho",mu,"+/-",sig
            print "E_level",E,"+/-",E_sig
            output = {'Reaction':reaction,'Rho':measured_value(mu,sig),'E_level':measured_value(E,E_sig),'Channel':channel,
                      'Trace':E_level.trace()} #gather values into dictionary
            self.output_peaks.append(output) #append to list 
        else:
            print "Need a dictionary of value and uncertainty!!"
        

    #function to read in a file with calibration points and preform a fit on them    
    def read_calibration(self,cal_file,reaction=None,scale_factor=1.0):
        #can just pick, or ask for user input for reaction
        if type(reaction) != int:
            reaction = int(raw_input('Which reaction(0...n)? \n'))
        data = pd.read_csv(cal_file,sep='\s+')
        for i in data.index:
            level = data["level"][i]
            level_unc = data["level_unc"][i]
            channel = (data["channel"][i])/scale_factor
            channel_unc = (data["channel_unc"][i])/scale_factor
            self.add_point(reaction,level,level_unc,channel,channel_unc)
            
    #reads file with channel and channel_unc and gives fitted values  
    def read_peaks(self,peak_file,reaction=None,fit_order=2,scale_factor=1.0):
        if type(reaction) != int:
            reaction = int(raw_input('Which reaction(0...n)? \n'))
        data = pd.read_csv(peak_file,sep='\s+')
        for i in data.index:
            channel = (data["channel"][i])/scale_factor
            channel_unc = (data["channel_unc"][i])/scale_factor
            self.peak_energy(reaction,measured_value(channel,channel_unc),fit_order=fit_order)
            
            
#test on some pretty suspect data, just for basic functionality checks
def atest():
    test = Focal_Plane_Fit()
    test.reactions[0] = Reaction('3He','23Na','2H','24Mg',1.05,19.917,.00001,6.0)
    test.read_calibration('/home/caleb/Research/Longland/Experiments/23Na_24Mg/DataAnalysis/6_deg_data/calibration.dat',reaction=0)
    test.bay_fit()
    test.fit()


# this data corresponds to a 'good' fit I got with a SPANC fit.
# Should be used to check the bay_fit function.
def btest():
    
    test = Focal_Plane_Fit()
    test.reactions[0] = Reaction('3He','23Na','2H','24Mg',1.0509,20.0,.00001,10.0)
    test.reactions[1] = Reaction('3He','12C','2H','13N',1.0509,20.0,.00001,10.0)
    test.reactions[2] = Reaction('3He','16O','2H','17F',1.0509,20.0,.00001,10.0)
    test.read_calibration('./23Na.dat',reaction=0)
    test.read_calibration('./12C.dat',reaction=1)
    test.read_calibration('./16O.dat',reaction=2)
    #test.fit(order=3)
    test.bay_fit(order=3,trace_plot=True,plot=True)
    test.peak_energy(0,measured_value(2232.1,2.2),fit_order=3)
