###############################################################
######################## NuDat.py #############################
###############################################################
# Pulls energy level information for a given nuclei from nudat#
###############################################################


#libraries for html parsing

from lxml import html
from lxml import etree
import requests
import re
import numpy as np

 
class Nudat_Page():

    #pull the page that the data is on
    def __init__(self,name):
        url = 'http://www.nndc.bnl.gov/chart/getdataset.jsp?nucleus=REPLACE&unc=nds' #where the data is
        self.url = url.replace('REPLACE',str(name))       
        self.page = requests.get(self.url) #fetch raw html
        self.tree = etree.HTML(self.page.content) #parse the web page
        self.E_xpath = '/html/body/table/tr/td[1]/text()' #table indices for E 
        self.Jpi_xpath = '/html/body/table/tr/td[3]/text()' #table indices for Jpi
        
    #use xpath query to get energy information
    def get_level(self,level_number):
        level_number = str(level_number+1) #Account for table label row
        raw_energy = self.tree.xpath(self.E_xpath.replace('tr','tr['+level_number+']')) #sub in energy level value
        clean_energy = self.clean_list(raw_energy) #get the energy from the list
        energy = re.findall(r'\d+\.\d+|\d+',clean_energy) #There are some ? that float around so this makes sure we just produce numbers
        if energy: #check for empty element
           energy = energy[0] #if not empty take list to string
        #now we move onto spin parity
        raw_angular = self.tree.xpath(self.Jpi_xpath.replace('tr','tr['+level_number+']')) #get momentum parity answer
        clean_angular = self.clean_list(raw_angular)
        spin = re.findall(r'\d+\/\d+|\d+',clean_angular)
        parity = re.findall('\\+|\-',clean_angular)
        if spin:
            spin = spin[0] #lists to strings
        if parity:
            parity = parity[0] #lists to strings
        
        return Level(energy,spin,parity)
    
    #clean up the unicode lists from the xpath search
    def clean_list(self,lst): 
        try: #make sure we were passed a list which is the only output from a successful xpath
            lst = lst[0]
            lst = lst.encode('ascii','ignore')
            return lst
        except IndexError:
            print 'Something went wrong with extracting the data!'

    #given a string find it in the table xpath given
    # def find(self,string,xpath):
    #     for ele in self.tree.xpath(xpath):
    #         if string in ele:
    #             return 

            
#basically a fancy dictinary for levels that right now contain energies(relative to COM)
#and spin parities
class Level():

    def __init__(self,E,J,parity):
        self.E = E
        self.J = J
        self.parity = parity
        self.info = str(E)+' keV '+str(J)+str(parity) #if you want to quickly check the level info



#class for a nuclei, deverives from the base class levels
class Nuclie():

    def __init__(self,name):
        self.name = str(name) #Class is initiated with name of nuclei we want to pull data for
        self.levels = {} #dictinary for energy levels, so that we can pull based on excited state
        self.page = Nudat_Page(self.name.upper()) #make sure letters are uppercase for nudat
        
    #add a level to the nuclei based on its number realtive to g.s.
    #i.e g.s=0, 1st =1, .etc
    def add_level(self,level_number):
        level = self.page.get_level(level_number)
        print "Adding Energy Level "+str(level.E)+"\n"
        self.levels[level_number] = level

    #add a range of values to the level dictonary
    def add_range(self,values):
        for i in values:
            self.add_level(i)

    #searches for a given range of energies and adds them to the nuclie
    # def add_energy_range(self,start,end):
    #     with self.page as f:
    #         number_of_levels = len(f.tree.)
    #         for ele in f.tree.xpath(f.)

    #make a line file for BaySpec first col rho second energy level
    def make_line_file(self,E_beam,Q_value,B,q,m):
        with open(self.name+'_'+str(B)+'T.lines','wr') as f:
            #calculate rho
            for level in self.levels.itervalues():
               E_level = float(level.E)
               E_level = E_level/1000.0 #convert to MeV
               E = E_beam+Q_value-E_level
               rho = str(np.sqrt(2*m*E)/(q*B))
               print rho
               f.write(rho+'\t'+level.E+'\n')
