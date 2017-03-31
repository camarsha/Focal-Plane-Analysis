"""
Take a jam text output file and convert it to a Rho v count file for BaySpec
"""

import numpy as np

def fit(x_0,parms):
    our_fit = np.poly1d(parms)
    def eval(x):
        return our_fit(x-x_0)    
    return eval


def jam_cal(datafile,fit):
    newname = datafile.split('.')[0]
    with open(datafile,'r') as f, \
         open(newname+'_calibrated.dat','wr') as g:
        for line in f:
            line = line.split()
            chan = line[0]
            cal_chan = fit(float(chan))
            g.write(str(cal_chan)+' '+line[1]+'\n')
        

 #make a line file for BaySpec first col rho second energy level
def make_line_file(E_Level,E_beam,Q_value,B,q,m):
    if type(E_Level) != list:
        E_Level = [E_Level]
        
    with open(str(B)+'T.lines','wr') as f:
        #calculate rho
        for ele in E_Level:
            E = (E_beam+Q_value-ele)
            rho = str(np.sqrt(2*m*E)/(q*B))
            f.write(rho+'\t'+str(ele)+'\n')
