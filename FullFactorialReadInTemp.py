# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:11:41 2022

@author: mkoerschner3
"""
import numpy as np
import csv

v = np.empty([5**6+1,7],dtype=object)
with open('planet_factorial.csv') as pl:
    reader = csv.reader(pl)
    for row in reader:
        v[reader.line_num-1,:] = row
        
#somehow this will have read in the full factorial
#of the trajectory options