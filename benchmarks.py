# -*- coding: utf-8 -*-
"""
Created on April 26 2019

@author: Himanshu Mittal
"""

import numpy
import math
from scipy.spatial.distance import cdist
import pandas as pd

def F1(x,df):

  x=numpy.asarray(x)
  x=x.reshape(1, -1)
  [m,n]=x.shape
  n1=len(df.columns)
  k1=n/n1
  x=numpy.reshape(x, (k1,-1))
  distn=cdist(df,x) 
  dmin=numpy.amin(distn, axis = 1)
  ind=numpy.argmin(distn, axis = 1)
  s=numpy.sum(dmin);
  return s

