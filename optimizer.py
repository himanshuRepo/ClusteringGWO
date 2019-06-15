# -*- coding: utf-8 -*-
"""
Created on April 26 2019

@author: Himanshu Mittal
"""
import GWO as gwo

import benchmarks
import csv
import numpy
import pandas as pd
import time
from scipy.spatial.distance import cdist


def selector(algo,func_details,popSize,Iter,df):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    if(algo==0):
        x=gwo.GWO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,df)
        x1=x.pos
        x1=numpy.asarray(x1)
        # x1=x.copy()
        x1=x1.reshape(1, -1)
        [m,n]=x1.shape
        n1=len(df.columns)
        k12=n/n1
        x1=numpy.reshape(x1, (k12,-1))
        distn=cdist(df,x1) 
        dmin=numpy.amin(distn, axis = 1)
        ind=numpy.argmin(distn, axis = 1)
        s=numpy.sum(dmin);
        print(ind.shape)
        ind=ind.reshape(1, -1)
        print(ind.shape)
        print(['The total intra-cluster cost: '+ str(s)+ ' the corresponding index: '+ str(ind)]);
    return x
    
    
# Select optimizers
GWO = True



optimizer=[GWO]
datasets=["Iris"] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=1

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 500
Iterations= 50

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))

k1=3 #The number of clusters
for j in range (0, len(datasets)):
    dataset=datasets[j]+".csv"
    df=pd.read_csv(dataset)
    # df1=pd.read_csv("Iris.csv")
    df1=pd.DataFrame(df, copy=True)
    df2=df1.iloc[:,1:]
    df=df2.iloc[:,:-1]
    lowdf=numpy.amin(df, axis = 0)
    updf=numpy.amax(df, axis = 0)
    lbdf=list(lowdf.values.flatten())
    ubdf=list(updf.values.flatten())
    lbdf=lbdf*k1
    ubdf=ubdf*k1


    for i in range (0, len(optimizer)):
        if((optimizer[i]==True)): # start experiment if an optimizer and an objective function is selected
            for k in range (0,NumOfRuns):
                # func_details=["F1",0,1,((len(df.columns))*k1)] 
                func_details=["F1",lbdf,ubdf,((len(df.columns))*k1)] 
                x=selector(i,func_details,PopulationSize,Iterations,df)
                if(Export==True):
                    with open(ExportToFile, 'a') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (Flag==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime"],CnvgHeader])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime],x.convergence])
                        writer.writerow(a)
                    out.close()
                Flag=True # at least one experiment
                
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        
