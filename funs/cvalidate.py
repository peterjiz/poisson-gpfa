#util.py
# A module that contains various small functions that the main engine makes use of.
import funs.learning as learning
import funs.inference as inference
import funs.engine as engine
import funs.util as util
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.optimize as op
from scipy.optimize import approx_fprime
import statsmodels.tools.numdiff as nd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pdb
import copy
import pickle
import sys
import pandas


class crossValidation():
    def __init__(
        self,
        experiment, 
        numTrainingTrials = 10,
        numTestTrials = 2,
        maxXdim = 6,
        maxEMiter = 3,
        batchSize = 5,
        inferenceMethod = 'laplace',
        learningMethod = 'batch', 
        errorFunc = util.leaveOneOutPrediction):
        print('Assessing optimal latent dimensionality will take a long time.')
        print('Error Func') 
        print(errorFunc)

        trainingSet, testSet = splitTrainingTestDataset(experiment, numTrainingTrials, numTestTrials)
        errs = []
        fits = []
        for xdimFit in np.arange(0,maxXdim+1):
            initParams = initializeParams(xdimFit, trainingSet.ydim, trainingSet)
            #print("params") 
            #print(initParams) 
            #print("\n")
            
            if learningMethod == 'batch':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Batch', 
                    maxEMiter = maxEMiter)
                predBatch,predErr = errorFunc(fit.optimParams, testSet)
                errs.append(predErr)

            if learningMethod == 'diag':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Online', 
                    onlineParamUpdateMethod = 'diag',
                    maxEMiter = maxEMiter,
                    batchSize = batchSize)
                predDiag,predErr = errorFunc(fit.optimParams, testSet)
                errs.append(predErr)
            
            if learningMethod == 'hess':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Online', 
                    onlineParamUpdateMethod = 'hess',
                    maxEMiter = maxEMiter,
                    batchSize = batchSize)
                predHess,predErr = errorFunc(fit.optimParams, testSet)
                errs.append(predErr)
            
            if learningMethod == 'grad':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Online', 
                    onlineParamUpdateMethod = 'grad',
                    maxEMiter = maxEMiter,
                    batchSize = batchSize)
                predGrad,predErr = errorFunc(fit.optimParams, testSet)
                errs.append(predErr)
            fits.append(fit) 
        
        self.fit_0, self.err_0 = fits[0], errs[0] 
        fits, errs = fits[1:], errs[1:]

        self.inferenceMethod = inferenceMethod
        self.learningMethod = learningMethod
        self.optimXdim=np.argmin(errs)+1 # because python indexes from 0 
        self.errs = errs
        self.maxXdim = maxXdim
        self.fits = fits

    def plotPredictionError(self):
        plt.figure(figsize=(5,4))
        plt.plot(np.arange(1,self.maxXdim+1),self.errs,'b.-',markersize=5,linewidth=2)
        plt.legend([self.method],fontsize=9,framealpha=0.2)
        plt.xlabel('Latent Dimensionality')
        plt.ylabel('Error')
        plt.title('Latent Dimension vs. Prediction Error')
        plt.grid(which='both')
        plt.tight_layout()