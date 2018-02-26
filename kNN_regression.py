import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator 
import math
import csv



def construct_knn_approx_1D(train_inputs, train_targets, k):
    """
    For 1 dimensional training data, it produces a function:reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.
    """
    # Create Euclidean distance.
    distance = lambda x,y: (x-y)**2
    train_inputs = np.resize(train_inputs, (1,train_inputs.size))
    def prediction_function(inputs):
        inputs = inputs.reshape((inputs.size,1))
        distances = distance(train_inputs, inputs)
        predicts = np.empty(inputs.size)
        for i, neighbourhood in enumerate(np.argpartition(distances, k)[:,:k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(train_targets[neighbourhood])
        return predicts
    # We return a handle to the locally defined function
    return prediction_function


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)

	

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(0,k):
		neighbors.append(distances[x][0])
	return neighbors


def getResponse(neighbors):
    length = len(neighbors)
    quality = []
    for x in range(0,length):
        c = neighbors[x][-1]
        quality.append(float(c))
    return np.mean(quality)


def getAccuracy(testSet,getResponse):
    error = len(getResponse)
    err = []
    rms = []
    for x in range(0,len(getResponse)):
        error = (float(testSet[x][-1]) - getResponse[x])**2
        ems = np.sqrt(error)
        err_rate = ems/float(testSet[x][-1])*100
        err.append(err_rate)
    err = np.mean(err)
    #return err
    return err
    
def rms_function(testSet,getResponse):
    error = len(getResponse)
    #err = []
    rms = []
    for x in range(0,len(getResponse)):
        error = (float(testSet[x][-1]) - getResponse[x])**2
        rms.append(error)
    rms = np.mean(rms)
    rms = np.sqrt(rms)
    #return err
    return rms
    
def accuracyRate_1D(prediction,real):
    errorRate = (abs(prediction - real))/real *100
    errorRate = np.mean(errorRate)
    return errorRate
    
def kNN_entry_point(data,field_names):


    K=[]
    Acc=[]
    trainingSet = data[0:1280,]
    testSet = data[1280:1600,]
    rms_array = []		
    for k in range(1,50):
        K.append(k)  
        predictions=[]
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
        error_rate = getAccuracy(testSet, predictions)
        RMS = rms_function(testSet, predictions)
        rms_array.append(RMS)
        Acc.append(error_rate)
        
		
    print("Choices of k: %r" %K)
    print("Error rate is for accuracy checking for the responsing k: %r" %Acc)
    print("rms for responsing k: %r" %rms_array)
    fig1 = plt.figure()
    fig1.suptitle("Error Rate(%) against k")
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Error Rate/ %")
    ax1.plot(K,Acc,'-',markersize=1)
    #fig1.savefig("Error Rate against k.pdf", fmt="pdf")
    plt.show()
    
    fig3 = plt.figure()
    fig3.suptitle("E_rms against k")
    ax3=fig3.add_subplot(1,1,1)
    ax3.set_xlabel("k")
    ax3.set_ylabel("E_rms")
    ax3.plot(K,rms_array,'-',markersize=1)
    
    # Then we notice that k in range(20,28)
    # can lead a stable rms
    # in this case k is choses as 24 for further analysis
    
    # cross validation for 5-folds
    # with k = 24, 
    folds = 5
    dataSet = data
    k = 23
    length =  round(len(dataSet)/folds)
    mean = []
    for y in range(1,folds):
        testingSet = dataSet[y*length:(y+1)*length,]
        #print(testingSet)
        trainingSet = np.delete(dataSet,[i for i in range(y*length,(y+1)*length)],0)
        #print(trainingSet)
        Acc = []
        predictions=[]
        for x in range(1,50):
            neighbors = getNeighbors(trainingSet, testingSet[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
        error_rate = getAccuracy(testingSet, predictions)
        rms_cv = rms_function(testingSet, predictions)
        Acc.append(error_rate)
        #print("Acc")
    mean.append(Acc)
    print("cross validation" )
    print(np.mean(mean))
    print("rms for cross validation")
    print(rms_cv)
   
    # let us do the knn approximation for individual attributors 
    # to check the individual effect on quality
    pred = construct_knn_approx_1D(data[0:1280,0], data[0:1280,-1], 1)(data[1280:1600,0])
    err_rate = accuracyRate_1D(pred,data[1280:1600,-1])
    print("error rate for 1D %r" %err_rate)
    
    individual_attributes_index = []
    error_rate_individual = []
    for x in range(0,12):
        individual_attributes_index.append(x)    
        pred = construct_knn_approx_1D(data[0:1280,x], data[0:1280,-1], 1)(data[1280:1600,0])
        error_rate_ind = accuracyRate_1D(pred,data[1280:1600,-1])
        #print(error_rate_ind)
        error_rate_individual.append(error_rate_ind)
    print(sorted(error_rate_individual))
    
    fig2 = plt.figure()
    fig2.suptitle("Individual Error Rate(%)")
    ax2=fig2.add_subplot(1,1,1)
    ax2.set_xlabel("Attributors")
    ax2.set_ylabel("Error Rate/ %")
    ax2.plot(individual_attributes_index,error_rate_individual,'-',markersize=1)
    plt.show()
    fig2.savefig("Individual Error Rate", fmt="pdf")
    
    
    #Then from the figure generated above, we can observe that
    #There are four attributors with index 0 1 3 4 will lead smaller error rate
    # with same k
    #let's do one more kNN regression with only collected 4 attributes
    testSet = data[0:1280,]
    trainingSet = data[1280:1600,]
    testSet = data[0:1280,]
    trainingSet = data[1280:1600,]
    testSet = np.delete(testSet,[2,5,6,7,8,9,10],1)
    trainingSet = np.delete(testSet,[2,5,6,7,8,9,10],1)

    K = []
    Acc = []
    for k in range(1,50):
        K.append(k)  
        predictions=[]
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
        error_rate = getAccuracy(testSet, predictions)
        Acc.append(error_rate)

		
    print("Choices of k: %r" %K)
    print("Error rate is for accuracy checking for the responsing k: %r" %Acc)
             		   		   		
