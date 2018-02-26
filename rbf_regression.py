import numpy as np
from regression_train_test import train_and_test
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from regression_plot import plot_train_test_errors
from regression_models import construct_rbf_feature_mapping
from regression_train_test import simple_evaluation_linear_model
from regression_models import construct_feature_mapping_approx
from regression_models import ml_weights
from regression_models import linear_model_predict
from regression_train_test import root_mean_squared_error
from regression_models import regularised_ml_weights


def parameter_search_rbf_without_cross(inputs, targets, test_fraction,test_error_linear,normalize=True):
    """
    """
    if(normalize):
        # normalise inputs (meaning radial basis functions are more helpful)
        for i in range(inputs.shape[1]):
            inputs[:,i]=(inputs[:,i]-np.mean(inputs[:,i]))/np.std(inputs[:,i])
    N = inputs.shape[0]

    # for the centres of the basis functions sample 10% of the data
    sample_fractions = np.array([0.05,0.1,0.15,0.2,0.25])
    scales = np.logspace(0,4,20 ) # of the basis functions
    reg_params = np.logspace(-16,-1, 20) # choices of regularisation strength.
    # create empty 3d arrays to store the train and test errors
    train_mean_errors = np.empty((sample_fractions.size,scales.size,reg_params.size))
    test_mean_errors = np.empty((sample_fractions.size,scales.size,reg_params.size))
    
    #Randomly generates a train/test split for data of size N. Returns a 2 arrays of boolean true/false.
    train_part, test_part = train_and_test_split(N, test_fraction=test_fraction)
    best_k=0
    best_i=0
    best_j=0
    test_error_temp=10**100
    
 
    #loop through the possible centres as a percentage (5%, 10%,15%, 20%, 25%)
    for k,sample_fraction in enumerate(sample_fractions):
        p = (1-sample_fraction,sample_fraction)
        centres = inputs[np.random.choice([False,True], size=N, p=p),:]       
        # iterate over the scales
        for i,scale in enumerate(scales):
            # i is the index, scale is the corresponding scale
            # we must recreate the feature mapping each time for different scales
            feature_mapping = construct_rbf_feature_mapping(centres,scale)
            designmtx = feature_mapping(inputs)
            # partition the design matrix and targets into train and test. This effectively takes as inputs the boolean arrays train_part, test_part and the whole design matrix and
            #creates 2 subsets of the design matrix  (train matrix, test matrix). The test data are splitted as well but the values are not affected
            train_designmtx, train_targets, test_designmtx, test_targets =  train_and_test_partition(designmtx, targets, train_part, test_part)
            # iteratre over the regularisation parameters
            for j, reg_param in enumerate(reg_params):
                # j is the index, reg_param is the corresponding regularisation
                # parameter
                # train and test the data
                train_error, test_error,weights = train_and_test(train_designmtx, train_targets, test_designmtx, test_targets,reg_param=reg_param)
                # store the train and test errors in our 2d arrays
                train_mean_errors[k,i,j] = train_error
                test_mean_errors[k,i,j] = test_error
                #When we've found a lowest than stores test error value, we store it's indices
                if (np.mean(test_error)<test_error_temp):
                    test_error_temp=test_error
                    best_k=k
                    best_i=i
                    best_j=j
    print ("The value with the lowest error is:",test_mean_errors[best_k][best_i][best_j])
    print("Best joint choice of parameters: sample fractions %.2g scale %.2g and lambda = %.2g" % (sample_fractions[best_k],scales[best_i],reg_params[best_j]))
    
    
    # now we can plot the error for different scales using the best
    # regularization choice
    
    # now we can plot the error for different scales using the best regularization choice and centres percentage
    fig , ax = plot_train_test_errors("scale", scales, train_mean_errors[best_k,:,best_j], test_mean_errors[best_k,:,best_j])
    ax.set_xscale('log')
    fig.suptitle('RBF regression for the best reg. parameter & centres', fontsize=10)
    xlim = ax.get_xlim()#get the xlim to graph the linear regression
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:') #graph the linear regression
    
    
    # ...and the error for  different regularisation choices given the best scale choice and centres percentage
    fig , ax = plot_train_test_errors("$\lambda$", reg_params, train_mean_errors[best_k,best_i,:], test_mean_errors[best_k,best_i,:])
    ax.set_xscale('log')
    fig.suptitle('RBF regression for the best scale parameter & centres', fontsize=10)
    xlim = ax.get_xlim()#get the xlim to graph the linear regression
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:')
    # #ax.set_ylim([0,20])
    
    
    # ...and the error for  different centres given the best reg.parameter and the best scale choice
    fig , ax = plot_train_test_errors("sample fractions", sample_fractions, train_mean_errors[:,best_i,best_j], test_mean_errors[:,best_i,best_j])
    fig.suptitle('RBF regression for the best scale parameter & reg. parameter', fontsize=10)
    ax.set_xlim([0.05, 0.25])
    xlim = ax.get_xlim()#get the xlim to graph the linear regression
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:')


def parameter_search_rbf_cross(inputs, targets, folds,test_error_linear,test_inputs,test_targets,normalize=True):
    """
    This function will take as inputs the raw data and targets, the folds for cross validation and the test linear error for plotting
    """
    if(normalize):
        # normalise inputs (meaning radial basis functions are more helpful)
        for i in range(inputs.shape[1]):
            inputs[:,i]=(inputs[:,i]-np.mean(inputs[:,i]))/np.std(inputs[:,i])
            test_inputs[:,i]=(test_inputs[:,i]-np.mean(test_inputs[:,i]))/np.std(test_inputs[:,i])
    N = inputs.shape[0]

    # for the centres of the basis functions sample 10% of the data
    sample_fractions = np.array([0.05,0.1,0.15,0.2,0.25])
    scales = np.logspace(0,4,20 ) # of the basis functions
    reg_params = np.logspace(-16,-1, 20) # choices of regularisation strength.
    # create empty 3d arrays to store the train and test errors
    train_mean_errors = np.empty((sample_fractions.size,scales.size,reg_params.size))
    test_mean_errors = np.empty((sample_fractions.size,scales.size,reg_params.size))
    
    best_k=0
    best_i=0
    best_j=0
    test_error_temp=10**100
    
    #loop through the possible centres as a percentage (5%, 10%,15%, 20%, 25%)
    for k,sample_fraction in enumerate(sample_fractions):
        p = (1-sample_fraction,sample_fraction)
        centres = inputs[np.random.choice([False,True], size=N, p=p),:]
        # iterate over the scales
        for i,scale in enumerate(scales):
            # i is the index, scale is the corresponding scale
            # we must recreate the feature mapping each time for different scales
            feature_mapping = construct_rbf_feature_mapping(centres,scale)
            designmtx = feature_mapping(inputs)
            # iteratre over the regularisation parameters
            for j, reg_param in enumerate(reg_params):
                # j is the index, reg_param is the corresponding regularisation
                # parameter for train and test the data
                train_error, test_error,weights = cv_evaluation_linear_model(designmtx, targets, folds,reg_param=reg_param)
                
                #When we've found a lowest than stores test error value, we store it's indices
                if (np.mean(test_error)<test_error_temp):
                    test_error_temp=np.mean(test_error)
                    best_k=k
                    best_i=i
                    best_j=j
                    optimal_weights=weights
                    optimal_feature_mapping=feature_mapping
                    
                # store the train and test errors in our 3d matrix
                train_mean_errors[k,i,j] = np.mean(train_error)
                test_mean_errors[k,i,j] = np.mean(test_error)
    
    print ("The value with the lowest test error at the training stage is:",test_mean_errors[best_k][best_i][best_j])
    print("Best joint choice of parameters: sample fractions %.2g scale %.2g and lambda = %.2g" % (sample_fractions[best_k],scales[best_i],reg_params[best_j]))
    
    
    # now we can plot the error for different scales using the best regularization choice and centres percentage
    fig , ax = plot_train_test_errors("scale", scales, train_mean_errors[best_k,:,best_j], test_mean_errors[best_k,:,best_j])
    ax.set_xscale('log')
    fig.suptitle('RBF regression for the best reg. parameter & centres using cross-validation', fontsize=10)
    xlim = ax.get_xlim()#get the xlim to graph the linear regression
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:') #graph the linear regression
    
    
    # ...and the error for  different regularisation choices given the best scale choice and centres percentage
    fig , ax = plot_train_test_errors("$\lambda$", reg_params, train_mean_errors[best_k,best_i,:], test_mean_errors[best_k,best_i,:])
    ax.set_xscale('log')
    fig.suptitle('RBF regression for the best scale parameter & centres using cross-validation', fontsize=10)
    xlim = ax.get_xlim()#get the xlim to graph the linear regression
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:')
    # #ax.set_ylim([0,20])
    
    
    # ...and the error for  different centres given the best reg.parameter and the best scale choice
    fig , ax = plot_train_test_errors("sample fractions", sample_fractions, train_mean_errors[:,best_i,best_j], test_mean_errors[:,best_i,best_j])
    fig.suptitle('RBF regression for the best scale parameter & reg. parameter using cross-validation', fontsize=10)
    ax.set_xlim([0.05, 0.25])
    xlim = ax.get_xlim()#get the xlim to graph the linear regression
    ax.plot(xlim, test_error_linear*np.ones(2), 'g:')
    
    predictive_func=construct_feature_mapping_approx(optimal_feature_mapping, optimal_weights)
    
    final_error=root_mean_squared_error(test_targets,predictive_func(test_inputs))
    print("final test error for RBF model:",final_error)
    
    

def simple_linear_regression(inputs,targets,folds,test_fraction,test_inputs,test_targets):
    train_error, test_error = simple_evaluation_linear_model(inputs, targets, test_fraction=test_fraction)
    print("Linear Regression with no cross-validation and no regularisation: (train_error, test_error) = %r" % ((train_error, test_error),))
    train_errors, test_errors,weights= cv_evaluation_linear_model(inputs, targets, folds)
    print("The train errors for linear regression using cross validation are:= %r" % (train_errors),)
    print ("The test error for linear regression using cross validation are:= %r" % (test_errors),)
    print ("The average means of errors are the following: (train_error,test_error)= %r" % (( np.mean(train_errors),np.mean(test_errors) ) , )  )  
    final_error=root_mean_squared_error(test_targets,linear_model_predict(test_inputs,weights))
    print("The final test error for linear regression is: %r" % final_error)
    return np.mean(train_errors),np.mean(test_errors)
        

    
def regression_with_regularization(inputs,targets,folds):    
    reg_params = np.logspace(-10,1)
    train_errors = []
    test_errors = []
    for reg_param in reg_params:
        # evaluate the test and train error for this regularisation parameter
        old_train,old_test=simple_evaluation_linear_model(inputs, targets, test_fraction=0.2, reg_param=reg_param)
        train_error, test_error,weights = cv_evaluation_linear_model(inputs, targets, folds,reg_param=reg_param)
        print(" (train_error_without_cross,test_error_without_cross,train_error_with_cross,test_error_with_cross)= %r" % ((old_train,old_test,np.mean(train_error),np.mean(test_error)),) )
        # collect the errors
        train_errors.append(np.mean(train_error))
        test_errors.append(np.mean(test_error))
    # plot the results
    fig, ax = plot_train_test_errors("$\lambda$", reg_params, train_errors, test_errors)        
    ax.set_xscale('log')
    
    
def cv_evaluation_linear_model(inputs, targets, folds, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on the both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix !!!!!!!!!!!!!!!!!!!!(any feature mapping should already be applied)
    targets - the targets as a vector
    num_folds - the number of folds
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_errors - the training errors for the approximation
    test_errors - the test errors for the approximation
    """
    # get the number of datapoints
    N = inputs.shape[0]
    # get th number of folds
    num_folds = len(folds)
    train_errors = np.empty(num_folds)
    test_errors = np.empty(num_folds)
    for f,fold in enumerate(folds):
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)
        # now train and evaluate the error on both sets
        train_error, test_error,weights = train_and_test(
            train_inputs, train_targets, test_inputs, test_targets,
            reg_param=reg_param)
        #print("train_error = %r" % (train_error,))
        #print("test_error = %r" % (test_error,))
        train_errors[f] = train_error
        test_errors[f] = test_error
    return train_errors, test_errors,weights
    
def train_and_test(train_inputs, train_targets, test_inputs, test_targets, reg_param=None):
    """
    Will fit a linear model with either least squares, or regularised least 
    squares to the training data, then evaluate on both test and training data

    parameters
    ----------
    train_inputs - the input design matrix for training
    train_targets - the training targets as a vector
    test_inputs - the input design matrix for testing
    test_targets - the test targets as a vector
    reg_param (optional) - the regularisation strength. If provided, then
        regularised maximum likelihood fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """
    # Find the optimal weights (depends on regularisation)
    if reg_param is None:
        # use simple least squares approach
        weights = ml_weights(
            train_inputs, train_targets)
    else:
        # use regularised least squares approach
        weights = regularised_ml_weights(
          train_inputs, train_targets,  reg_param)
    # predictions are linear functions of the inputs, we evaluate those here
    train_predicts = linear_model_predict(train_inputs, weights)
    test_predicts = linear_model_predict(test_inputs, weights)
    # evaluate the error between the predictions and true targets on both sets
    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)
    if np.isnan(test_error):
        print("test_predicts = %r" % (test_predicts,))
    return train_error, test_error,weights
    