import csv
import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import matplotlib.pyplot as plt


# for creating synthetic data
# for performing regression
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
from regression_models import calculate_weights_posterior
from regression_models import predictive_distribution
from regression_models import ml_weights
# for plotting results

# from regression_plot import plot_train_test_errors
from regression_plot import exploratory_plots
from regression_plot import plot_train_test_errors
# for evaluating fit
from regression_train_test import cv_evaluation_linear_model
from regression_train_test import create_cv_folds


def bayesian_regression_entry_point(data):
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    data_targets=data[:,-1]
    data=data[:,0:11]
    
    print (data)
    print (data_targets)
    for i in range(data.shape[1]):
        data[:,i]=(data[:,i]-np.mean(data[:,i]))/np.std(data[:,i])
    print("standard deviation is %s" % str(np.std(data,axis=0)))
    
    
    inputs=data[0:960,:]
    targets=data_targets[0:960]
    test_inputs=data[1300:1599,:]
    test_targets=data_targets[1300:1599]
    
    # specify the centres of the rbf basis functions
    N=inputs.shape[0]
    centres1 = inputs[np.random.choice([False,True], size=N, p=[0.9,0.1]),:]
    # centres1 = data[10,:]
    # centres1 = np.linspace(4,20,10)
    print(centres1)

    # the width (analogous to standard deviation) of the basis functions
    scale = 47
    print("centres = %r" % (centres1,))
    print("scale = %r" % (scale,))
    # create the feature mapping
    feature_mapping = construct_rbf_feature_mapping(centres1,scale)
    # plot the basis functions themselves for reference

    # sample number of data-points: inputs and targets
    # define the noise precision of our data
    beta = (1/0.01)**2
    # now construct the design matrix for the inputs
    designmtx = feature_mapping(inputs)
    test_designmtx = feature_mapping(test_inputs)
    print(designmtx.shape)
    # the number of features is the width of this matrix
    M = designmtx.shape[1]
    # define a prior mean and covaraince matrix
    # m0 = np.random.randn(M)
    m0=np.zeros(M)
    print ("m0 equals %r" %(m0))
    alpha = 50
    S0 = alpha * np.identity(M)
    # find the posterior over weights
    mN, SN = calculate_weights_posterior(designmtx, targets, beta, m0, S0)
    # for i in range(500):
    #     mN, SN = calculate_weights_posterior(designmtx, targets, beta, mN, SN)
    
    train_error, test_error = train_and_test(
            designmtx, targets, test_designmtx, test_targets,
            mN)
    print(train_error, test_error)
    
    # cross-validation
    # train_error, test_error = cv_evaluation_linear_model(designmtx, targets, folds, mN)
    # print(train_error, test_error, np.mean(train_error), np.mean(test_error))
            
            
    # the posterior mean (also the MAP) gives the central prediction
    mean_approx = construct_feature_mapping_approx(feature_mapping, mN)
    fig, ax, lines = plot_function_data_and_approximation(
        mean_approx, test_inputs, test_targets)
    ax.legend(lines, ['Prediction', 'True value'])
    ax.set_xticks([])
    ax.set_ylabel("Quality")
    fig.suptitle('Prediction vlaue against True value', fontsize=10)
    fig.savefig("regression_bayesian_rbf.pdf", fmt="pdf")
    

    # search the optimum alpha for baysian model regression
    train_inputs = data[0:960,:]
    train_targets = data_targets[0:960]
    test_inputs = data[960:1300,:]
    test_targets = data_targets[960:1300]
    # folds = create_cv_folds(train_inputs.shape[0], num_folds)
    alphas = np.logspace(1,3)

    # convert the raw inputs into feature vectors (construct design matrices)
    # train_errors = np.empty(alphas.size)
    # test_errors = np.empty(alphas.size)
    train_errors = []
    test_errors = []
    for a,alpha in enumerate(alphas):
        # we must construct the feature mapping anew for each scale
        feature_mapping = construct_rbf_feature_mapping(centres1,scale)  
        train_designmtx = feature_mapping(train_inputs)
        test_designmtx = feature_mapping(test_inputs)
        
        beta = (1/0.01)**2
        M = train_designmtx.shape[1]
        # define a prior mean and covaraince matrix
        m0 = np.zeros(M)
        
        S0 = alpha * np.identity(M)
        # find the posterior over weights
        mN, SN = calculate_weights_posterior(train_designmtx, train_targets, beta, m0, S0)
    
        # evaluate the test and train error for this regularisation parameter
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets,
            mN)
        train_errors.append(train_error)
        test_errors.append(test_error)
        # train_error, test_error = cv_evaluation_linear_model(train_designmtx, train_targets, folds, mN)
        # train_errors[a] = np.mean(train_error)
        # test_errors[a] = np.mean(test_error)
    # plot the results
    min_error=np.min(test_errors)
    min_error_index=np.argmin(test_errors)
    fig, ax = plot_train_test_errors(
        "alpha", alphas, train_errors, test_errors)   
    fig.suptitle('Alpha vs Error in Bayesian', fontsize=10) 
    ax.plot(alphas[min_error_index],min_error,"ro")
    # ax.text(scales[min_error_index],min_error,(str(scales[min_error_index]),str(min_error))) 
    ax.annotate((str(alphas[min_error_index]),str(min_error)),xy=(alphas[min_error_index],min_error),xytext=(alphas[min_error_index]+0.01,min_error+0.01),arrowprops=dict(facecolor='green',shrink=0.1))   
    ax.set_xscale('log')
    fig.savefig("alpha.pdf", fmt="pdf")
    
    



    # search the optimum beta for baysian model regression
    train_inputs = data[0:960,:]
    train_targets = data_targets[0:960]
    test_inputs = data[960:1300,:]
    test_targets = data_targets[960:1300]
    # folds = create_cv_folds(train_inputs.shape[0], num_folds)
    betas = (1./np.logspace(-3,1))**2

    # convert the raw inputs into feature vectors (construct design matrices)
    # train_errors = np.empty(betas.size)
    # test_errors = np.empty(betas.size)
    train_errors = []
    test_errors = []
    for b,beta in enumerate(betas):
        # we must construct the feature mapping anew for each scale
        feature_mapping = construct_rbf_feature_mapping(centres1,scale)  
        train_designmtx = feature_mapping(train_inputs)
        test_designmtx = feature_mapping(test_inputs)
        
        M = train_designmtx.shape[1]
        # define a prior mean and covaraince matrix
        m0 = np.zeros(M)
        alpha=50
        S0 = alpha * np.identity(M)
        # find the posterior over weights
        mN, SN = calculate_weights_posterior(train_designmtx, train_targets, beta, m0, S0)
    
        # evaluate the test and train error for this regularisation parameter
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets,
            mN)
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        # train_error, test_error = cv_evaluation_linear_model(train_designmtx, train_targets, folds, mN)
        # train_errors[b] = np.mean(train_error)
        # test_errors[b] = np.mean(test_error)
    # plot the results
    min_error=np.min(test_errors)
    min_error_index=np.argmin(test_errors)
    fig, ax = plot_train_test_errors(
        "beta", betas, train_errors, test_errors)   
    fig.suptitle('Beta vs Error in Bayesian', fontsize=10) 
    ax.plot(betas[min_error_index],min_error,"ro")
    # ax.text(scales[min_error_index],min_error,(str(scales[min_error_index]),str(min_error))) 
    ax.annotate((str(betas[min_error_index]),str(min_error)),xy=(betas[min_error_index],min_error),xytext=(betas[min_error_index]+0.05,min_error+0.05),arrowprops=dict(facecolor='green',shrink=0.1))   
    ax.set_xscale('log')
    fig.savefig("beta.pdf", fmt="pdf")
    
    
    

    # search the optimum scale for baysian model regression
    scales = np.logspace(0.5,3)
    train_inputs = data[0:960,:]
    train_targets = data_targets[0:960]
    test_inputs = data[960:1300,:]
    test_targets = data_targets[960:1300]
    # folds = create_cv_folds(train_inputs.shape[0], num_folds)

    # convert the raw inputs into feature vectors (construct design matrices)
    # train_errors = np.empty(scales.size)
    # test_errors = np.empty(scales.size)
    train_errors = []
    test_errors = []
    for j,scale in enumerate(scales):
        # we must construct the feature mapping anew for each scale
        feature_mapping = construct_rbf_feature_mapping(centres1,scale)  
        train_designmtx = feature_mapping(train_inputs)
        test_designmtx = feature_mapping(test_inputs)
        
        beta = (1./0.01)**2
        M = train_designmtx.shape[1]
        # define a prior mean and covaraince matrix
        m0 = np.zeros(M)
        alpha = 50
        S0 = alpha * np.identity(M)
        # find the posterior over weights
        mN, SN = calculate_weights_posterior(train_designmtx, train_targets, beta, m0, S0)
    
        # evaluate the test and train error for this regularisation parameter
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets,
            mN)
        # train_error, test_error = cv_evaluation_linear_model(train_designmtx, train_targets, folds, mN)
        # train_errors[j] = np.mean(train_error)
        # test_errors[j] = np.mean(test_error)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    min_error=np.min(test_errors)
    min_error_index=np.argmin(test_errors)
    fig, ax = plot_train_test_errors(
        "scale", scales, train_errors, test_errors)   
    fig.suptitle('Scale vs Error in Bayesian', fontsize=10) 
    ax.plot(scales[min_error_index],min_error,"ro")
    # ax.text(scales[min_error_index],min_error,(str(scales[min_error_index]),str(min_error))) 
    ax.annotate((str(scales[min_error_index]),str(min_error)),xy=(scales[min_error_index],min_error),xytext=(scales[min_error_index]+0.2,min_error+0.2),arrowprops=dict(facecolor='green',shrink=0.1))   
    ax.set_xscale('log')
    fig.savefig("scale.pdf", fmt="pdf")
    
    
    
    
    
    # Here we vary the number of centres and evaluate the performance
    scale = 60
    train_inputs = data[0:960,:]
    train_targets = data_targets[0:960]
    test_inputs = data[960:1300,:]
    test_targets = data_targets[960:1300]
    # folds = create_cv_folds(train_inputs.shape[0], num_folds)
    cent_parts=np.linspace(0.05,0.8,16)
    # train_errors = np.empty(cent_parts.size)
    # test_errors = np.empty(cent_parts.size)
    train_errors = []
    test_errors = []
    N=train_inputs.shape[0]
    
    for n,cent_part in enumerate(cent_parts):
        # we must construct the feature mapping anew for each number of centres
        centres1 = train_inputs[np.random.choice([False,True], size=N, p=[1-cent_part,cent_part]),:]
        
        feature_mapping = construct_rbf_feature_mapping(centres1,scale)  
        train_designmtx = feature_mapping(train_inputs)
        test_designmtx = feature_mapping(test_inputs)
        # evaluate the test and train error for this regularisation parameter
        
        M = train_designmtx.shape[1]
        # define a prior mean and covaraince matrix
        m0 = np.zeros(M)
        beta = (1./0.01)**2
        alpha = 50
        S0 = alpha * np.identity(M)
        # find the posterior over weights
        mN, SN = calculate_weights_posterior(train_designmtx, train_targets, beta, m0, S0)
        
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets,
            mN)
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        # train_error, test_error = cv_evaluation_linear_model(train_designmtx, train_targets, folds, mN)
        # train_errors[n] = np.mean(train_error)
        # test_errors[n] = np.mean(test_error)
    # plot the results
    min_error=np.min(test_errors)
    min_error_index=np.argmin(test_errors)
    fig, ax = plot_train_test_errors(
        "Num. Centres", cent_parts, train_errors, test_errors)
    fig.suptitle('Num. Centres vs Error in Bayesian', fontsize=10)
    ax.plot(cent_parts[min_error_index],min_error,"ro")
    ax.text(cent_parts[min_error_index],min_error,(str(cent_parts[min_error_index]),str(min_error))) 
    fig.savefig("Num. centres.pdf", fmt="pdf")

    plt.show()

def display_basis_functions(feature_mapping):
    datamtx = np.linspace(0,1, 51)
    designmtx = feature_mapping(datamtx)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for colid in range(designmtx.shape[1]):
      ax.plot(datamtx, designmtx[:,colid])
    ax.set_xlim([0,1])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

def plot_function_data_and_approximation(
        predict_func, inputs, targets, linewidth=3, xlim=None,
        **kwargs):
    """
    Plot a function, some associated regression data and an approximation
    in a given range

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets
    true_func - the true function
    <for optional arguments see plot_function_and_data>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    # fig, ax, lines = plot_function_and_data(
    #     inputs, targets, true_func, linewidth=linewidth, xlim=xlim, **kwargs)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    line, =ax.plot(inputs[:,1], targets, 'bo', markersize=3)
    xs = np.linspace(7.5, 15, 101)
    # ys = predict_func(xs)
    ys = predict_func(inputs)

    line, = ax.plot(inputs[:,1], ys, 'r+', markersize=3)
    # lines.append(line)
    lines=[line]
    return fig, ax, lines
    
def train_and_test(
        train_inputs, train_targets, test_inputs, test_targets, mN):
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
    
    from regression_models import linear_model_predict
    from regression_train_test import root_mean_squared_error
    # predictions are linear functions of the inputs, we evaluate those here
    train_predicts = np.around(linear_model_predict(train_inputs, mN))
    test_predicts = np.around(linear_model_predict(test_inputs, mN))
    # evaluate the error between the predictions and true targets on both sets
    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)
    return train_error, test_error



