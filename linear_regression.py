# Linear Regression Model

# import the libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
from regression_train_test import simple_evaluation_linear_model
from regression_plot import plot_train_test_errors
from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model

def linear_regression_entry_point(field_names,inputs,targets,folds,test_fraction):


    
    
    # get the number of datapoints
    N = inputs.shape[0]
    
    
    # let's inspect the data a little more
    fixed_acidity = inputs[:,0]
    volatile_acidity = inputs[:,1]
    citric_acid = inputs[:,2]
    residual_sugar = inputs[:,3]
    chlorides = inputs[:,4]
    free_sulfur_dioxide = inputs[:,5]
    total_sulfur_dioxide = inputs[:,6]
    density = inputs[:,7]
    pH = inputs[:,8]
    sulphates = inputs[:,9]
    alcohol = inputs[:,10]
    
    # compute the arithmetic means of attributes
    print("np.mean(fixed_acidity) = %r" % (np.mean(fixed_acidity),))
    print("np.mean(volatile_acidity) = %r" % (np.mean(volatile_acidity),))
    print("np.mean(citric_acid) = %r" % (np.mean(citric_acid),))
    print("np.mean(residual_sugar) = %r" % (np.mean(residual_sugar),))
    print("np.mean(chlorides) = %r" % (np.mean(chlorides),))
    print("np.mean(free_sulfur_dioxide) = %r" % (np.mean(free_sulfur_dioxide),))
    print("np.mean(total_sulfur_dioxide) = %r" % (np.mean(total_sulfur_dioxide),))
    print("np.mean(density) = %r" % (np.mean(density),))
    print("np.mean(pH) = %r" % (np.mean(pH),))
    print("np.mean(sulphates) = %r" % (np.mean(sulphates),))
    print("np.mean(alcohol) = %r" % (np.mean(alcohol),))
    
    # compute the standard deviation of attributes
    print("np.std(fixed_acidity) = %r" % (np.std(fixed_acidity),))
    print("np.std(volatile_acidity) = %r" % (np.std(volatile_acidity),))
    print("np.std(citric_acid) = %r" % (np.std(citric_acid),))
    print("np.std(residual_sugar) = %r" % (np.std(residual_sugar),))
    print("np.std(chlorides) = %r" % (np.std(chlorides),))
    print("np.std(free_sulfur_dioxide) = %r" % (np.std(free_sulfur_dioxide),))
    print("np.std(total_sulfur_dioxide) = %r" % (np.std(total_sulfur_dioxide),))
    print("np.std(density) = %r" % (np.std(density),))
    print("np.std(pH) = %r" % (np.std(pH),))
    print("np.std(sulphates) = %r" % (np.std(sulphates),))
    print("np.std(alcohol) = %r" % (np.std(alcohol),))
    
    # normalisation
    """
    There is no need to normalise the inputs as the simple linear regression
    is unaffected by this. Normalisation is when you shift the data to have 
    zero mean and variance 1 in each input dimension. Therefore this code 
    provides the error before input normalisation.
    """
    
    # evaluate the linear performance
    train_error_linear, test_error_linear = evaluate_linear_approx(inputs, targets, test_fraction)

    # simple linear regression without regularisation
    train_error_linear,test_error_linear=simple_linear_without_regularisation(inputs,targets,folds,test_fraction)
    
    #plot_without_regularisation(inputs,targets,folds,test_fraction)

    # simple linear regression with regularisation
    plot_with_regularisation(inputs,targets,folds)
    
    # ...
    evaluate_reg_param(inputs, targets, folds, reg_params=None)
    
    """
    regularisation smooths out the approximation function
    """
    
def evaluate_linear_approx(inputs, targets, test_fraction):
    # the linear performance
    train_error, test_error = simple_evaluation_linear_model(
        inputs, targets, test_fraction=test_fraction)
    print("Linear Regression:")
    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))
    return train_error, test_error

def simple_linear_without_regularisation(inputs,targets,folds,test_fraction):
    # train_error - the training error for the approximation
    # test_error - the test error for the approximation
    train_error, test_error = simple_evaluation_linear_model(inputs, targets, test_fraction=test_fraction)
    print("Linear Regression without Regularistaion:")
    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))


    # cross-validation evaluation of linear model
    train_errors, test_errors= cv_evaluation_linear_model(inputs, targets, folds)
    print("Train Errors for Linear Regression without Regularisation:")
    print("\t(train_error) = %r" % train_errors)
    # output:
    # train errors: [ 0.64904221,  0.63487906,  0.64919719,  0.64966031,  0.64325636]
    
    print ("Test Errors for Linear Regression without Regularisation:")
    print("\t(test_error) = %r" % test_errors)
    # output:
    # test errors: [ 0.63450161,  0.68883366,  0.63512469,  0.6360172 ,  0.65815973]
    
    print ("Average Mean Errors:")
    print("\t(train_error,test_error)= %r" % (( np.mean(train_errors),np.mean(test_errors)),)) 
    # output:
    # average mean errors - train error: 0.64510190051379168
    # average mean errors - test error: 0.651298048054331
    
    return np.mean(train_errors),np.mean(test_errors)



def plot_with_regularisation(inputs,targets,folds):    
    """
    Linear regression does not use a feature mapping, typically with such a 
    simple model regularisation does not have much effect. The plot is the
    same with and without regularisation. Regularisation has only a weak affect
    on simple linear regression. Using regularisation on simple linear
    regression may not be that effective.  In simple linear regression, 
    regularisation will slightly penalise functions that are further from 
    constant (i.e. with larger gradients). So it will have an effect, but 
    only a small one and that will be to give slightly lower weights than if 
    you had used least squares.
    """
    reg_params = np.logspace(-10,1)
    train_errors = []
    test_errors = []
    for reg_param in reg_params:
        print("Evaluating reg_params: " + str(reg_param))
        old_train,old_test=simple_evaluation_linear_model(inputs, targets, test_fraction=0.2, reg_param=reg_param)
        train_error, test_error = cv_evaluation_linear_model(inputs, targets, folds,reg_param=reg_param)
        # collect the errors
        train_errors.append(np.mean(train_error))
        test_errors.append(np.mean(test_error))
    # plot the results
    fig, ax = plot_train_test_errors("$\lambda$", reg_params, train_errors, test_errors)  
    plt.title('Linear Regression Model')      
    ax.set_xscale('log')

def evaluate_reg_param(inputs, targets, folds, reg_params=None):
    """
      Evaluate then plot the performance of different regularisation parameters
    """
    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-2,0)
    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(inputs, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[r] = train_mean_error
        test_mean_errors[r] = test_mean_error
        train_stdev_errors[r] = train_stdev_error
        test_stdev_errors[r] = test_stdev_error

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')
 

