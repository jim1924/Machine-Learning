# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
from regression_train_test import create_cv_folds
from regression_cross_validation import evaluate_reg_param
from exploratory_plots import exploratory_plots
from linear_regression import linear_regression_entry_point
from bayesian_regression import bayesian_regression_entry_point
from rbf_regression import parameter_search_rbf_without_cross
from rbf_regression import parameter_search_rbf_cross
from rbf_regression import simple_linear_regression
from rbf_regression import regression_with_regularization
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from kNN_regression import kNN_entry_point


def main(ifname=None, delimiter=None, columns=None):
    delimiter=';'
    columns=np.arange(12)
    if ifname is None:
        ifname='datafile.csv'
    data, field_names = import_data(ifname, delimiter=delimiter, has_header=True, columns=columns)
    targets=data[:,-1]
    inputs=data[:,0:11]
    
    #We decided that the test fraction will be 0.2
    test_fraction=0.2
    #np.random.seed(5)
    #let's leave 20% out for the 
    train_part,test_part=train_and_test_split(data.shape[0], test_fraction)
    train_inputs, train_targets, test_inputs, test_targets=train_and_test_partition(inputs, targets, train_part, test_part)
    
    # get the cross-validation folds
    num_folds = 5
    folds = create_cv_folds(train_inputs.shape[0], num_folds) # this is just an array of arrays where folds[0][0]= [true,false,false] and folds[0][1]=[false,true,true]
    

    
    
    #first of all let's plot some exploratory plots
    exploratory_plots()
    
    
    
    #Now, let's try some linear regression
    linear_regression_entry_point(field_names,train_inputs,train_targets,folds,test_fraction)
        
        
    #Now, let's see the performance of the bayesian regression
    bayesian_regression_entry_point(data)
    
    #Let's see how the kNN model will behave
    kNN_entry_point(data,field_names)
 
 
    #Finally, let's see how the RBF model will behave
    train_error_linear,test_error_linear=simple_linear_regression(train_inputs,train_targets,folds,test_fraction,test_inputs,test_targets)
    #RBF regression with normalisation but without cross validation
    parameter_search_rbf_without_cross(train_inputs, train_targets, test_fraction,test_error_linear,normalize=True)
    
    #RBF regression with cross-validation and normalisation
    parameter_search_rbf_cross(train_inputs, train_targets,folds,test_error_linear,test_inputs,test_targets)
    
    #RBF regression with cross-validation but without normalisation
    parameter_search_rbf_cross(train_inputs, train_targets, folds,test_error_linear,test_inputs,test_targets,normalize=False)
    
    
    plt.show()
    
    
        
        
 
def import_data(ifname, delimiter=None, has_header=False, columns=None):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of 
    floating point numbers. If the import file has a header then this should
    be specified, and the field names will be returned as the second argument.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)

    returns
    -------
    data_as_array -- the data as a numpy.array object  
    field_names -- if file has header, then this is a list of strings of the
      the field names imported. Otherwise, it is a None object.
    """
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if has_header:
            field_names = next(datareader)
            
            print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        print("There are %d entries" % len(data))
        print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not field_names is None:
        # thin the associated field names if needed
        field_names=np.array(field_names)
        field_names = [field_names[c] for c in columns]
    # return this array to caller (and field_names if provided)
    return data_as_array, field_names
    
    
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        main() # calls the main function with no arguments     
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1])
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(ifname=sys.argv[1], delimiter=sys.argv[2])
    elif len(sys.argv) == 4:
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(","))) 
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)
        
        

        

    
    
