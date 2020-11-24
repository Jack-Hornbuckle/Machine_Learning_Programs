import sys
import yfinance as yf
import pandas as pd
import numpy as np
from IPython.display import display
#import matplotlib.pyplot as plt


##Gradient descent functions
def normalize(col):
    return ((col - col.min())/(col.max() - col.min()))


#split the data into X and y
def X_y_Split_stock_close(data_frame):

    #get the X matrix data
    X_df = data_frame.drop(columns=['Close']) #drop the 'Close' column since thats our y column vector
    X_df = X_df.apply(normalize) #normalize the data
    X = X_df.values[:, :] #convert data into a numpy array

    #append a column vector of ones to the X array
    bias = np.ones([np.shape(X)[0], 1])
    X = np.hstack((bias, X))

    #get the y column vector data
    y_df = pd.DataFrame(data=data_frame['Close'])
    y_df = y_df.apply(normalize)
    y = y_df.values[:, :]

    #our return data
    return [X, y]


#split X and y into test and training data
def train_test_split(X, y):
    split_point = int(np.floor(((np.shape(X)[0])/3) * 2)) #get the point where we split the data
    X_train = X[0:split_point, :] #the X training data
    X_test = X[split_point:, :] #the X testing data
    y_train = y[0:split_point, :] #the y training data
    y_test = y[split_point:, :] #the y testing data

    return [X_train, y_train, X_test, y_test]

#get the mean square error
def mse(X, theta, y):
    y_tilda = np.dot(X, theta) #y_tila computed by dot product of X matrix and theta
    errors = np.subtract(y_tilda, y) #all the errors computed from subtracting y from y_tilda
    square_errors = [err**2 for err in errors]  #compute square errors
    MSE = (1/(2*np.shape(X)[0]))*np.sum(square_errors) #now plug in square errors and shape of X into mean square error equation
    return MSE

#get the gradient for all given theta values
def gradient(X, y, theta):
    #get y_tilda and mse using ame process as above
    y_tilda = np.dot(X, theta)
    errors = np.subtract(y_tilda, y)
    #compute the derivative using the formula of the derivative of the mse
    derivative = np.array([(1/np.shape(X)[0])*np.dot(X[:, i], errors) for i in range(np.shape(X)[1])])
    return derivative

#peform gradient descent
def gradient_descent(data, num_epochs=2000, alpha=0.999999):

    #get X and y then split the data into training and testing data
    [X, y] = X_y_Split_stock_close(data)
    [X_train, y_train, X_test, y_test] = train_test_split(X, y)

    theta = np.zeros([np.shape(X_train)[1], 1]) #initial theta

    #initialize training and testing error lists with current mean squared errors
    train_errors = [mse(X_train, theta, y_train)]
    test_errors = [mse(X_test, theta, y_test)]

    #loop through number of epochs and gradually adjust theta
    for i in range(num_epochs):
        derivative = gradient(X_train, y_train, theta) #get the derivatives of the mean squared error with respect to each theta value
        step = [alpha*deriv for deriv in derivative] #multiply alpha by each theta value to get the gradient descent step
        old_theta = theta.copy() #save a copy of theta
        theta = np.subtract(old_theta, step) #subtract the gradient descent step from theta
        train_errors.append(mse(X_train, theta, y_train)) #append the new mean squared error of the training data
        test_errors.append(mse(X_test, theta, y_test))   #append new mean squared error of testing data.

    return([train_errors, test_errors, theta])

def graphing(x, y, the_labels):
    '''plt.plot([i for i in range(len(train_errors))], train_errors, label = the_labels['x_label'])
    plt.plot([i for i in range(len(test_errors))], test_errors, label = the_labels['y_label'])
    plt.legend()
    plt.suptitle(the_labels['subtitle'])
    plt.show()'''

#check if an element of an array is an integer
def inputIsInt(input, index):
    return (index < len(input)) and input[index].isdigit() and input[index][0] != '0'


#check if one index is smaller than the first one.
#If so, print a usage error and exit the program
def check_indexes(index1, index2, option):
    if index2 <= index1:
        print("Usage Error: You specified the", option, "option and entered two index values but the first value was greater than or equal to than the second value")
        sys.exit()


#tell the program to print either errors or predictions
def assign_values(flag, first, last, array, index, message_part):

    #if the element in the index following the passed in index is an integer
    if inputIsInt(array, index+1):
        flag = 1 #set the printing flag to 1 to tell the program to print
        first = int(array[index+1]) #get the first index to print
        last = int(array[index+2]) if inputIsInt(array, index+2) else 0 #get the last index to print
        check_indexes(first, last, array[index]) #make sure the last index is not smaller than the first one

    #Otherwise if it not an integer, print a usage error and exit the program
    else:
        print("Usage Error: You specified the", array[index], "option did not provide the index of", message_part, "to start printing at")
        sys.exit()
    return [flag, first, second] #Return the values we found.


#the default company
company = 'GOOG'

#our default print errors, graph errors, print predictions, graph predictions, and the index flags for printing errors and predictions
print_errors, first_print_error, last_print_error, graph_errors, show_predictions, first_predict_shown, last_predict_shown, graph_predictions = 0, 0, 0, 0, 0, 0, 0, 0

#Loop through command line arguments
for j in range(len(sys.argv)):
    if (sys.argv[j] == "company"): #if we are at the "company" argument extract the abbreviations
        if ((j+1) < len(sys.argv)): #If there are more strings following our current index in the command line
            company = sys.argv[j+1] #extract the company

        else:   #Otherwie print a usage error and exit the system.
            print("Usage Error: You used the 'company' option but did not provide the company to download data on")
            sys.exit()

    if (sys.argv[j]) == "print_errors": #If "print_errors" is specified, collect the variables required for that
        [print_errors, first_print_error, last_print_error] = assign_values(print_errors, first_print_error, last_print_error, sys.argv, j, "errors")
    if sys.argv[j] == "show_predictions": #Do the same if "show_predictions" is specified
        [show_predictions, first_predict_shown, last_predict_shown] = assign_values(show_predictions, first_predict_shown, last_predict_shown, sys.argv, j, "predicted values")
    
    if sys.argv[j] == "graph_errors":   #If graph errors is specified, set the flag to graph errors
        graph_errors = 1
    if sys.argv[j] == "graph_predictions":  #Do the same to graph predictions
        graph_predictions = 1

#get data from the company we are performing data analysis on
yf_data = yf.download(company, start='2017-01-01', end='2019-01-03', progress=False) #download the data
main_data = pd.DataFrame(data=yf_data) #convert the data to a dataframe
main_data = main_data.drop(columns=['Adj Close']) #drop the 'Adj Close' column

#now train a model by performing gradient descent on the data
[train_errors, test_errors, theta] = gradient_descent(main_data)

#print theta
print("theta: ")
print(theta)

#If the user wants to see some errors
if print_errors:
    #make sure the first and last errors are in range respectively
    first_print_error = min([len(train_errors)-1, first_print_error])
    second_print_error = max([0, min([last_print_error, len(train_errors)])])
    print("train errors:")
    print(train_errors[first_print_error:last_print_error])
    print("test errors:")
    print(test_errors[first_print_error:last_print_error])

#if the user wants to graph the training errors and test errors, graph them
if graph_errors:
    graphing(train_errors, test_errors, {"x_label":"train_errors", "y_label":"test_errors", "subtitle":"MSE Errors vs Epochs"})


###If the user wants to see the algorithm predict data
if (show_predictions or graph_predictions):

    [X_norm, y_norm] = X_y_Split_stock_close(main_data) #get the original data's normalized X matrix and y vector respectively
    y_tilda_norm = np.dot(X_norm, theta)    #get the normalized y_tilda of that X matrix multiplied with theta


    #get the real y vector of the original data in numpy form
    y_df = pd.DataFrame(main_data['Close'])
    y = y_df.values[:, :]
    maximum = max(y)
    minimum = min(y)

    #get the non normalized version of y_tilda_norm
    y_tilda = (y_tilda_norm * (maximum - minimum)) + minimum #get the real y tilda of

    #get the non normalized version of the X matrix of the original data
    X_df = main_data.drop(columns=['Close'])
    X = X_df.values[:, :]

    #get the maximum and minimum for each column of that X matrix
    X_minimums = [] #the minimums of each X column
    X_maximums = []  #the maximums of each X column

    #loop through the columns
    for i in range(np.shape(X)[1]):
        X_minimums.append(min(X[:, i]))
        X_maximums.append(max(X[:, i]))

    #Get some new data using the same process we used to extract the original data
    yf_data2 = yf.download(company, start='2019-01-04', end='2020-10-01', progress=False)
    new_data = pd.DataFrame(data=yf_data2)
    new_data = new_data.drop(columns=['Adj Close'])
    X_new_df = new_data.drop(columns=['Close'])
    X_new = X_new_df.values[:, :]

    #get the column of y new
    y_new_df = pd.DataFrame(new_data['Close'])
    y_new = y_new_df.values[:, :]

    #normalize each column of this new X matrix using the maximums and minimums we collected earlier.
    for j in range(np.shape(X_new)[1]):
        old_data = X_new[:, j].copy()
        X_new[:, j] = (np.subtract(old_data, X_minimums[j]))/(X_maximums[j] - X_minimums[j])

    #append a bias term to each row
    new_bias = np.ones([np.shape(X_new)[0], 1])
    X_new = np.hstack((new_bias, X_new))

    #get the normalized version of y_tilda that comes from the dot product X_new and theta
    y_tilda_new_norm = np.dot(X_new, theta)

    #now de normalize the new version of y_tilda using the maximum and minimum values of the original data's y vector
    y_tilda_new = (y_tilda_new_norm * (maximum - minimum)) + minimum

    #if the user wants to print the predictions print the predictions
    if show_predictions:
        #make sure that the first index and last index they want to display are not out of bounds
        first_predict_shown = min([np.shape(y_new)[0]-1, first_predict_shown])
        last_predict_shown = max([0, min([last_predict_shown, np.shape(y_new)[0]])])
        #now print the predictions
        print("real y_tilda_new: ")
        print(y_tilda_new[first_predict_shown:last_predict_shown, :])
        print("real y_new: ")
        print(y_new[first_predict_shown:last_predict_shown, :])

    #if the user wants to graph the predictions then graph them
    if graph_predictions:
        graphing(y_tilda_new, y_new, {"x_label":"y_tilda_new", "y_label":"y_new", "subtitle":"y_tilda of new data vs actual y of new data"})
