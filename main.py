import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model
import pandas as pd



def load_and_normalize_data():
    # load the numpy arrays inputs and labels from the data folder
    # TODO
    X, y = np.loadtxt('data/inputs.txt'), np.loadtxt('data/labels.txt')

    # normalize the target y
    # TODO
    y = (y)/np.std(y)
    return X, y


def data_summary(X, y):

    # return several statistics of the data
    # TODO
    X_mean = np.mean(X)
    X_std = np.std(X)
    y_mean = np.mean(y)
    y_std = np.std(y)
    X_min = np.min(X)
    X_max = np.max(X)

    return {'X_mean': X_mean,
            'X_std': X_std, 
            'X_min': X_min, 
            'X_max': X_max, 
            'y_mean': y_mean, 
            'y_std': y_std}


def data_split(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.25, random_state=4)

    return X_train, X_test, y_train, y_test, X_validation, y_validation


def fit_linear_regression(X, y, lmbda=0.0, regularization=None):
    """
    Fit a ridge regression model to the data, with regularization parameter lambda and a given
    regularization method.
    If the selected regularization method is None, fit a linear regression model without a regularizer.

    !! Do not fit the intercept in all cases.

    y = wx+c

    X: 2D numpy array of shape (n_samples, n_features)
    y: 1D numpy array of shape (n_samples,)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None

    Returns: The coefficients and intercept of the fitted model.
    """

    # TODO: use the sklearn linear_model module
    if regularization in ['ridge']:
        lr = linear_model.Ridge(alpha = lmbda, fit_intercept=False).fit(X,y)
        w = lr.coef_  # coefficients
        c = lr.intercept_ # intercept
    elif regularization in ['lasso']:
        lr = linear_model.Lasso(alpha = lmbda, fit_intercept=False).fit(X,y)
        w = lr.coef_  # coefficients
        c = lr.intercept_ # intercept
    else:
        lr = linear_model.Ridge(alpha = 0.0, fit_intercept=False).fit(X,y)
        w = lr.coef_ 
        c = lr.intercept_

    return w, c


def predict(X, w, c):
    """
    Return a linear model prediction for the data X.

    X: 2D numpy array of shape (n_samples, n_features) data
    w: 1D numpy array of shape (n_features,) coefficients
    c: float intercept

    Returns: 1D numpy array of shape (n_samples,)
    """
    # TODO
    y_pred = np.dot(X,w) +  np.ones(len(X))*c
    return y_pred


def mse(y_pred, y):
    """
    Return the mean squared error between the predictions and the true labels.

    y_pred: 1D numpy array of shape (n_samples,)
    y: 1D numpy array of shape (n_samples,)

    Returns: float
    """

    # TODO
    MSE = np.round((np.linalg.norm(y-y_pred))**2/len(y),15)

    return MSE



def fit_predict_test(X_train, y_train, X_test, y_test, lmbda=0.0, regularization=None):
    """
    Fit a linear regression model, possibly with L2 regularization, to the training data.
    Record the training and testing MSEs.
    Use methods you wrote before

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbda: float, regularization parameter

    Returns: The coefficients and intercept of the fitted model, the training and testing MSEs in a dictionary.
    """

    w, c = fit_linear_regression(X_train,y_train,lmbda,regularization) # TODO

    results = {
        'mse_train': mse(y_train,predict(X_train,w,c)),
        'mse_test': mse(y_test,predict(X_test,w,c)),
        'lmbda': lmbda,
        'w': w,
        'c': c
    }

    return results


def plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=0.0, regularization=None, filename=None):
    """
    Plot the training and testing MSEs against the regularization parameter alpha.
    Use the functions you just wrote.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    alphas: list of values, the dataset percentage to be checked (alpha=n/d)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """
    
    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here

    train_set_sizes = np.array(np.multiply((X_train.shape[1]),alphas),dtype = int)
    mse_train = []
    mse_test = []
    for n in train_set_sizes:
        X_train_ = X_train[:n]
        y_train_ = y_train[:n]
        results = fit_predict_test(X_train_, y_train_, X_test, y_test, lmbda, regularization)
        mse_train.append(results['mse_train'])
        mse_test.append(results['mse_test']) 

    plt.plot(alphas,mse_train,'k',label = 'Train error')
    plt.plot(alphas,mse_test,'b',label = 'Test error')
    plt.xlabel(r'$\alpha=\frac{n}{d}$')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.savefig(f'results/{filename}.png')
    plt.clf()



def plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs, regularization='ridge', filename=None):
    """
    Plot the coefficients of the fitted model against the regularization parameter lambda.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbdas: list of values, the regularization parameter
    plot_coefs: list of integers, the coefficients of w to be plotted
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """

    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here
    plt.figure()
    for coef in plot_coefs:
        W = []
        for lambeda in lmbdas:
            results = fit_predict_test(X_train, y_train, X_test, y_test, lambeda, 'ridge')
            w = results['w']
            W.append(w[coef])
        plt.plot(lmbdas,W,label=f'$w{coef}$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Weights value')
    plt.legend(loc='best')
    plt.savefig(f'results/{filename}.png')
    plt.clf()

def add_poly_features(X):
    """
    Add squared features to the data X and return a new vector X_poly that contains
    X_poly[i,j]   = X[i,j]
    X_poly[i,2*j] = X[i,j]^2

    X: 2D numpy array of shape (n_samples, n_features)

    Returns: 2D numpy array of shape (n_samples, 2 * n_features ) with the normal and squared features
    """

    # TODO
    X_square = np.array([i**2 for i in X])
    X_poly = np.hstack((X,X_square))

    return X_poly

def optimize_lambda(X_train,X_test,X_validation,y_train,y_test,y_validation, lmbdas,filename):
    """
    Optimize the regularization parameter lambda for the training data for ridge over the validation error and plot the validation error against the parameter lambda
    Show the best parameter lambda and the corresponding test error (not validation error!).

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    X_validation: 2D numpy array of shape (n_validation_samples, n_features)
    y_validation: 1D numpy array of shape (n_validation_samples,)
    lmbdas: list of values, the regularization parameter, the bounds of the optimization range

    Returns: The best regularization parameter and the corresponding test error (!= validation error).
    """

    regularization='ridge'

    # TODO: Experiment code goes heres
    MSE =[]
    for i in lmbdas:
        w,c = fit_linear_regression(X_train, y_train, i, 'ridge')
        mse_v = mse(y_validation,predict(X_validation,w,c))
        MSE.append(mse_v)
    best_lmbda = lmbdas[np.argmin(MSE)]
    final = fit_predict_test(X_train, y_train, X_test, y_test, best_lmbda, 'ridge')
    best_test_mse = final['mse_test']

    # TODO: Plotting code goes here
    plt.figure()
    plt.plot(lmbdas,MSE,label='Validation error')
    plt.axvline(best_lmbda, color='red', label=f'Best Lambda : {np.round(best_lmbda,2)}')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.legend(loc = 'best')
    plt.savefig(f'results/{filename}.png')
    plt.clf()

    return best_test_mse, best_lmbda
  

if __name__ == "__main__":

    """ 
    !!!!! DO NOT CHANGE THE NAME OF THE PLOT FILES !!!!!
    They need to show in the Readme.md when you submit your code.

    It is executed when you run the script from the command line.
    'conda activate ml4phys-a1'
    'python main.py'

    This already includes the code for generating all the relevant plots.
    You need to fill in the ...
    """

    ## Exercise 1.
    # Load the data
    X, y = load_and_normalize_data()
    print("Successfully loaded and normalized data.")
    print(data_summary(X, y))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, X_validation, y_validation = data_split(X, y)

    n, d = X_train.shape

    ## Exercise 4.
    # Plot the learning curves
    print('Plotting dataset size vs. mse curve ...')
    alphas = np.linspace(2,6,20) 
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas,0.0,'none', filename='dataset_size_vs_mse')

    print('Plotting dataset size vs. mse curve for L2 ...')
    alphas = np.linspace(0.1,2.5,20)
    lmbda = 0.001
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas,lmbda,'ridge', filename='dataset_size_vs_mse_l2=001')

    lmbda = 10.0
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas,lmbda,'ridge', filename='dataset_size_vs_mse_l2=10')

    ## Exercise 5.
    print('Plotting regularizer vs. coefficient curve...')
    lmbdas = np.logspace(-3,3,100)
    plot_coefs = [0,3,7,8]
    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs,'ridge', filename='regularizer_vs_coefficients_Ridge')
    

    ## Exercise 6.
    print('Find the optimal parameters for the Ridge regression...')
    lmbdas = np.arange(0.1,100,0.01)
    n_ = 50
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='optimal_lambda_ridge_n50')
    print(n_, lmbda, gen_error)
    n_ = 150
    lmbda, gen_error = optimize_lambda(X_train[:n_],X_test,X_validation,y_train[:n_],y_test,y_validation, lmbdas,filename='optimal_lambda_ridge_n150')
    print(n_, lmbda, gen_error)
    print()

    ## Exercise 7.
    X_train_poly = add_poly_features(X_train) 
    X_test_poly = add_poly_features(X_test) 

    lmbdas = np.arange(0.0001,1.2,0.001)
    plot_coeffs = [0,2,6,7]
    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coeffs,'lasso', filename='regularizer_vs_coefficients_LASSO')
    
    plot_coeffs = [0,2,6,7,129,154]
    plot_regularizer_vs_coefficients(X_train_poly, y_train, X_test_poly, y_test, lmbdas,  plot_coeffs,'lasso', filename='regularizer_vs_coefficients_LASSO_polyfeat')

    print('Done. All results saved in the results folder.')

