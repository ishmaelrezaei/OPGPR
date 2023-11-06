"""Online Parametric Gaussian Process Regression
    Author: Esmaeil Rezaei, 10/02/2022

Example: 2D toy example

Steps:
    1- run the "DataGenerator.Py" to generate the data
    2 - run the "Main.py"
"""
import sys
sys.path.insert(0, '../OPGPR_Functions/')

from sklearn.cluster import KMeans
from Utilities import Normalize, KMeans_cluster_centers, KMeans_center_update, kernel, update_parameters, plot_OneVar_example
import autograd.numpy as np
from GP import OPGPR
import pickle
import pandas as pd
import os                    # Import the operating system module
import time
import shutil
from natsort import natsorted


if __name__ == "__main__":

    # Initial paramteres
    M = 8
    StartFromScratch = 1   # set 1 to start from initial training and 0 to run only online part
    update_file = 'UpdatedParam'

    if StartFromScratch == 1 and os.path.exists(update_file + '.pkl'):
        os.remove(update_file + '.pkl')

    # Run the initial training if have not done
    if not os.path.exists(update_file + '.pkl'):
        if os.path.exists('FIGs/'):
            shutil.rmtree('FIGs/')
            time.sleep(10)
        os.makedirs('FIGs/')

        # Import the data
        path = 'MyInitialData/'

        filename_X = 'MyInitialData/X_initial.csv'
        X = pd.read_csv(filename_X)

        filename_Y = 'MyInitialData/Y_initial.csv'
        Y = pd.read_csv(filename_Y)

        X = X.values.reshape(-1, 1)
        Y = Y.values.reshape(-1, 1)

        (N, D) = X.shape

        X_m = np.mean(X, axis=0)
        X_s = np.std(X, axis=0)
        X = Normalize(X, X_m, X_s)

        y_m = np.mean(Y, axis=0)
        y_s = np.std(Y, axis=0)
        Y = Normalize(Y, y_m, y_s)

        idx = np.random.choice(len(X), len(X), replace=False)
        kmeans = KMeans(n_clusters=M, random_state=0, n_init=10).fit(X[idx])
        cluster_centers_, clusters_population = KMeans_cluster_centers(kmeans.labels_, data=X[idx], M=M)
        Z = cluster_centers_

        hyp = np.log(np.ones(D + 1))
        logsigma_n = np.array([-4.0])
        hyp = np.concatenate([hyp, logsigma_n])

        m = np.zeros((M, 1))
        S = kernel(Z, Z, hyp[:-1])

        # Adam optimizer parameters
        mt_hyp = np.zeros(hyp.shape)
        vt_hyp = np.zeros(hyp.shape)

        # Model creation
        pgp = OPGPR(X, Y, Z, clusters_population, M, hyp, mt_hyp, vt_hyp, m, S, max_iter=20, lrate=1e-3)

        # Training
        self = pgp.train()

        Aug_X = X
        Aug_Y = Y

        update_parameters(self, "../OneVar_OPGPR/" + update_file, X_m, X_s, y_m, y_s, Aug_X, Aug_Y)

        # Import the data for prediction
        filename_X = 'MyPredictionData/X_pred.csv'
        X = pd.read_csv(filename_X)
        filename_Y = 'MyPredictionData/Y_pred.csv'
        Y = pd.read_csv(filename_Y)

        X = X.values.reshape(-1, 1)
        Y = Y.values.reshape(-1, 1)

        (N, D) = X.shape

        # Normalize Y scale and offset
        X = Normalize(X, X_m, X_s)
        Y = Normalize(Y, y_m, y_s)

        # Prediction
        mean_star, var_star = pgp.predict(X)

        # Plot Results
        Z = pgp.Z
        m = pgp.m._value

        plot_OneVar_example(X, Y, mean_star, var_star, M, Z, m, clusters_population, Aug_X, Aug_Y)


if os.path.exists(update_file + '.pkl'):
    X_list = natsorted([filename for filename in os.listdir('MyOnlineData/X') if filename.endswith('.csv')])
    Y_list = natsorted([filename for filename in os.listdir('MyOnlineData/Y') if filename.endswith('.csv')])
    for (i, j) in zip(X_list, Y_list):
        filename_X = 'MyOnlineData/X/' + i
        X = pd.read_csv(filename_X)
        filename_Y = 'MyOnlineData/Y/' + j
        Y = pd.read_csv(filename_Y)

        with open(update_file + '.pkl', 'rb') as file:
            M, Z, m, S, hyp, mt_hyp, vt_hyp, clusters_population, X_m, X_s, y_m, y_s, Aug_X, Aug_Y = pickle.load(
                file)
        X = X.values.reshape(-1, 1)
        Y = Y.values.reshape(-1, 1)

        (N, D) = X.shape

        # Normalize Y scale and offset
        X = Normalize(X, X_m, X_s)
        Y = Normalize(Y, y_m, y_s)

        Z, clusters_population = KMeans_center_update(X, Z, M, clusters_population)

        # Model creation
        pgp = OPGPR(X, Y, Z, clusters_population, M, hyp, mt_hyp, vt_hyp, m, S, max_iter=20, lrate=1e-3)

        # Training
        self = pgp.train()


        Aug_X = np.concatenate((Aug_X, X), axis=0)
        Aug_Y = np.concatenate((Aug_Y, Y), axis=0)
        update_parameters(self, "../OneVar_OPGPR/" + update_file, X_m, X_s, y_m, y_s, Aug_X, Aug_Y)

        with open(update_file + '.pkl', "wb") as file:
            pickle.dump((M, Z, m, S, hyp, mt_hyp, vt_hyp, clusters_population, X_m, X_s, y_m, y_s, Aug_X, Aug_Y),
                        file)
        time.sleep(2)  # Pause to let save

        # Import the data for prediction
        filename_X = 'MyPredictionData/X_pred.csv'
        X = pd.read_csv(filename_X)
        filename_Y = 'MyPredictionData/Y_pred.csv'
        Y = pd.read_csv(filename_Y)

        X = X.values.reshape(-1, 1)
        Y = Y.values.reshape(-1, 1)

        (N, D) = X.shape

        # Normalize Y scale and offset
        X = Normalize(X, X_m, X_s)
        Y = Normalize(Y, y_m, y_s)

        # Prediction
        mean_star, var_star = pgp.predict(X)

        # Plot Results
        Z = pgp.Z
        m = pgp.m._value
        plot_OneVar_example(X, Y, mean_star, var_star, M, Z, m, clusters_population, Aug_X, Aug_Y)





