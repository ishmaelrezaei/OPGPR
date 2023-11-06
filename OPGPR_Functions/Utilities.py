import autograd.numpy as np
from sklearn.cluster import KMeans
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')  # specify backend
import matplotlib.pyplot as plt
import pandas as pd
import os                    # Import the operating system module


def KMeans_cluster_centers(labels, data, M):
    (N, D) = data.shape
    cluster_centers = np.zeros(shape=(M, D))
    clusters_population = [0] * M
    for i in range(M):
        data_label_i = np.where(labels == i)
        data_0 = data[data_label_i, :][0]
        cluster_centers[i, :] = np.mean(data_0, axis=0)
        # counting population
        clusters_population[i] = np.array(data_label_i).shape[1]
    return cluster_centers, clusters_population


def KMeans_center_update(new_data, Z, M, clusters_population):
    (N, D) = Z.shape
    kmeans = KMeans(n_clusters=M, random_state=0, init=Z, n_init=1).fit(new_data)
    labels = list(kmeans.labels_)
    for i in range(N):
        count = labels.count(i)
        new_data_label_i = np.where(kmeans.labels_ == i)
        new_data_i = new_data[new_data_label_i, :][0]
        Z[i, :] = (Z[i, :]*clusters_population[i] + new_data_i.sum(axis=0))/(clusters_population[i] + count)
        clusters_population[i] = clusters_population[i] + count
    return Z, clusters_population


def kernel(X, Xp, hyp):
    output_scale = np.exp(hyp[0])
    lengthscales = np.sqrt(np.exp(hyp[1:]))
    X = X/lengthscales
    Xp = Xp/lengthscales
    X_SumSquare = np.sum(np.square(X),axis=1);
    Xp_SumSquare = np.sum(np.square(Xp),axis=1);
    mul = np.dot(X,Xp.T);
    dists = X_SumSquare[:,np.newaxis]+Xp_SumSquare-2.0*mul
    return output_scale * np.exp(-0.5 * dists)


def stochastic_update_Adam(w,grad_w,mt,vt,lrate,iteration):
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

    mt = mt*beta1 + (1.0-beta1)*grad_w;
    vt = vt*beta2 + (1.0-beta2)*grad_w**2;

    mt_hat = mt/(1.0-beta1**iteration);
    vt_hat = vt/(1.0-beta2**iteration);

    scal = 1.0/(np.sqrt(vt_hat) + epsilon);

    w = w - lrate*mt_hat*scal;
    
    return w,mt,vt

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)
     
def Denormalize(X, X_m, X_s):    
    return X_s*X + X_m


def update_parameters(self, update_file, X_m, X_s, y_m, y_s, Aug_X, Aug_Y, MSE, MSE_mean):
    M = self.M
    Z = self.Z
    m = self.m._value
    S = self.S._value
    hyp = self.hyp
    mt_hyp = self.mt_hyp
    vt_hyp = self.vt_hyp
    clusters_population = self.clusters_population
    with open(update_file + '.pkl', "wb") as file:
        pickle.dump((M, Z, m, S, hyp, mt_hyp, vt_hyp, clusters_population, X_m, X_s, y_m, y_s, Aug_X, Aug_Y, MSE, MSE_mean), file)

def plot_OneVar_example(X, Y, mean_star, var_star, M, Z, m, clusters_population, Aug_X, Aug_Y):
    plt.figure(figsize=(10, 10))
    # plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 15})
    plt.subplot(2, 1, 1)
    plt.plot(Aug_X, Aug_Y, 'b+', alpha=1)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('(A)')
    plt.legend(['%d training Data' % np.sum(clusters_population)], loc='lower left')

    plt.subplot(2, 1, 2)
    plt.plot(Z, m, 'ro', alpha=1, markersize=14)
    plt.plot(X, Y, 'b-', linewidth=2)
    plt.plot(X, mean_star, 'r--', linewidth=2)
    lower = mean_star - 2.0 * np.sqrt(var_star)
    upper = mean_star + 2.0 * np.sqrt(var_star)
    plt.fill_between(X.flatten(), lower.flatten(), upper.flatten(), facecolor='orange', alpha=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x), \overline{f}(x)$')
    plt.title('(B)')
    plt.tight_layout()
    plt.legend(['%d hypothetical data' % M, '$f(x)$', '$\overline{f}(x)$', 'Two standard deviations'],
               loc='lower left')
    # save file
    current_time = datetime.now()
    date_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"FIGs/2D_{date_string}.png"
    plt.savefig(filename, format='png', dpi=1000)



def read_airlines_data(data_dir):
    data = pd.read_csv(data_dir)
    names = [
        'ArrDelay',
        'DepTime',
        'ArrTime',
        'Distance',
        'AirTime',
        'DayOfWeek',
        'DayofMonth',
        'Month',
        'Year'
    ]
    data = data[names]
    # Remove NaN values
    data = data.dropna()

    # Convert time of day from hhmm to minutes since midnight
    data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
    data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)

    Y = data['ArrDelay'].values
    X = data[names[1:]].values
    Y = Y.reshape(-1, 1)
    return X, Y, names


def plot_airlines_delay_example(hyp, names):
    # ARD
    ARD = 1 / np.sqrt(np.exp(hyp[1:-1]))
    ARD_x = np.arange(len(ARD))

    # Plot ARD
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({'font.size': 16})
    ax.barh(ARD_x, ARD)
    ax.set_yticks(ARD_x)
    ax.set_yticklabels(names[1:])
    ax.set_xlabel('ARD weights')

    # save file
    current_time = datetime.now()
    date_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"FIGs/Delay_{date_string}.png"
    plt.savefig(filename, format='png', dpi=1000)



def data_split(data, chunk_size, directory, Airline):
    partitions = data.shape[0] // chunk_size
    for i in range(partitions):
        data_i = data.iloc[chunk_size * i:(i + 1) * chunk_size, :]
        filename = os.path.join('../Airline_Delay_Prediction/' + directory, Airline + f'_Data_{i + 1}.csv')
        data_i.to_csv(filename, index=False)
        print(directory + f': Created {i + 1}th table from {partitions}')

    data_i = data.iloc[(i + 1) * chunk_size:, :]
    filename = os.path.join('../Airline_Delay_Prediction/' + directory, Airline + f'_Data_{i + 1}.csv')
    data_i.to_csv(filename, index=False)
    print(directory + f': Created {i + 1}th table from {partitions}')
