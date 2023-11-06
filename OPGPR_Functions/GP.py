
import autograd.numpy as np
from autograd import value_and_grad
from Utilities import kernel, stochastic_update_Adam
import timeit


class OPGPR:
    def __init__(self, X, y, Z, clusters_population, M, hyp, mt_hyp, vt_hyp, m, S, max_iter, lrate):
        # A chunk of data
        self.X_new = X
        self.y_new = y

        # The most recent optimal parameter values
        self.M = M
        self.Z = Z
        self.m = m
        self.S = S
        self.hyp = hyp
        self.best_m = m
        self.best_S = S

        self.clusters_population = clusters_population
        self.jitter_cov = 1e-8
        
        # Adam optimizer parameters
        self.mt_hyp = mt_hyp
        self.vt_hyp = vt_hyp
        self.best_mt_hyp = mt_hyp
        self.best_vt_hyp = vt_hyp
        self.lrate = lrate
        self.max_iter = max_iter



    def train(self):
        best_mt_hyp = self.best_mt_hyp
        best_vt_hyp = self.best_vt_hyp
        best_m = self.best_m
        best_S = self.best_S
        best_nlml = 1e+10
        print("New data arrived:")
        print("Total number of parameters: %d" % (self.hyp.shape[0]))
        
        # Gradients from autograd 
        NLML = value_and_grad(self.negative_log_likelihood)
        
        start_time = timeit.default_timer()
        for i in range(1, self.max_iter+1):

            # Compute likelihood
            nlml, D_NLML = NLML(self.hyp)
            
            # Update hyper-parameters
            self.hyp, self.mt_hyp, self.vt_hyp = stochastic_update_Adam(self.hyp, D_NLML, self.mt_hyp, self.vt_hyp, self.lrate, i)
            
            elapsed = timeit.default_timer() - start_time
            if nlml < best_nlml:
                best_nlml = nlml
                best_m = self.m_temp
                best_S = self.S_temp
                best_mt_hyp = self.mt_hyp
                best_vt_hyp = self.vt_hyp

            print('Iteration: %d, NLML: %.2f, best_NLML: %.2f, Time: %.2f' % (i, nlml, best_nlml, elapsed))
            start_time = timeit.default_timer()
        self.m = best_m
        self.S = best_S
        self.mt_hyp = best_mt_hyp
        self.vt_hyp = best_vt_hyp
        return self
    
    def negative_log_likelihood(self, hyp):
        # Initializing parameters and data
        M = self.M
        Z = self.Z
        m = self.m
        S = self.S
        X_new = self.X_new
        y_new = self.y_new
        jitter_cov_matrix = self.jitter_cov
        N = X_new.shape[0]

        # Extracting noise level information
        logsigma_n = hyp[-1]
        sigma_n = np.exp(logsigma_n)

        # Calculating the inverse of the covariance matrix K_inv
        K = kernel(Z, Z, hyp[:-1])
        K_inv = np.linalg.solve(K + np.eye(M) * jitter_cov_matrix, np.eye(M))

        # Computing the mean vector MU
        q_n = kernel(Z, X_new, hyp[:-1])
        MU = np.matmul(q_n.T, np.matmul(K_inv, m))

        # Evaluating the covariance matrix COV
        K_inv_q_n = np.matmul(K_inv, q_n)
        cov_matrix = kernel(X_new, X_new, hyp[:-1]) - np.matmul(q_n.T, np.matmul(K_inv, q_n)) + \
              np.matmul(K_inv_q_n.T, np.matmul(S, K_inv_q_n))

        # Calculating the inverse of the covariance matrix COV_inv
        cov_matrix_inv = np.linalg.solve(cov_matrix + np.eye(N) * sigma_n + np.eye(N) * jitter_cov_matrix, np.eye(N))

        # Computing the covariance between Z and X, cov_matrix_ZX
        cov_matrix_ZX = np.matmul(S, K_inv_q_n)

        # Updating the mean vector (m) and covariance matrix (S)
        m = m + np.matmul(cov_matrix_ZX, np.matmul(cov_matrix_inv, y_new - MU))
        S = S - np.matmul(cov_matrix_ZX, np.matmul(cov_matrix_inv, cov_matrix_ZX.T))

        # Storing temporary values for m and S
        self.m_temp = m
        self.S_temp = S

        # Calculating the Negative Log Marginal Likelihood (NLML)
        L = np.linalg.cholesky(K + np.eye(M) * jitter_cov_matrix)
        NLML = 0.5 * np.matmul(m.T, np.matmul(K_inv, m)) + np.sum(np.log(np.diag(L))) + 0.5 * np.log(2. * np.pi) * M

        return NLML[0, 0]

    def predict(self, X_star):
        M = self.M
        Z = self.Z
        m = self.m
        S = self.S
        hyp = self.hyp
        jitter_cov_matrix = self.jitter_cov

        # Calculate the inverse of the covariance matrix K_inv
        K = kernel(Z, Z, hyp[:-1])  # Compute the kernel matrix
        K_inv = np.linalg.solve(K + np.eye(M) * jitter_cov_matrix, np.eye(M))  # Solve for K_inv

        N_star = X_star.shape[0]  # Number of new data points
        partitions_size = 10000  # Partition size for efficient processing
        (number_of_partitions, remainder_partition) = divmod(N_star, partitions_size)

        # Initialize arrays to store mean and variance predictions
        mean_star = np.zeros((N_star, 1))
        var_star = np.zeros((N_star, 1))

        for partition in range(0, number_of_partitions):
            print("Predicting partition: %d" % (partition))
            idx_1 = partition * partitions_size
            idx_2 = (partition + 1) * partitions_size

            # Compute the mean (mu) for the current partition
            q_n = kernel(Z, X_star[idx_1:idx_2, :], hyp[:-1])
            mu = np.matmul(q_n.T, np.matmul(K_inv, m))
            mean_star[idx_1:idx_2, 0:1] = mu._value

            # Compute the covariance (cov) for the current partition
            K_inv_q_n = np.matmul(K_inv, q_n)
            cov_matrix = kernel(X_star[idx_1:idx_2, :], X_star[idx_1:idx_2, :], hyp[:-1]) - \
                         np.matmul(q_n.T, np.matmul(K_inv, q_n)) + np.matmul(K_inv_q_n.T, np.matmul(S, K_inv_q_n))
            var = np.abs(np.diag(cov_matrix)) + np.exp(hyp[-1])
            var_star[idx_1:idx_2, 0] = var._value

        print("Predicting the last partition")
        idx_1 = number_of_partitions * partitions_size
        idx_2 = number_of_partitions * partitions_size + remainder_partition

        # Compute the mean (mu) for the last partition
        q_n = kernel(Z, X_star[idx_1:idx_2, :], hyp[:-1])
        mu = np.matmul(q_n.T, np.matmul(K_inv, m))
        mean_star[idx_1:idx_2, 0:1] = mu._value

        # Compute the covariance (cov) for the last partition
        K_inv_q_n = np.matmul(K_inv, q_n)
        cov_matrix = kernel(X_star[idx_1:idx_2, :], X_star[idx_1:idx_2, :], hyp[:-1]) - \
                     np.matmul(q_n.T, np.matmul(K_inv, q_n)) + np.matmul(K_inv_q_n.T, np.matmul(S, K_inv_q_n))
        var = np.abs(np.diag(cov_matrix)) + np.exp(hyp[-1])
        var_star[idx_1:idx_2, 0] = var._value

        return mean_star, var_star
