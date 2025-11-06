import numpy as np
from utils.kernels import kernel_rbf
from utils.math import incr_inv



class GaussianProcessRegressorRBF: 
    """
    Implements a Gaussian Process Regressor with an RBF kernel.

    This class models a function $y = f(x) + e$, where $f(x)$ is a
    Gaussian Process and $e$ is i.i.d. subgaussian noise.

    The class is optimized for incremental updates, allowing single data points
    to be added efficiently using the block matrix inversion formula without 
    refitting the entire model. It can also optionally track the information gain 
    (mutual information) $I(y; f)$ 
    as data is added.
    """


    def __init__(self, kernel_L, sigma_sq_process, input_dim=1, keep_info_gain_estimate=False):
        """
        Initializes the Gaussian Process Regressor with an RBF kernel.

        Args:
            kernel_L (float): The length-scale parameter (L) for the RBF kernel.
                              Smaller 'L' implies a smoother function.
            sigma_sq_process (float): The variance of the observation noise.
                                      This accounts for uncertainty in the 
                                      $y$ values.
            input_dim (int, optional): The dimensionality of the input space.
                                       Defaults to 1.
            keep_info_gain_estimate (bool, optional): If True, the model will
                                       incrementally compute the information
                                       gain with each added sample.
                                       Defaults to False.
        """
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.keep_info_gain_estimate = keep_info_gain_estimate
        self.reset()


    def load_data(self, x, y):
        """
        Loads a batch of data into the model, overwriting any existing data.

        This method builds the kernel matrix and computes its inverse.

        Args:
            x (numpy.ndarray): A 2D array of input data, shape (n_samples, input_dim).
            y (numpy.ndarray): A 1D or 2D array of output data, shape (n_samples,)
                               or (n_samples, 1).
        """
        self.n_samples = x.shape[0]
        
        self.x_vect = np.array([x]).reshape(self.n_samples, self.input_dim)
        self.y_vect = np.array([y]).reshape(self.n_samples, 1)
        
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) \
                        + self.sigma_sq_process * np.eye(self.n_samples)
        self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.n_samples))
        
        if self.keep_info_gain_estimate: 
            _, value = np.linalg.slogdet(self.K_matrix/self.sigma_sq_process)
            self.info_gain = 0.5 * value
    

    def add_sample(self, x, y):
        """
        Adds a single new data point ($x, y$) to the model incrementally.

        If this is the first sample, it initializes the $1x1$ kernel matrix.
        For subsequent samples, it uses the efficient block matrix inversion
        formula to update the kernel inverse without re-computing it from scratch.

        Args:
            x (float, int, or numpy.ndarray): The new input point.
                Can be a scalar if input_dim=1, or an array of shape
                (1, input_dim).
            y (float, int, or numpy.ndarray): The new output value.
                Can be a scalar or an array of shape (1,) or (1, 1).
                
        Raises:
            ValueError: If provided $x$ or $y$ are numpy arrays with
                        incompatible shapes.
        """
        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim), "add_sample() function: Error in input"
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), \
                "add_sample() function: Error in input"
        
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        
        self.n_samples += 1
        
        if self.x_vect is None:
            
            self.x_vect = x
            self.y_vect = y

            self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_sq_process
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(1))
            
            if self.keep_info_gain_estimate:
                self.info_gain = (0.5 * np.log(1 + 1 / self.sigma_sq_process))
        
        else:

            self.x_vect = np.vstack((self.x_vect, x))
            self.y_vect = np.vstack((self.y_vect, y))
            
            K_star = kernel_rbf(self.x_vect[:-1, :].reshape(-1, self.input_dim), 
                                self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            
            elem = kernel_rbf(self.x_vect[-1, :].reshape(1, self.input_dim), 
                              self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            
            if self.keep_info_gain_estimate:
                sigma_i = elem - K_star.T @ self.K_matrix_inv @ K_star
                self.info_gain += (0.5 * np.log(1 + sigma_i / self.sigma_sq_process))
            
            elem = np.array(elem + self.sigma_sq_process).reshape(1, 1)
            self.K_matrix = np.vstack((np.hstack((self.K_matrix, K_star)), np.hstack((K_star.T, elem))))
            self.K_matrix_inv = incr_inv(self.K_matrix_inv, K_star, K_star.T, elem)
            

    def compute(self, x):
        """
        Computes the posterior mean and variance at given test points.

        Args:
            x (numpy.ndarray): A 2D array of test points to predict at,
                               with shape (n_test_points, input_dim).

        Returns:
            tuple: A tuple `(mu, sigma)`:
                - **mu (numpy.ndarray)**: 1D array of posterior mean values
                  $f(x)$. Shape (n_test_points,).
                - **sigma (numpy.ndarray)**: 1D array of posterior variance
                  values $Var[f(x)]$. Shape (n_test_points,).

        Note:
            The returned `sigma` is the variance of the *noiseless function* $f(x)$.
            To get the predictive variance for a new *observation* $y^*$,
            you must add the noise variance: `sigma + self.sigma_sq_process`.
            
        Raises:
            ValueError: If input $x$ is not a 2D array or has the wrong
                        number of features (input_dim).
        """
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute() function: Error in input"
        
        n = x.shape[0]
        
        mu = np.zeros(n)
        sigma = np.zeros(n)
        
        for i in range(n):
            
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigma[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), 
                                  self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        
        return mu, sigma
    

    def get_info_gain(self):
        """
        Returns the currently computed information gain.

        Returns:
            float: The total information gain $I(y; f)$.

        Raises:
            ValueError: If `keep_info_gain_estimate` was set to `False`
                        during initialization.
        """
        if self.keep_info_gain_estimate:
            return self.info_gain[0, 0] if isinstance(self.info_gain, np.ndarray) else self.info_gain
        else:
            raise ValueError("Info Gain not computed, use flag "
                             "keep_info_gain_estimate=True during initialization")
        

    def reset(self):
        """
        Reset the regressor.
        """
        self.n_samples = 0
        self.info_gain = None
        self.x_vect = None
        self.y_vect = None
    
    

class HeteroscedasticGaussianProcessRegressorRBF: 
    """
    Implements a Heteroscedastic Gaussian Process Regressor with an RBF kernel.

    "Heteroscedastic" means that this model assumes the noise variance
    is *not* constant for all data points. Each observation $y_i$ has its
    own noise variance.

    **Performance Warning:**
    In `one_sample_mod=True`, the `add_sample()` method is not
    incrementally optimized like in the homoscedastic case, 
    as it is not possible.
    """


    def __init__(self, kernel_L, sigma_sq_process, input_dim=1, one_sample_mod=False):
        """
        Initializes the Heteroscedastic Gaussian Process Regressor with an RBF kernel.

        Args:
            kernel_L (float): The length-scale parameter (L) for the RBF kernel.
            sigma_sq_process (float): The base observation noise variance.
            input_dim (int, optional): The dimensionality of the input space.
                                       Defaults to 1.
            one_sample_mod (bool, optional): Selects the operating mode.
                                       Defaults to False.
        """
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.one_sample_mod = one_sample_mod
        self.reset()


    def load_data(self, x, y, sigmasqs):
        """
        Loads a batch of data with specified noise variances for each point.

        This method is *only* available if `one_sample_mod=False`.

        Args:
            x (numpy.ndarray): 2D array of input data, shape (n, input_dim).
            y (numpy.ndarray): 1D or 2D array of output data, shape (n,) or (n, 1).
            sigmasqs (numpy.ndarray): 1D array of noise variances for each data
                                      point, shape (n,).

        Raises:
            ValueError: If called when `one_sample_mod=True`.
        """
        if self.one_sample_mod:
            raise ValueError("load_data() cannot be used with one_sample_mod=True")
        
        n = x.shape[0]
        
        self.x_vect = np.array([x]).reshape(n, self.input_dim)
        self.y_vect = np.array([y]).reshape(n, 1)
        self.sigmasqs = np.array([sigmasqs]).reshape(n,)
        
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + np.diag(self.sigmasqs)
        self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(n))

    
    def add_sample(self, x, y, sample_weight=1):
        """
        Adds a single sample.

        This method is *only* available if `one_sample_mod=True`.

        Args:
            x (float, int, or numpy.ndarray): The new input point.
                Shape (1, input_dim).
            y (float, int, or numpy.ndarray): The new output value.
                Shape (1,) or (1, 1).
            sample_weight (float or int, optional): The weight of this sample.
                Used to update the average $y$ and the sample count $N_i$.
                Defaults to 1.

        Raises:
            ValueError: If called when `one_sample_mod=False` or if
                        input shapes are incorrect.
        """
        if not self.one_sample_mod:
            raise ValueError("add_sample() cannot be used with one_sample_mod=False")
        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim), "add_sample() function: Error in input"
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), \
                "add_sample() function: Error in input"
        
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        
        if self.x_vect is None:

            self.x_vect = x
            self.y_vect = y

            self.num_samples = [sample_weight]
            
            self.K_matrix_noreg = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L)
            self.K_matrix = self.K_matrix_noreg + (self.sigma_sq_process / sample_weight)
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(1))
        
        else:

            matches = np.where(np.all(self.x_vect == x, axis=1))[0]

            if len(matches) > 0:

                pos_first_found = matches[0]
                self.y_vect[pos_first_found] = (
                    self.y_vect[pos_first_found] * self.num_samples[pos_first_found] + y * sample_weight
                    ) / (self.num_samples[pos_first_found] + sample_weight)
                
                self.num_samples[pos_first_found] = self.num_samples[pos_first_found] + sample_weight
                
                self.K_matrix = self.K_matrix_noreg + np.diag(self.sigma_sq_process / np.array(self.num_samples))
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))
            
            else: 

                self.x_vect = np.vstack((self.x_vect, x))
                self.y_vect = np.vstack((self.y_vect, y))
                self.num_samples.append(sample_weight)
                
                self.K_matrix_noreg = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L)
                self.K_matrix = self.K_matrix_noreg + np.diag(self.sigma_sq_process / np.array(self.num_samples))
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))


    def compute(self, x):
        """
        Computes the posterior mean and variance at given test points.

        Args:
            x (numpy.ndarray): A 2D array of test points to predict at,
                               with shape (n_test_points, input_dim).

        Returns:
            tuple: A tuple `(mu, sigmasq)`:
                - **mu (numpy.ndarray)**: 1D array of posterior mean values
                  $f(x)$. Shape (n_test_points,).
                - **sigmasq (numpy.ndarray)**: 1D array of posterior variance
                  values $Var[f(x)]$. Shape (n_test_points,).

        Note:
            The returned `sigmasq` is the variance of the *noiseless function*
            $f(x)$. To get the predictive variance for a new *observation* $y^*$,
            you must add a noise term (e.g., `self.sigma_sq_process` if
            assuming a single new sample).
            
        Raises:
            ValueError: If input $x$ is not a 2D array or has the wrong
                        number of features (input_dim).
        """
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        
        n = x.shape[0]
        mu = np.zeros(n)
        sigmasq = np.zeros(n)
        
        for i in range(n):
            
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigmasq[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), 
                                    self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star

        return mu, sigmasq


    def get_info_gain(self):
        """
        Computes the information gain $I(y; f)$ for the heteroscedastic model.

        Returns:
            float: The computed information gain.
        """
        if self.one_sample_mod:
            temp = self.sigma_sq_process / np.array(self.num_samples)
        else:
            temp = self.sigmasqs
        
        D = np.diag(temp ** -0.5)
        _, value = np.linalg.slogdet(D @ (self.K_matrix - np.diag(temp)) @ D + np.eye(D.shape[0]))
        return 0.5 * value
    

    def reset(self):
        """
        Reset the regressor.
        """
        self.x_vect = None
        self.y_vect = None
        
        if self.one_sample_mod:
            self.num_samples = []
