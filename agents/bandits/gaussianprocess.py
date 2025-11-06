import numpy as np
from regressors.gaussianprocess import (
    HeteroscedasticGaussianProcessRegressorRBF, 
    GaussianProcessRegressorRBF
)

class IGPUCB:
    """
    Implements the Improved Gaussian Process Upper Confidence Bound (IGP-UCB)
    algorithm for multi-armed bandits.
    This agent models the unknown reward function using a Gaussian Process (GP).
    It selects actions by maximizing an Upper Confidence Bound (UCB), which
    balances exploration (trying actions with high uncertainty) and
    exploitation (picking actions with high estimated mean rewards).

    The class supports two modes via the `het` flag:
    1.  `het=True` (Heteroscedastic): Uses a GP that assumes noise can
        vary per point. This mode is implemented to handle sample averaging,
        where multiple pulls of the *same* action decrease its effective
        noise variance.
    2.  `het=False` (Homoscedastic): Uses a standard GP with constant
        noise for all observations.
    """
    
    
    def __init__(self, n_actions, action_dim, actions, kernel_L, sigma_sq, B, delta, het=True):
        """
        Initializes the IGP-UCB agent.

        Args:
            n_actions (int): The total number of available actions.
            action_dim (int): The dimension of each action vector.
            actions (numpy.ndarray): A 2D array of shape (n_actions, action_dim)
                                     listing all discrete actions.
            kernel_L (float): The length-scale parameter for the RBF kernel.
            sigma_sq (float): The base noise variance of observations.
            B (float): A scaling parameter for the UCB bonus term.
            delta (float): The confidence parameter.
            het (bool, optional): If True, uses the heteroscedastic GP
                for sample averaging. If False, uses the standard
                (homoscedastic) GP. Defaults to True.
        
        Raises:
            ValueError: If action dimensions are incorrect or kernel_L is not
                        positive.
        """
        
        if not (actions.ndim == 2 and actions.shape == (n_actions, action_dim)):
            raise ValueError(f"Error in action dimension. Expected ({n_actions}, {action_dim}), "
                             f"got {actions.shape}")
        if not kernel_L > 0:
            raise ValueError("kernel_L must be positive")
        
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.actions = actions
        self.kernel_L = kernel_L
        self.sigma_sq = sigma_sq
        self.B = B
        self.delta = delta
        self.het = het
        
        self.reset()

        
    def pull(self):
        """
        Selects the next action to play based on the UCB maximization.

        If no samples have been observed, a random action is chosen.
        Otherwise, it computes $UCB$ for all actions and selects the 
        action with the highest UCB value.

        Returns:
            numpy.ndarray: The selected action vector of shape (action_dim,).
        """
        
        if self.no_samples:
            # On the first pull, choose randomly
            choice = np.random.choice(self.n_actions)
            self.last_action = self.actions[choice, :]
        
        else:
            
            self.last_action = self.actions[np.argmax(self.get_optimistic_estimates()), :]
        
        return self.last_action
    
    
    def get_optimistic_estimates(self, return_raw_info=False):
        """
        Computes the optimistic (UCB) estimates for all actions.

        This is the core calculation used by `pull()` but exposed for
        analysis or plotting.

        Args:
            return_raw_info (bool, optional): If True, returns the UCBs,
                means, and variances. If False, returns only the UCBs.
                Defaults to False.

        Returns:
            numpy.ndarray or tuple:
            - If `return_raw_info=False`: 1D array (n_actions,) of UCB values.
            - If `return_raw_info=True`: Tuple (ucbs, mu, sigmasq), where each
              is a 1D array of shape (n_actions,).
        """
        
        if self.no_samples:
            # Return uniform estimates if no data
            mu = np.zeros((self.n_actions))
            sigmasq = np.ones((self.n_actions)) # Initial variance is 1 (k(x,x))
            ucbs = np.ones((self.n_actions))
        
        else:
            # Compute posterior mean and variance
            mu, sigmasq = self.regressor.compute(self.actions)
            
            # Get the dynamic exploration bonus
            beta = self._get_beta()
            
            # Compute UCB
            ucbs = mu + beta * np.sqrt(sigmasq)
            
        if return_raw_info:
            return ucbs.ravel(), mu.ravel(), sigmasq.ravel()
        else:
            return ucbs.ravel()
        
        
    def update(self, reward, sample_weight=1):
        """
        Updates the GP model with a reward for the last action selected by pull().

        Args:
            reward (float): The observed reward.
            sample_weight (float or int, optional): The weight of this sample.
                Only used if `het=True`. Defaults to 1.
        
        Raises:
            ValueError: If `pull()` has not been called before `update()`.
            ValueError: If `sample_weight` is 0.
            NotImplementedError: If `sample_weight` != 1 is used with
                                 `het=False` (standard GP).
        """
        
        if self.last_action is None:
            raise ValueError("No action has been selected yet. Call pull() before update().")
        if sample_weight != 1 and not self.het:
            raise NotImplementedError("Batch/weighted updates are not implemented "
                                      "for the standard (non-heteroscedastic) GP.")
        if sample_weight == 0:
            raise ValueError("sample_weight cannot be 0")
        
        # Add the sample to the regressor
        self.regressor.add_sample(self.last_action.reshape(1, self.action_dim), 
                                  np.array(reward).reshape(1, 1), 
                                  sample_weight=sample_weight)
        
        self.no_samples = False
        
       
    def update_complete(self, action, reward, sample_weight=1):
        """
        Updates the GP model with an explicitly provided action-reward pair.

        This is useful for offline learning or if the action selection
        is handled externally.

        Args:
            action (numpy.ndarray): The action that was taken, shape (action_dim,).
            reward (float): The observed reward.
            sample_weight (float or int, optional): The weight of this sample.
                Only used if `het=True`. Defaults to 1.

        Raises:
            NotImplementedError: If `sample_weight` != 1 is used with
                                 `het=False` (standard GP).
        """
        
        if sample_weight != 1 and not self.het:
            raise NotImplementedError("Batch/weighted updates are not implemented "
                                      "for the standard (non-heteroscedastic) GP.")

        self.regressor.add_sample(np.array(action).reshape(1, self.action_dim), 
                                  np.array(reward).reshape(1, 1), 
                                  sample_weight=sample_weight)
        
        self.no_samples = False


    def _get_beta(self):
        """
        Calculates the coefficient beta of the exploration bonus.
        
        Returns:
            float: The exploration bonus.
        """

        return self.B + np.sqrt(self.sigma_sq) * np.sqrt(2 * (
            self.regressor.get_info_gain() + 1 + np.log(1/self.delta)))
        

    def reset(self):
        """
        Resets the agent to its initial state.

        Clears all observed data by re-initializing the used GP regressor.
        """

        self.no_samples = True
        self.last_action = None
        
        if self.het:
            self.regressor = HeteroscedasticGaussianProcessRegressorRBF(
                self.kernel_L, self.sigma_sq, 
                input_dim=self.action_dim, one_sample_mod=True)
        else:
            self.regressor = GaussianProcessRegressorRBF(
                self.kernel_L, self.sigma_sq, 
                input_dim=self.action_dim, keep_info_gain_estimate=True)
