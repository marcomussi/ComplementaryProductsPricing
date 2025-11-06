import numpy as np
from agents.bandits.gaussianprocess import IGPUCB



class OptIndepPricingAgent:
    """
    An agent for solving an independent product pricing problem.
    This agent uses a Improved Gaussian Process Upper Confidence Bound (IGP-UCB)
    bandit to find the optimal margin (price) for a single product.
    The agent doesn't model the objective function directly. Instead, it
    models the demand probability as a function of the margin.
    It uses an `IGPUCB` agent for this, which assumes the demand observations
    are noisy (following a Bernoulli/Binomial distribution).
    """


    def __init__(self, actions, kernel_L, horizon, alpha):
        """
        Initializes the pricing agent.

        Args:
            actions (numpy.ndarray): A 2D array of shape (n_actions, 1)
                listing the discrete margin values (prices) available.
            kernel_L (float): The length-scale parameter for the RBF kernel
                of the demand-modeling GP.
            horizon (int): The total time horizon. Used to set the $\delta$
                confidence parameter for the IGPUCB agent (as $1/T$).
            alpha (float): The weighting parameter [0, 1] for the objective.
                - `alpha = 0` optimizes for pure Profit (Margin).
                - `alpha = 1` optimizes for pure Revenue (Price).

        Raises:
            ValueError: If `actions` is not a 2D array with one column,
                or if `alpha` is not in the [0, 1] range.
        """
        
        if not (isinstance(actions, np.ndarray) and actions.ndim == 2 and
                actions.shape[0] > 1 and actions.shape[1] == 1):
            raise ValueError("actions must be a 2D array with one column "
                             "and more than one action.")
        if not (alpha >= 0 and alpha <= 1):
            raise ValueError("alpha must be in [0, 1]")

        self.actions = actions
        self.action_dim = self.actions.shape[1]
        self.n_actions = self.actions.shape[0]
        self.alpha = alpha
        self.bandit_agent = IGPUCB(n_actions=self.n_actions,
                                   action_dim=1,
                                   actions=actions,
                                   kernel_L=kernel_L,
                                   sigma_sq=0.25,
                                   B=1,
                                   delta=1/horizon,
                                   het=True)
        self.last_action = None
    

    def pull(self, return_val=False):
        """
        Selects the best margin based on the optimistic objective.

        1.  Gets the UCBs for the demand from the GP.
        2.  Calculates the UCBs for the objective.
        3.  Picks the margin that maximizes the objective UCB.

        Args:
            return_val (bool): return value of optimistic estimate.
                               Defaults to False.

        Returns:
            float: The chosen margin value.
            float: The value of the optimistic estimate 
                   (only if return_val is set to True).
        """

        demand_ucbs = self.bandit_agent.get_optimistic_estimates()
        
        obj_ucbs = (self.alpha + self.actions.ravel()) * demand_ucbs
        
        choice_idx = np.argmax(obj_ucbs)
        self.last_action = self.actions[choice_idx, :]

        if return_val:
            return float(self.last_action), np.max(obj_ucbs)
        else:
            return float(self.last_action)

    
    def update(self, sales, impressions):
        """
        Updates the internal demand model with observed data.

        This agent is designed to be updated with aggregated data from a
        Bernoulli/Binomial process (e.g., from an environment step).

        Args:
            sales (int or float): The number of successful sales (conversions).
            impressions (int): The number of total trials (users/opportunities).
        """

        if impressions > 0:
            self.bandit_agent.update_complete(self.last_action,
                                              float(sales / impressions),
                                              sample_weight=int(impressions))
    

    def update_complete(self, action, sales, impressions):
        """
        Updates the internal demand model with observed data.

        This agent is designed to be updated with aggregated data from a
        Bernoulli/Binomial process.

        Args:
            action (float): Action associated with the sales.
            sales (int or float): The number of successful sales (conversions).
            impressions (int): The number of total trials (users/opportunities).
        """
        
        if impressions > 0:
            self.bandit_agent.update_complete(action,
                                              float(sales / impressions),
                                              sample_weight=int(impressions))

    def get_expected_obj_max(self):
        """
        Returns the maximum estimated objective (in expectation).

        Returns:
            float: The maximum estimated objective.
        """

        _, mu, _ = self.bandit_agent.get_optimistic_estimates(return_raw_info=True)
        
        obj = (self.alpha + self.actions.ravel()) * mu

        return np.max(obj)
