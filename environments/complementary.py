import numpy as np


class ComplementaryPricingEnvironment:
    """
    Simulates a retail environment with complementary product pricing.
    This environment models customer purchasing behavior where "leader" products
    (e.g., game consoles) can increase the demand for "follower" products
    (e.g., games). The relationships are defined by a graph.
    The environment is used to simulate sales based on a given pricing strategy
    (a set of margins) and to compute the expected total value (a mix of
    revenue and profit) for different pricing combinations.
    """


    def __init__(self, n_products, n_actions, margins, demands, n_baskets, alpha, graph_dict, mc_ep=1000, seed=0):
        """
        Initializes the pricing environment.

        Args:
            n_products (int): The total number of products.
            n_actions (int): The number of available pricing actions for each product.
            margins (np.ndarray): 2D array of shape (n_products, n_actions) of
                                  possible margin values.
            demands (np.ndarray): 3D array of shape (n_products, n_actions, 2).
                                  demands[i, j, 0] is the base demand for product i
                                  at margin j.
                                  demands[i, j, 1] is the enhanced demand if its
                                  leader is purchased.
            n_baskets (int): Default number of customer baskets for `step()`.
            alpha (float): Balances profit (0) vs. revenue (1). Must be in [0, 1].
            graph_dict (dict): Defines complementary relationships.
                               Keys: leader product indices (int).
                               Values: lists of follower indices (list[int]).
            mc_ep (int, optional): Monte Carlo episodes for `compute_values()`.
                                   Defaults to 1000.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        """
        
        if not demands.shape == (n_products, n_actions, 2):
            raise ValueError(f"Shape of the demand not coherent. Expected {(n_products, n_actions, 2)}, got {demands.shape}")
        if not margins.shape == (n_products, n_actions):
            raise ValueError(f"Shape of the margins not coherent. Expected {(n_products, n_actions)}, got {margins.shape}")
        if not ((demands <= 1).all() and (demands >= 0).all()):
            raise ValueError("Error in demand values: all demands must be probabilities in [0, 1]")
        if not (alpha >= 0 and alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not n_baskets >= 1:
            raise ValueError(f"n_baskets must be positive, got {n_baskets}")
        if not mc_ep >= 10:
            raise ValueError(f"mc_ep too low for reliable estimates (e.g., >= 10), got {mc_ep}")
        
        self.n_products = n_products
        self.n_actions = n_actions
        self.n_baskets = n_baskets
        self.demands = demands
        self.margins = margins
        self.alpha = alpha # 1 revenue, 0 profit
        self.graph_dict = graph_dict # elements for 0 to num_products-1
        self.mc_ep = mc_ep
        self.margins_to_idx_lst = []
        
        for prod in range(n_products):
            self.margins_to_idx_lst.append({self.margins[prod, idx]: idx for idx in range(0, self.n_actions)})

        self.leaders_lst = list(self.graph_dict.keys())
        self.followers_lst = list(self.graph_dict.values())
        aux = self.followers_lst.copy()
        self.followers_lst = np.array([x for sublist in self.followers_lst for x in sublist])
        
        aux.append(self.leaders_lst)
        aux = np.array([x for sublist in aux for x in sublist])
        if not (np.issubdtype(aux.dtype, np.integer) and
                np.all(aux >= 0) and
                np.all(aux <= self.n_products - 1) and
                len(aux) == len(list(set(aux)))):
            raise ValueError("Error in graph_dict: All product indices must be unique integers between 0 and n_products - 1")
        
        self.follower_to_leader_dict = {value: key for key, values in self.graph_dict.items() for value in values}
        
        for leader in self.leaders_lst:
            followers = self.graph_dict[leader]
            if not isinstance(followers, list):
                raise TypeError(f"All graph_dict values must be of type list, but leader {leader} has type {type(followers)}")
            if not len(followers) <= 2:
                raise NotImplementedError(f"Leaders with >2 followers not implemented yet in this env (leader {leader} has {len(followers)})")

        self.compute_values()

        self.reset(seed)
    

    def step(self, margins, override_n_baskets=None):
        """
        Simulates sales for a given set of margins.

        Generates sales results for a number of customer baskets based on the
        chosen margins for all products. Follower product demand is enhanced
        if its corresponding leader is sold in the same basket.

        Args:
            margins (np.ndarray): 1D array of shape (n_products,).
                                  Specifies the *margin value* (not index)
                                  for each product.
            override_n_baskets (int, optional): If provided, simulates this
                                                number of baskets instead of
                                                `self.n_baskets`.

        Returns:
            np.ndarray: 2D int array of shape (n_products, n_bsk).
                        `sales_mx[i, j] = 1` if product `i` was sold in
                        basket `j`, and 0 otherwise.
        """
        
        if not margins.ndim == 1:
            raise ValueError(f"The action (margins) must be 1-dimensional, but got {margins.ndim} dimensions")
        if not margins.shape[0] == self.n_products:
            raise ValueError(f"The action (margins) must be of dimension n_products ({self.n_products}), but got {margins.shape[0]}")
        
        if override_n_baskets is not None:
            n_bsk = override_n_baskets
        else:
            n_bsk = self.n_baskets
        
        sales_mx = np.ones((self.n_products, n_bsk), dtype=int)
        
        for leader in self.leaders_lst:

            demand = self.demands[leader, self.margins_to_idx_lst[leader][margins[leader]], 0] 
            sales_mx[leader, :] = np.random.uniform(0, 1, (n_bsk)) < demand
        
        for follower in self.followers_lst:
            
            corr_leader = self.follower_to_leader_dict[follower]
            
            demand = self.demands[follower, self.margins_to_idx_lst[follower][margins[follower]], 0] 
            enhancement_demand = self.demands[follower, self.margins_to_idx_lst[follower][margins[follower]], 1]
            
            mask_leader_sales = sales_mx[corr_leader, :] == 1
            
            sales_demand = np.random.uniform(0, 1, (n_bsk - np.sum(mask_leader_sales))) < demand
            sales_enhancement_demand = np.random.uniform(0, 1, (np.sum(mask_leader_sales))) < enhancement_demand

            sales_mx[follower, ~mask_leader_sales] = sales_demand
            sales_mx[follower, mask_leader_sales] = sales_enhancement_demand

        return sales_mx


    def compute_values(self):
        """
        Pre-computes expected values for all margin combinations.

        Iterates through each leader and its followers, simulating all possible
        margin combinations for that subgraph using Monte Carlo.
        The total number of samples for each combination is
        `self.mc_ep * self.n_baskets`.

        This method populates the `self.action_values` dictionary.
        The keys are leader indices, and the values are N-dimensional arrays
        (where N = 1 + num_followers) containing the expected objective value
        for each margin combination.
        """
        
        self.action_values = {}
        n_samples = int(self.mc_ep * self.n_baskets)
        
        for leader in self.leaders_lst:
            
            followers = self.graph_dict[leader]
            
            if len(followers) == 0:
                vals = -1 * np.ones((self.n_actions, ))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    margins_action = self.margins[:, 0].copy()
                    margins_action[leader] = leader_margin
                    sales_mx = self.step(margins_action, override_n_baskets=n_samples)
                    empirical_demand = np.sum(sales_mx[leader, :]) / n_samples
                    vals[leader_margin_i] = (self.alpha + leader_margin) * empirical_demand

            if len(followers) == 1:
                vals = -1 * np.ones((self.n_actions, self.n_actions))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    for follower_margin_i, follower_margin in enumerate(self.margins[followers[0], :]):
                        margins_action = self.margins[:, 0].copy() 
                        margins_action[leader] = leader_margin
                        for follower in followers:
                            margins_action[follower] = follower_margin
                        sales_mx = self.step(margins_action, override_n_baskets=n_samples)
                        mc_sales = np.sum(sales_mx, axis=1) / n_samples
                        obj_fun = (self.alpha + leader_margin) * mc_sales[leader]
                        for follower in followers:
                            obj_fun += (self.alpha + follower_margin) * mc_sales[follower]
                        vals[leader_margin_i, follower_margin_i] = obj_fun
            
            if len(followers) == 2:
                vals = -1 * np.ones((self.n_actions, self.n_actions, self.n_actions))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    for follower0_margin_i, follower0_margin in enumerate(self.margins[followers[0], :]):
                        for follower1_margin_i, follower1_margin in enumerate(self.margins[followers[1], :]):
                            margins_action = self.margins[:, 0].copy() 
                            margins_action[leader] = leader_margin
                            margins_action[followers[0]] = follower0_margin
                            margins_action[followers[1]] = follower1_margin
                            sales_mx = self.step(margins_action, override_n_baskets=n_samples)
                            mc_sales = np.sum(sales_mx, axis=1) / n_samples
                            obj_fun = (self.alpha + leader_margin) * mc_sales[leader]
                            obj_fun += (self.alpha + follower0_margin) * mc_sales[followers[0]]
                            obj_fun += (self.alpha + follower1_margin) * mc_sales[followers[1]]  
                            vals[leader_margin_i, follower0_margin_i, follower1_margin_i] = obj_fun
            
            self.action_values[leader] = vals
    
    
    def compute_givenaction_value(self, margins):
        """
        Calculates the total expected value for a specific action.

        Uses the pre-computed `self.action_values` to look up the expected
        value for the given margin combination for each leader-follower
        subgraph and sums them up.

        Args:
            margins (np.ndarray): 1D array of shape (n_products,).
                                  Specifies the *margin value* for each product.

        Returns:
            float: The total expected value for the given set of margins.
        """
        
        if not margins.ndim == 1:
            raise ValueError(f"The action (margins) must be 1-dimensional, but got {margins.ndim} dimensions")
        if not margins.shape[0] == self.n_products:
            raise ValueError(f"The action (margins) must be of dimension n_products ({self.n_products}), but got {margins.shape[0]}")
        
        value = 0
        
        for leader in self.leaders_lst:
            
            followers = self.graph_dict[leader]
            if len(followers) == 0:
                value += self.action_values[leader][
                    self.margins_to_idx_lst[leader][margins[leader]]
                    ]     
            if len(followers) == 1:
                value += self.action_values[leader][
                    self.margins_to_idx_lst[leader][margins[leader]], 
                    self.margins_to_idx_lst[followers[0]][margins[followers[0]]]
                    ]
            if len(followers) == 2:
                value += self.action_values[leader][
                    self.margins_to_idx_lst[leader][margins[leader]], 
                    self.margins_to_idx_lst[followers[0]][margins[followers[0]]], 
                    self.margins_to_idx_lst[followers[1]][margins[followers[1]]]
                    ]
        
        return value


    def compute_best_action_value(self):
        """
        Calculates the maximum possible expected value achievable.

        Finds the maximum value within each leader's `action_values` matrix
        (representing the optimal policy for that subgraph) and sums these
        maximums.

        Returns:
            float: The theoretical maximum expected value.
        """

        value = 0

        for leader in self.leaders_lst:
            value += np.max(self.action_values[leader])

        return value


    def reset(self, seed=0):
        """
        Resets the environment's random number generator.

        Args:
            seed (int, optional): The seed for np.random.seed(). Defaults to 0.
        """
        
        np.random.seed(seed)
