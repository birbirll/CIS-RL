import numpy as np
from scipy.stats import poisson


class BISolverSto(object):

    def __init__(self, env):
        '''
        Initialize the agent
        '''
        self.env = env
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.g_T = env.g_T
        self.g_t = env.g_t
        self.d_t = env.d_t
        self.T = env.T
        self.f = env.f

        self.waiting_capacity = env.waiting_capacity
        self.max_on_demand_doc = env.max_on_demand_doc
        self.open_time = env.open_time
        self.arrival_distrib = env.arrival_distrib

        # Precompute the poisson distribution to save time, saved in a dictionary
        self.poission_dist = {}
        for lmbda in set(self.arrival_distrib):
            self.poission_dist[lmbda] = poisson.pmf(np.arange(len(self.state_space)), lmbda)
    
    
    def p_trans(self, x_t, u_t, t):
        '''
        Return a list of probabilities of the next state 
        given the current state and action at time t
        Assume that the number of new patients is a poisson distribution
        '''
        # Get the expected number of new patients
        lmbda = self.d_t(t)
        # Probability of transition of (x_t, u_t) to each next state
        trans_prob = np.zeros(len(self.state_space))

        # Iterate through all possible number of new patients
        for d in range(self.waiting_capacity + 1):
            # Compute the next state given the number of new patients
            next_state = self.f(x_t, u_t, t, d)
            # If this next state can be reached
            if next_state < len(self.state_space):
                # Compute the probability of transition
                # trans_prob[next_state] += poisson.pmf(d, lmbda) # Slow
                trans_prob[next_state] += self.poission_dist[lmbda][d]

        # Normalize the probability by adding all remaining probability to the last state
        trans_prob[-1] += 1 - np.sum(trans_prob)
        # trans_prob = trans_prob * (1 / np.sum(trans_prob))

        return trans_prob


    def backward_induction(self):
        '''
        Perform backward induction to find the optimal policy and cost.

        Return the optimal policy and cost
        '''
        # Matrix of optimal policy for all states and actions
        U_t = np.zeros((len(self.state_space), self.T))
        # Matrix of cost for all states and actions
        J_t = np.inf * np.ones((len(self.state_space), self.T + 1))

        # Initialize the final cost
        for x in self.state_space:
            J_t[x][self.T] = self.g_T(x)

        # Iterate backward
        for t in range(self.T - 1, -1, -1):
            for x in self.state_space:
                for u in self.action_space:

                    # Compute the cost of choosing action u at state x
                    cost = self.g_t(x, u) + self.p_trans(x, u, t).T @ J_t[:, t + 1]

                    # Update the policy and cost if the cost is lower
                    if cost < J_t[x][t]:
                        J_t[x][t] = cost
                        U_t[x][t] = u

        return U_t, J_t