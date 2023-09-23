import numpy as np


class BISolverDet(object):

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

                    # Next state, make sure it is within the hospital capacity
                    x_ = self.f(x, u, t)
                    # Compute the cost
                    cost = self.g_t(x, u) + J_t[x_][t + 1]

                    # Update the policy and cost if the cost is lower
                    if cost < J_t[x][t]:
                        J_t[x][t] = cost
                        U_t[x][t] = u

        return U_t, J_t