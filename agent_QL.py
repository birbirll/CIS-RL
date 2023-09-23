import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class AgentQL(object):
    '''
    A Q-learning agent that learn a Q-table

    The stage, state, and action should all be discrete and finite
    '''
    def __init__(self, gamma, epsilon, lr, 
                 n_states, n_stages, n_actions,
                 eps_min=0.005, eps_dec=5e-4,
                 lr_min=0.001, lr_dec=5e-4):
        '''
        Parameters:
            gamma: discount factor
            epsilon: exploration rate
            lr: learning rate

            n_states: a finite number of states
            n_stages: a finite number of stages
            n_actions: a finite number of actions

            eps_min: minimum epsilon
            eps_dec: epsilon decay
        '''
        # Initialize the parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr_min = lr_min
        self.lr_dec = lr_dec
        self.action_space = [i for i in range(n_actions)]

        self.n_stages = n_stages
        self.n_states = n_states

        # Initialize the Q table
        self.Q_table = np.zeros((n_stages, n_states, n_actions))

        # Initialize a table to count the number of times each state is visited
        self.visit_count = np.zeros((n_stages, n_states))


    def choose_action(self, stage, state):
        '''
        Choose action based on epsilon greedy policy
        '''
        if np.random.random() > self.epsilon:
            # Exploit
            actions = self.Q_table[stage, state, :]
            action = np.argmax(actions)
        else:
            # Explore
            action = np.random.choice(self.action_space)

        return action

    

    def learn(self, stage, state, action, reward, new_state, terminal):
        '''
        Learn from the experience
        '''

        # Extract the current Q value from the Q-table
        old_value = self.Q_table[stage, state, action]

        # Update the visit count
        self.visit_count[stage, state] += 1

        if terminal:
            # In the terminal state, q value should be 0 since there is no future reward
            next_best = 0
        else:
            # Get the best action in the next state
            next_best = np.max(self.Q_table[stage + 1, new_state, :])

        # Compute the new Q value by applying the Bellman equation
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_best)
        # new_value = old_value + self.lr * (reward + self.gamma * next_best - old_value)

        # Update the Q table
        self.Q_table[stage, state, action] = new_value

        # Decay the epsilon
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        # Decay the learning rate
        self.lr = max(self.lr_min, self.lr - self.lr_dec)
        # self.lr = max(self.lr_min, self.lr * 0.99999)


    def get_policy(self):
        '''
        Get the policy from the trained agent
        '''
        policy = np.zeros((self.n_states, self.n_stages))
        for stage in range(self.n_stages):
            for state in range(self.n_states):
                policy[state][stage] = np.argmax(self.Q_table[stage, state, :])
        return policy
    
    

    def plot_state_visited_count(self, figsize, title):
        '''
        Plot the number of times each state is visited as a heatmap
        '''
        plt.figure(figsize=figsize)
        # Convert table into log scale
        # count = np.log(self.visit_count + 1)
        count = self.visit_count
        # Transpose the table
        count = count.T

        # make the heatmap wider
        im = plt.imshow(np.repeat(count, 10, axis=1), cmap='Oranges', interpolation='nearest', 
                        norm=colors.LogNorm())

        # Reverse the y-axis
        plt.gca().invert_yaxis()


        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.colorbar(im, label='Number of times visited', shrink=0.8)
        plt.title(title)
        
        plt.show()

