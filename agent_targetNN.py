import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
                 n_actions):
        '''
        lr: learning rate
        input_dims: input dimensions
        fc1_dims: first fully connected layer dimensions
        fc2_dims: second fully connected layer dimensions
        n_actions: number of actions
        '''
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        '''
        returns the Q values for all actions
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class AgentTargetNN(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_min=0.005, eps_dec=5e-4, tar_update=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0 # memory counter
        self.iter_cntr = 0 # iteration counter
        self.tar_update = tar_update # target network update frequency
        
        # set up Q networks
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=128, fc2_dims=128)

        # Initialize the target network with the same architecture and weights
        self.Q_next = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=128, fc2_dims=128)
        
        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()  # Set the target network to evaluation mode

        # Current state memory
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        # Current action memory
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        # Memory of state after action
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        # Memory of reward
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # Memory of terminal state, true means terminal state
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store_transition(self, state, action, reward, state_, terminal):
        '''
        Store the transition in memory
        '''
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        '''
        Choose action based on epsilon greedy policy
        '''
        if np.random.random() > self.epsilon:
            # Exploit
            state = T.tensor(np.array([observation]), dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            # Explore
            action = np.random.choice(self.action_space)

        return action

    def choose_best_action(self, observation):
        '''
        Choose the best action
        '''
        state = T.tensor([observation], dtype=T.float32).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()

        return action
    
    def learn(self):
        '''
        Learn from the experience
        '''
        # Learn when memory is full
        if self.mem_cntr < self.batch_size:
            return
        
        # Reset the gradients
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size) # in case of memory is not full yet

        # Generate random indices
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Get the random batch of states, actions, rewards, new states, terminal states
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.bool).to(self.Q_eval.device)

        # Get the Q values for the current state
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        # Get the Q values for the next state use the target network
        q_next = self.Q_next.forward(new_state_batch)

        # Q values of terminal state is 0
        q_next[terminal_batch] = 0.0

        # Get the target Q values by Bellman equation
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # Calculate the loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        # Backpropagate the loss
        loss.backward()
        self.Q_eval.optimizer.step()

        # Increment the iteration counter
        self.iter_cntr += 1
        
        # Decay the epsilon
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        # Update the target network
        if self.iter_cntr % self.tar_update == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())


    def save_model(self, path):
        '''
        Save the model
        '''
        T.save(self.Q_eval.state_dict(), path)


    def load_model(self, path):
        '''
        Load the model
        '''
        self.Q_eval.load_state_dict(T.load(path))
