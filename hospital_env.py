import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class HospitalEnv(object):

    def __init__(self,
                 arrival_distrib,
                 open_time=8,
                 work_hours=12,
                 patient_waiting_cost=30,
                 doc_cost=500,
                 waiting_capacity=100,
                 max_on_demand_doc=10,
                 patient_not_treated_cost=300,
                 default_queue_length=15,
                 deterministic=False):
        
        # Initialize the parameters
        self.open_time = open_time
        self.T = work_hours
        self.patient_waiting_cost = patient_waiting_cost
        self.doc_cost = doc_cost
        self.waiting_capacity = waiting_capacity
        self.max_on_demand_doc = max_on_demand_doc
        self.patient_not_treated_cost = patient_not_treated_cost
        self.default_queue_length = default_queue_length
        self.deterministic = deterministic

        # Clip the arrival distribution to be 0 to maximum capacity
        self.arrival_distrib = np.clip(arrival_distrib, 0, waiting_capacity)

        # Define the state space and action space
        self.state_space = np.arange(waiting_capacity + 1)
        self.action_space = np.arange(max_on_demand_doc + 1)
        

    def d_t(self, t):
        '''
        Return the number of new patients arriving at the hospital at time t.
        Should be less than the waiting capacity.
        '''
        return self.arrival_distrib[(t + self.open_time) % 24]

    
    def f(self, x_t, u_t, t, d=-1):
        '''
        The transition function,
        return x_{t+1}, the queue length at time t + 1.
        '''
        # If d is not specified, use the actual number from the given distribution
        d = self.d_t(t) if d == -1 else d 
        # return np.max([x_t + d - 2 * (u_t + 10), 0])
        return int(np.clip(x_t + d - 2 * (u_t + 10), 0, self.waiting_capacity))


    def g_t(self, x_t, u_t):
        '''
        Cost function for action u_t at state x_t
        '''
        return self.patient_waiting_cost * x_t + self.doc_cost * u_t
    

    def g_T(self, x_T):
        '''
        Cost function for the end state
        '''
        return self.patient_not_treated_cost * x_T
    

    def reset(self):
        '''
        Reset the environment to the initial state
        '''
        self.current_state = self.default_queue_length
        self.current_time = 0
        return self.current_state, self.current_time

    
    def step(self, action, render=False):
        '''
        Take an action and return the next state and cost

        Return:
            next_state: the next state
            cost: the cost of taking the action
            finished: whether the episode is finished
        '''
        
        if self.deterministic:
            # Draw from the given distribution directly
            new_patient_count = self.d_t(self.current_time)
        else:
            # Use the given distribution as mean to draw a sample from the poisson distribution
            d_mean = self.d_t(self.current_time)
            new_patient_count = np.random.poisson(d_mean)

        if render:
            print(f"  * {new_patient_count} new patients arrived")
            print(f"  * Using {int(action)} on-demand doctors")
        next_state = self.f(self.current_state, action, self.current_time, new_patient_count)
        
        # Update the time and state
        self.current_time += 1
        self.current_state = next_state

        done = self.current_time == self.T
        
        cost = self.g_t(self.current_state, action)
        if done:
            cost += self.g_T(self.current_state)

        return (next_state, self.current_time), cost, done
    

    def render(self):
        '''
        Render the environment
        '''
        print(f"\n---- t = {self.current_time} ----")
        print(f"Queue Length: {self.current_state}")
        print(f"[] {'! ' * self.current_state}\n")


    def visualize_policy(self, U_t, title='fig', figsize=(8, 8)):
        '''
        Visualize the given policy
        '''
        _, ax = plt.subplots(nrows=2, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        plt.suptitle(title)

        # Visualize the optimal policy in a heatmap
        heatmap = sns.heatmap(U_t, cmap=sns.diverging_palette(230, 20, as_cmap=True),
                    yticklabels=5, cbar_kws={'ticks': np.arange(self.max_on_demand_doc + 1)}, ax=ax[0])
        
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('# of on-demand doc')
        
        # Invert the y axis to have the first row at the top
        ax[0].invert_yaxis()
        # Hide the ticks
        ax[0].tick_params(axis='both', which='both', length=0)
        # Align the x-axis ticks with the time
        ax[0].set_xticklabels([(i + self.open_time) % 24 for i in range(self.T)])

        ax[0].set_xlabel("Hour")
        ax[0].set_ylabel("Queue Length")


        # Visualize the number of arrivals in a bar chart
        ax[1].bar(np.arange(24), self.arrival_distrib, color='grey')
        # Color the bar of the open hours
        ax[1].bar(np.arange(self.open_time, self.open_time + self.T), 
                  self.arrival_distrib[self.open_time:self.open_time + self.T], color='salmon')
        # Set a legend for the open/closed hours
        ax[1].legend(['Closed', 'Open'], loc='upper left')
        # Make the x-axis ticks time
        ax[1].set_xticks(np.arange(24))
        ax[1].set_xticklabels([f"{i}:00" for i in range(24)])
        # Make x tick labels 45 degrees
        plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("# of new patients arriving")

        # Plot the mean of arrivals
        ax[1].axhline(y=np.mean(self.arrival_distrib), color='darkgrey', linestyle='--', label='Mean')

        plt.tight_layout()
        # plt.savefig(f"figs/{title}.png", dpi=300)
        plt.show()