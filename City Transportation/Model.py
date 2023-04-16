import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(
            self, 
            lr: float,
            input_dims: int, 
            hidden_layer_1_dims: int, 
            hidden_layer_2_dims: int,
            n_actions: int
        ):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_1_dims = hidden_layer_1_dims
        self.hidden_layer_2_dims = hidden_layer_2_dims
        self.n_actions = n_actions
        self.hidden_layer_1 = nn.Linear(self.input_dims, self.hidden_layer_1_dims)
        self.hidden_layer_2 = nn.Linear(self.hidden_layer_1_dims, self.hidden_layer_2_dims)
        self.output_layer = nn.Linear(self.hidden_layer_2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.hidden_layer_1(state.to(self.hidden_layer_1.weight.dtype)))
        x = F.relu(self.hidden_layer_2(x.to(self.hidden_layer_2.weight.dtype)))
        actions = self.output_layer(x.to(self.output_layer.weight.dtype))

        return actions


class BusDriver:
    def __init__(
            self, 
            gamma: float = 0.99, 
            epsilon: float = 1.0,
            lr: float = 0.001,
            input_dims: int = 110, 
            batch_size: int = 64, 
            n_actions: int = 10,
            max_mem_size: int = 100000, 
            eps_end: float = 0.05, 
            eps_dec: float = 1e-5
            ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(lr=lr, n_actions=n_actions, input_dims=input_dims, hidden_layer_1_dims=256, hidden_layer_2_dims=256)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims),  dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = 1 / np.exp(self.iter_cntr * self.eps_dec) if self.epsilon > self.eps_min else self.eps_min

    def save_model(self, path: str):
        T.save(self.Q_eval.state_dict(), path)