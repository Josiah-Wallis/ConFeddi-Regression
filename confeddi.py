import os
import tensorflow as tf
import numpy as np
from math import log2
from random import sample
from timeit import default_timer
from typing import Union, Callable
from tensorflow import keras
from keras import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.optimizers import Adam
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONHASHSEED'] = str(50)

# Context Functions
def total_time(data):
    # returns client update time + communication delay

    return data[1] + data[-1]

def normalized_local_training_loss(data):
    # returns final local training loss divided by initial local training loss

    history = data[2].history
    return history['loss'][-1] / history['loss'][0]

def normalized_local_validation_loss(data):
    # returns final local validation loss divided by initial local validation loss

    history = data[2].history
    return history['val_loss'][-1] / history['val_loss'][0]

def prev_reward(data):
    # returns previous round reward

    return data[3]

def invalid():
    # produces Exception if GetContext accesses invalid context function

    raise Exception('Invalid')
    
""" Updated Context Elements

0: Client Update Time + Communication Delay
1: Normalized Local Training Loss
2: Normalized Local Validation Loss
3: Average Local Validation Change Rate (Reward)
- |v_{t}-v_{t-1}| / (round time)
- v_{t} = average of final local validation loss for round t

"""

# FL Class
class FederatedSystem:
    def __init__(self, clients_X: list[np.array], clients_y: list[np.array], distances: np.array, seed: int = 50) -> None:
        """
        Unpack client data, labels, and distances.
        Initialize containers for storing run history.
        """

        # Unpack data used for training and timing
        self.seed = seed
        self.log = []
        self.clients_X = clients_X
        self.clients_y = clients_y
        self.distances = distances
        self.comm_times = None

        # Define default local model
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()
        self.model = Sequential([
            Dense(32, activation = 'relu', input_shape = (10,)),
            Dense(16, activation = 'relu'),
            Dense(1)
        ])

        # Initialize FL containers
        self.trainable_layers = []
        self.w_history = []
        self.b_history = []

        # Record trainable layers
        for i, x in enumerate(self.model.layers):
            if x.weights:
                self.trainable_layers.append(i)

        # ConFeddi context definition
        self.ContextElements = [0, 1, 2, 3]
        self.ContextDef = {
            0: total_time,
            1: normalized_local_training_loss,
            2: normalized_local_validation_loss,
            3: prev_reward
        }

        # C2UCB Containers
        self.V_ucb = []
        self.b_ucb = []

        # Performance
        self.test_data = None

    def clear_history(self) -> None:
        """
        Refreshes FL system to be trained again.
        """

        self.log = []
        self.comm_times = []
        self.w_history = []
        self.b_history = []
        self.V_ucb = []
        self.b_ucb = []

    def DefaultModel(self) -> None:
        """
        If local models have been altered, reverts back to base models. 
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()
        self.model = Sequential([
            Dense(32, activation = 'relu', input_shape = (10,)),
            Dense(16, activation = 'relu'),
            Dense(1, activation = 'relu')
        ])

        # Record trainable layers
        for i, x in enumerate(self.model.layers):
            if x.weights:
                self.trainable_layers.append(i)

    # Setters + Getters
    def SetModel(self, model: tf.keras.Model) -> None:
        """
        Sets FL local models to model.
        """

        self.model = model

        # Record trainable layers
        for i, x in enumerate(self.model.layers):
            if x.weights:
                self.trainable_layers.append(i)

    def SetTestData(self, test_data: np.array) -> None:
        """
        FL system evaluates on test_data at test time.
        Should be set prior to using self.test_loss.
        """

        self.test_data = test_data
    
    def SetSeed(self, seed: int) -> None:
        """
        Sets seed for numpy, random, and Python.
        """

        self.seed = seed

    def SetContextElements(self, options: list[int]) -> None:
        """
        Sets which context elements ConFeddi will evaluate on.
        """

        self.ContextElements = options

    def GetLog(self) -> list[float]:
        """
        Returns timing history of round ends in seconds.
        """

        return self.log

    def AddContextElements(self, func: Callable) -> None:
        """
        Adds a new context criteria to be evaluated.
        Requires some tweaking with default context arguments.
        """

        keys = list(self.ContextDef.keys()).sort()
        new_key = keys[-1] + 1
        self.ContextDef[new_key] = func
        self.ContextElements += [new_key]

    # FedAvg
    def initialize_weights(self):
    # Returns initial model parameters of self.model
        w = []
        b = []

        for l in self.trainable_layers:
            w.append(self.model.layers[l].get_weights()[0])
            b.append(self.model.layers[l].get_weights()[1])

        return w, b

    def generate_model(self, w: list[np.array], b: list[np.array], skip: int = 0) -> tf.keras.Model:
        """
        Given model parameters and bias of same shape as self.model,
        initializes copy of self.model with said parameters.
        Must be tweaked for custom or more complex NN layers.
        """
        
        model = clone_model(self.model)
        if not skip:
            for i, x in enumerate(self.trainable_layers):
                if type(model.layers[x]) is keras.layers.BatchNormalization:
                    shape = model.layers[x].weights[2].shape
                    mn = np.zeros(shape)
                    vr = np.ones(shape)
                    model.layers[x].set_weights([w[i], b[i], mn, vr])
                else:
                    model.layers[x].set_weights([w[i], b[i]])

        return model

    def ClientUpdate(self, X, y, lr, w, b, E):
        # Trains a local model
        model = self.generate_model(w, b)
        model.compile(optimizer = Adam(learning_rate = lr), loss = 'mse')
        history = model.fit(X, y, validation_split = 0.2, epochs = E, verbose = 0, shuffle = False, use_multiprocessing = True)

        w = []
        b = []
        for l in self.trainable_layers:
            w.append(model.layers[l].get_weights()[0])
            b.append(model.layers[l].get_weights()[1])

        return w, b, model, history

    def aggregate(self, w_updates, b_updates, n_k, S_t):
        # FedAvg model aggregation step
        n = np.sum(n_k[S_t])
        num_trainable_layers = len(self.trainable_layers)
        w = [0 for _ in range(num_trainable_layers)]
        b = [0 for _ in range(num_trainable_layers)]

        for k in S_t:
            w_k = w_updates[k]
            b_k = b_updates[k]
            for l in range(num_trainable_layers):
                w[l] += (n_k[k] / n) * w_k[l]
                b[l] += (n_k[k] / n) * b_k[l]

        return w, b

    def initialize(self, system, reg_coeff = None, frac_clients = None):
        # Initializes FedAvg or ConFeddi model with appropriate parameters

        # Compute communication times from distances
        self.communication_times()

        # Initialize w, b from standard normal
        w, b = self.initialize_weights()
        self.w_history.append(w)
        self.b_history.append(b)

        # Number of Clients
        K = len(self.clients_X)

        # Record number of samples per client
        n_k = []
        for x in self.clients_X:
            n_k.append(x.shape[0])
        n_k = np.array(n_k)

        # Get initial validation accuracy for calculating reward
        loss = 500
        self.log.append(round(default_timer(), 2))

        if system == 'fedavg':
            m = max(int(frac_clients * K), 1)
            client_set = range(K)

            return {'K': K, 'm': m, 'client set': client_set, 'n_k': n_k, 'w': w, 'b': b}

        if system == 'confeddi':
            S = range(K)
            d = len(self.ContextElements)

            reward = 50
            self.V_ucb.append(reg_coeff * np.identity(d))
            self.b_ucb.append(np.zeros(d))

            # Context Elements
            X = np.zeros((K, d))
            r = np.zeros(K)

            return {'K': K, 'd': d, 'n_k': n_k, 'loss': loss, 'S': S, 'X': X, 'r': r, 'w': w, 'b': b, 'reward': reward}

    def FedAvg(self, lr: float = 0.001, epochs: int = 5, frac_clients: float = 1, rounds: int = 20) -> tuple[list[np.array], list[np.array]]:
        """
        Performs Federated Averaging on the given client data.
        Local client models are aggregated rounds times.
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        initializer = self.initialize('fedavg', frac_clients = frac_clients)

        K = initializer['K']
        m = initializer['m']
        client_set = initializer['client set']
        n_k = initializer['n_k']
        w = initializer['w']
        b = initializer['b']

        # Start federating process
        for t in range(rounds):
            if ((t + 1) % 5) == 0:
                print(f'Round {t + 1}')
            w_updates = [None for _ in range(K)]
            b_updates = [None for _ in range(K)]
            S = sample(client_set, m)

            # Client updates
            for k in S:
                w_updates[k], b_updates[k], _, _ = self.ClientUpdate(self.clients_X[k], self.clients_y[k], lr, w, b, epochs)

            w, b = self.aggregate(w_updates, b_updates, n_k, S)
            self.w_history.append(w)
            self.b_history.append(b)

            channel_time_t = np.sum(self.comm_times[S])

            marker = round(default_timer() + channel_time_t, 2)
            self.log.append(marker)

        self.log = np.array(self.log) - self.log[0]
            
        return w, b

    # ConFeddi
    def communication_times(self):
        # Simulates communication delay between client and server
        log2_vec = np.vectorize(log2)
        fading = 10 ** -12.81 * self.distances ** -3.76
        gain = 23
        noise_power = -107
        bandwidth = 15e3
        message_size = 5e3
        average_rate = log2_vec(1 + 10 ** (gain / 10) * fading / 10 ** (noise_power / 10))
        self.comm_times = message_size / (bandwidth * average_rate)

    def GetContext(self, data):
        # Generates context vector for given client
        context = np.zeros(len(self.ContextElements))

        for i, c in enumerate(self.ContextElements):
            element = self.ContextDef.get(c, invalid)
            context[i] = element(data)

        return context
    
    def UCB_scores(self, x, Vt, bt, alpha):
        # C2UCB with fixed alpha. Scores clients based on context elements
        K = x.shape[0]
        V_inv = np.linalg.inv(Vt)
        theta = np.dot(V_inv, bt)

        scores = []
        for i in range(K):
            term = np.dot(theta.T, x[i])
            score = term + alpha * np.sqrt(np.dot(x[i], V_inv).dot(x[i]))
            scores.append(score)

        return scores

    def ConFeddi(self, alpha: float, reg_coeff: float, lr: float = 0.001, epochs: int = 5, rounds: int =  20, Mt: np.array = None, deterministic: Union[int, bool] = 0) -> tuple[list[np.array], list[np.array]]:
        """
        Performs ConFeddi for the regressive task.
        Aggregates local model parameters rounds times.
        Selects clients based on context criteria.
        ----------------
        `alpha` is the exploration strength (fixed every round). Sometimes denoted as `a`.
        `reg_coeff` is lambda, the regularization strength for `UCB_scores()`. Sometimes denoted as `l`.
        `Mt` is a vector of how many clients are chosen each round.
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        initializer = self.initialize('confeddi', reg_coeff = reg_coeff)

        # Unpack
        K = initializer['K']
        d = initializer['d']
        n_k = initializer['n_k']
        loss = initializer['loss']
        S = initializer['S']
        reward = initializer['reward']
        X = initializer['X']
        r = initializer['r']
        w = initializer['w']
        b = initializer['b']

        for t in range(rounds):
            if ((t + 1) % 5) == 0:
                print(f'Round {t + 1}')

            round_start = 1 if deterministic else default_timer()
            w_updates = [None for _ in range(K)]
            b_updates = [None for _ in range(K)]
            reward_vec = []

            # Client updates
            for k in S:
                start = 2 if deterministic else default_timer()
                w_updates[k], b_updates[k], model, history = self.ClientUpdate(self.clients_X[k], self.clients_y[k], lr, w, b, epochs)
                end = 5 if deterministic else default_timer()

                reward_vec.append(history.history['val_loss'][-1])

                time = end - start
                X[k] = self.GetContext([model, time, history, reward, self.comm_times[k]])

            w, b = self.aggregate(w_updates, b_updates, n_k, S)
            self.w_history.append(w)
            self.b_history.append(b)
            
            round_end = 10 if deterministic else default_timer()

            # For logging time
            channel_time_t = np.sum(self.comm_times[S])

            # ConFeddi Portion
            X_standardized = X / np.linalg.norm(X, axis = 0)
            scores = self.UCB_scores(X_standardized, self.V_ucb[t], self.b_ucb[t], alpha)
            S = np.argsort(scores)[::-1][:Mt[t]]
            xx_sum = np.zeros((d, d))
            rx_sum = np.zeros(d)

            for k in S:
                x1 = X[k].reshape((d, 1))
                x2 = x1.T
                xx_sum += np.dot(x1, x2)

                rx = r[k] * X[k]
                rx_sum += rx

            self.V_ucb.append(self.V_ucb[t] + xx_sum)
            self.b_ucb.append(self.b_ucb[t] + rx_sum)

            marker = round(default_timer() + channel_time_t, 2)
            self.log.append(marker)

            # Compute reward
            curr_loss = np.array(reward_vec).mean()
            reward = np.absolute(loss - curr_loss) / (round_end - round_start)
            loss = curr_loss
            
        self.log = np.array(self.log) - self.log[0]
            
        return w, b

    # Performance
    def test_loss(self) -> list[float]:
        """
        For each global model generated during aggregation, evaluates on self.test_data.
        """

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        T = len(self.w_history)
        mse_losses = []

        test_data = self.test_data['Data']
        test_labels = self.test_data['Labels']

        for t in range(T):
            model = self.generate_model(self.w_history[t], self.b_history[t])
            model.compile(optimizer = 'adam', loss = 'mse')
            mse_loss = model.evaluate(test_data, test_labels, verbose = 0, use_multiprocessing = True)
            mse_losses.append(mse_loss)

        return mse_losses

    
