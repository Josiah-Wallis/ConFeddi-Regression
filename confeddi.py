import tensorflow as tf
import numpy as np
import os
import random

from math import log2
from random import sample
from timeit import default_timer
from inspect import signature
from sklearn.model_selection import train_test_split
from tensorflow.random import set_seed
from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam

#model, time, r_ti, prev_r_t, history, comm_time
def total_time(data):
    return data[1] + data[-1]
def noise_info(data):
    history = data[4].history
    r_ti = data[2]
    return np.absolute(history['loss'][-1] - r_ti)
def normalized_local_training_loss(data):
    history = data[4].history
    return history['loss'][-1] / history['loss'][0]
def normalized_local_validation_loss(data):
    history = data[4].history
    return history['val_loss'][-1] / history['val_loss'][0]
def prev_reward(data):
    return data[3]
def invalid():
    raise Exception('Invalid')

class FederatedSystem:
    def __init__(self, clients_X, clients_y, distances = None, seed = 50):
        self.seed = seed
        self.log = []
        self.clients_X = clients_X
        self.clients_y = clients_y
        self.distances = distances
        self.comm_times = None

        self.trainable_layers = []
        self.w_history = []
        self.b_history = []

        self.ContextElements = [0, 1, 2, 3, 4]
        self.ContextDef = {
            0: total_time,
            1: noise_info,
            2: normalized_local_training_loss,
            3: normalized_local_validation_loss,
            4: prev_reward
        }

        self.model = Sequential([
            Dense(32, activation = 'relu', input_shape = (10,)),
            Dense(16, activation = 'relu'),
            Dense(1, activation = 'relu')
        ])

        # ConFeddi
        self.V_ucb = []
        self.b_ucb = []

        # Performance
        self.val_data = None
        self.test_data = None
        self.val_loss = []

    def clear_data(self):
        self.clients_X = None
        self.clients_y = None
        self.distances = None
        self.val_data = None
        self.test_data = None

    def clear_history(self):
        self.log = []
        self.comm_times = []
        self.trainable_layers = []
        self.w_history = []
        self.b_history = []
        self.V_ucb = []
        self.b_ucb = []
        self.val_loss = []


    # Setters + Getters
    def SetModel(self, model):
        self.model = model

    def SetValData(self, val_data):
        self.val_data = val_data

    def SetTestData(self, test_data):
        self.test_data = test_data
    
    def SetSeed(self, seed):
        self.seed = seed

    def SetContextElements(self, options):
        self.ContextElements = options

    def GetLog(self):
        return self.log

    def AddContextElements(self, func):
        keys = list(self.ContextDef.keys()).sort()
        new_key = keys[-1] + 1
        self.ContextDef[new_key] = func
        self.ContextElements += [new_key]

    # FEDAVG
    def initialize_weights(self):
        w = []
        b = []

        for l in self.trainable_layers:
            w_shape = self.model.layers[l].weights[0].shape
            b_shape = self.model.layers[l].weights[1].shape
            w.append(np.random.standard_normal(w_shape))
            b.append(np.random.standard_normal(b_shape))

        return w, b

    def generate_model(self, w, b, skip = 0):
        model = clone_model(self.model)

        if not skip:
            for i, x in enumerate(self.trainable_layers):
                model.layers[x].set_weights([w[i], b[i]])

        return model

    def ClientUpdate(self, X, y, w, b, E):
        model = self.generate_model(w, b)
        model.compile(optimizer = 'adam', loss = 'mse')
        history = model.fit(X, y, validation_split = 0.2, epochs = E, verbose = 0, shuffle = False, use_multiprocessing = True)

        w = []
        b = []
        for l in self.trainable_layers:
            w.append(model.layers[l].get_weights()[0])
            b.append(model.layers[l].get_weights()[1])

        return w, b, model, history

    def aggregate(self, w_updates, b_updates, n_k, S_t):
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

    def initialize(self, system, reg_coeff = None, frac_clients = None) -> dict:
        # Compute communication times from distances
        self.communication_times()

        # Record trainable layers
        for i, x in enumerate(self.model.layers):
            if x.weights:
                self.trainable_layers.append(i)

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
        model = self.generate_model(w, b)
        model.compile(optimizer = 'adam', loss = 'mse')
        loss = model.evaluate(self.val_data['Val Data'], self.val_data['Val Labels'], verbose = 0, use_multiprocessing = True)
        self.log.append(default_timer())
        self.val_loss.append(loss)

        if system == 'fedavg':
            m = max(int(frac_clients * K), 1)
            client_set = range(K)

            return {'K': K, 'm': m, 'client set': client_set, 'n_k': n_k, 'w': w, 'b': b}

        if system == 'confeddi':
            S = range(K)
            vcr = 1
            d = len(self.ContextElements)

            self.V_ucb.append(reg_coeff * np.identity(d))
            self.b_ucb.append(np.zeros(d))

            # Context Elements
            X = np.zeros((K, d))
            r = np.zeros(K)

            return {'K': K, 'd': d, 'n_k': n_k, 'loss': loss, 'S': S, 'vcr': vcr, 'X': X, 'r': r, 'w': w, 'b': b}

    def FedAvg(self, epochs = 5, frac_clients = 1, rounds = 20):
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
            w_updates = [None for _ in range(m)]
            b_updates = [None for _ in range(m)]
            S = sample(client_set, m)

            # Client updates
            for k in S:
                w_updates[k], b_updates[k], _, _ = self.ClientUpdate(self.clients_X[k], self.clients_y[k], w, b, epochs)

            w, b = self.aggregate(w_updates, b_updates, n_k, S)
            self.w_history.append(w)
            self.b_history.append(b)

            channel_time_t = np.sum(self.comm_times[S])

            model = self.generate_model(w, b)
            model.compile(optimizer = 'adam', loss = 'mse')
            loss = model.evaluate(self.val_data['Val Data'], self.val_data['Val Labels'], verbose = 0, use_multiprocessing = True)
            self.log.append(default_timer() + channel_time_t)
            self.val_loss.append(loss)

        self.log = np.array(self.log) - self.log[0]
            
        return w, b

    # CONFEDDI
    def communication_times(self):
        log2_vec = np.vectorize(log2)
        fading = 10 ** -12.81 * self.distances ** -3.76
        gain = 23
        noise_power = -107
        bandwidth = 15e3
        message_size = 5e3
        average_rate = log2_vec(1 + 10 ** (gain / 10) * fading / 10 ** (noise_power / 10))
        self.comm_times = message_size / (bandwidth * average_rate)

    def GetContext(self, data: dict) -> np.ndarray:
        context = np.zeros(len(self.ContextElements))

        for i, c in enumerate(self.ContextElements):
            element = self.ContextDef.get(c, invalid)
            context[i] = element(data)

        return context
    
    # C2UCB, fixed alpha
    def UCB_scores(self, x, Vt, bt, alpha, S):
        K = x.shape[0]
        d = x.shape[1]
        V_inv = np.linalg.inv(Vt)
        theta = np.dot(V_inv, bt)

        scores = []
        for i in range(K):
            term = np.dot(theta.T, x[i])
            score = term + alpha * np.sqrt(np.dot(x[i], V_inv).dot(x[i]))
            scores.append(score)

        return scores

    def ConFeddi(self, alpha, reg_coeff, epochs = 5, rounds =  20, Mt = None, deterministic = 0):
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        initializer = self.initialize('confeddi', reg_coeff = reg_coeff)

        # Unpack
        K = initializer['K']
        d = initializer['d']
        n_k = initializer['n_k']
        loss = initializer['loss']
        S = initializer['S']
        vcr = initializer['vcr']
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

            # Client updates
            for k in S:
                start = 2 if deterministic else default_timer()
                w_updates[k], b_updates[k], model, history = self.ClientUpdate(self.clients_X[k], self.clients_y[k], w, b, epochs)
                end = 5 if deterministic else default_timer()

                time = round(end - start, 2)
                r[k] = model.evaluate(self.val_data['Val Data'], self.val_data['Val Labels'], verbose = 0, use_multiprocessing = True)
                X[k] = self.GetContext([model, time, r[k], vcr, history, self.comm_times[k]])

            w, b = self.aggregate(w_updates, b_updates, n_k, S)
            self.w_history.append(w)
            self.b_history.append(b)
            
            round_end = 10 if deterministic else default_timer()

            # For logging time
            channel_time_t = np.sum(self.comm_times[S])

            # ConFeddi Portion
            X_standardized = X / np.linalg.norm(X, axis = 0)
            scores = self.UCB_scores(X_standardized, self.V_ucb[t], self.b_ucb[t], alpha, S)
            S = np.argsort(scores)[::-1][:Mt[t]]
            xx_sum = np.zeros((d, d))
            rx_sum = np.zeros(d)

            # used to be for i in range(len(S)) and all ks were is
            for k in S:
                x1 = X[k].reshape((d, 1))
                x2 = x1.T
                xx_sum += np.dot(x1, x2)

                rx = r[k] * X[k]
                rx_sum += rx

            self.V_ucb.append(self.V_ucb[t] + xx_sum)
            self.b_ucb.append(self.b_ucb[t] + rx_sum)

            model = self.generate_model(w, b)
            model.compile(optimizer = 'adam', loss = 'mse')
            curr_loss = model.evaluate(self.val_data['Val Data'], self.val_data['Val Labels'], verbose = 0, use_multiprocessing = True)
            self.log.append(default_timer() + channel_time_t)
            vcr = np.absolute(loss - curr_loss) / (round_end - round_start)
            loss = curr_loss
            self.val_loss.append(loss)
            
        self.log = np.array(self.log) - self.log[0]
            
        return w, b

    # PERFORMANCE
    def val_loss(self):
        return self.val_loss

    def test_loss(self):
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

    