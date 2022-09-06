import multiprocessing
import os
from tkinter import W
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from confeddi import FederatedSystem
from distribute_data import generate_data
from dataset import RTTSplitStrategy
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONHASHSEED'] = str(50)

# finish adding docstrings

class Test():
    def __init__(self, dataset, data_args, Mt, model_seed):
        # Handler for splitting data
        self.StrategyHandler = RTTSplitStrategy(dataset, data_args)
        self.opts = {
            1: self.StrategyHandler.random,
            2: self.StrategyHandler.correspondence,
            3: self.StrategyHandler.spatial
        }

        # Federated System
        self.fed = None
        self.Mt = Mt
        self.model_seed = model_seed

        # For CV
        self.data_args = data_args


    def split(self, scheme = 1, args = None):
        """
        Distribute data based on given scheme
        - 1: random sampling
        - 2: correspondence sampling
        - 3: spatial sampling
        """
        self.scheme = scheme

        # Choose split scheme
        split_func = self.opts[scheme]
        pkg = split_func(args)

        # Unpack
        data = pkg['Split Data']
        val = pkg['Validation']
        test = pkg['Test']

        # Build federated system
        self.fed = FederatedSystem(data['Client Data'], data['Client Labels'], data['Client Distances'])
        self.fed.SetValData(val)
        self.fed.SetTestData(test)

    def display_metadata(self):
        self.StrategyHandler.display_metadata()
    
    def display_client_distribution(self):
        self.StrategyHandler.display_client_distribution()

    def SetModel(self, model, default = 0):
        if not default:
            self.fed.SetModel(model)
        else:
            self.fed.DefaultModel()

    def SetDefaultContext(self):
        self.fed.SetContextElements([0, 1, 2, 3, 4])

    def SetMt(self, Mt):
        self.Mt = Mt

    def GetDataset(self):
        return self.StrategyHandler.dataset

    def GetDataSeed(self):
        return self.StrategyHandler.data_seed

    def GetModelSeed(self):
        return self.model_seed

    def load_baseline_fedavg_data(self, mse_path, log_path):
        self.fedavg_test_mse = np.load(mse_path)
        self.fedavg_log = np.load(log_path)

    def run_fedavg_test(self, lr = 0.001, epochs = 5, frac_clients = 1, rounds = 20):
        w, b = self.fed.FedAvg(lr = lr, epochs = epochs, frac_clients = frac_clients, rounds = rounds)
        fedavg_test_mse = self.fed.test_loss()
        fedavg_log = self.fed.GetLog()
        self.fed.clear_history()

        return w, b, fedavg_test_mse, fedavg_log

    def run_confeddi_test(self, alpha, reg_coeff, lr = 0.001, epochs = 5, rounds =  20, deterministic = 0, context = [0, 1, 2, 3, 4]):
        self.fed.SetContextElements(context)
        w, b = self.fed.ConFeddi(alpha, reg_coeff, lr, epochs, rounds, self.Mt, deterministic)
        conf_test_mse = self.fed.test_loss()
        conf_log = self.fed.GetLog()
        self.fed.clear_history()
        self.SetDefaultContext()

        return w, b, conf_test_mse, conf_log

    def confeddi_gs(self, a_search, l_search, context = [0, 1, 2, 3, 4], lr = 0.001, epochs = 5, rounds =  20, deterministic = 0):
        self.conf_gs_history = dict()
        conf_gs_wb_history = dict()
        self.r = len(a_search)
        self.c = len(l_search)
        count = 1

        for a in a_search:
            for l in l_search:
                print(f'Training Model {count}')
                w, b, conf_test_mse, conf_log = self.run_confeddi_test(a, l, lr = lr, rounds = rounds, epochs = epochs, deterministic = deterministic, context = context)
                self.conf_gs_history[(a, l)] = (conf_test_mse, conf_log)
                conf_gs_wb_history[(a, l)] = (w, b)
                count += 1
                print()

        return conf_gs_wb_history

    def confeddi_gs_test_plots(self, figsize, ylim, top = 0.95):
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Test Loss: Mt = {self.Mt[0]}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)

        plot = 1
        for run in self.conf_gs_history.items():
            a, l = run[0]
            err, time = run[1]
            fig.add_subplot(self.r, self.c, plot)
            plt.plot(time, err, color = 'blue', label = 'conf_mse', marker = 'o')
            plt.plot(self.fedavg_log, self.fedavg_test_mse, color = 'green', label = 'fedavg_mse', marker = 'o')
            plt.title(f'a = {a}, l = {l}')
            plt.ylim(ylim[0], ylim[1])
            plt.ylabel('Error')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.legend()

            plot += 1

    def confeddi_gs_improvement_plots(self, figsize, ylim = None, top = 0.95, trim_bias = 0):
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Improvement: Mt = {self.Mt[0]}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)

        plot = 1
        for run in self.conf_gs_history.items():
            # Unpack
            err, _ = run[1]
            fig.add_subplot(self.r, self.c, plot)

            #
            ratios = np.array(err[1:]) / np.array(self.fedavg_test_mse[1:]) 
            med = np.median(ratios)
            if not trim_bias:
                proper_mean = ratios.mean()
                improvement = 1 - proper_mean
                if improvement < 0: improvement = 0
                title = f'Improvement: {improvement * 100:.2f}%'
            else:
                proper_mean = ratios[ratios < med + trim_bias].mean()
                improvement = 1 - proper_mean
                if improvement < 0: improvement = 0
                title = f'Improvement: {improvement * 100:.2f}%, Median: {med:.2f}'

            plt.plot(ratios, color = 'red', label = 'Conf / Fedavg', marker = 'o')
            plt.title(title)
            plt.ylabel('Error Ratio')
            plt.xlabel('Rounds')
            plt.hlines(1, 0, len(ratios), color = 'green', label = 'Baseline')
            plt.hlines(proper_mean, 0, len(ratios), color = 'blue', label = 'Trimmed Ratio', linestyle = 'dashed')
            plt.grid()
            plt.legend()

            if ylim:
                plt.ylim(ylim[0], ylim[1])

            plot += 1

    def confeddi_as(self, context_elements, a, l, lr = 0.001, epochs = 5, rounds =  20, deterministic = 0):
        self.conf_as_history = dict()
        conf_as_wb_history = dict()
        count = 1

        for context in context_elements:
            print(f'Training Model {count}')
            w, b, mse, log = self.run_confeddi_test(a, l, lr = lr, epochs = epochs, rounds = rounds, deterministic = deterministic, context = context)
            self.conf_as_history[tuple(context)] = (mse, log)
            conf_as_wb_history[tuple(context)] = (w, b)
            
            count += 1
            print()

        return conf_as_wb_history

    def confeddi_as_test_plots(self, figsize, ylim, r, c, top = 0.92, titles = None):
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Ablation Study: Mt = {self.Mt[0]}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)
        
        plot = 1
        for run in self.conf_as_history.items():
            context = run[0]
            err, time = run[1]
            fig.add_subplot(r, c, plot)
            plt.plot(time, err, color = 'blue', label = 'conf_mse', marker = 'o')
            plt.plot(self.fedavg_log, self.fedavg_test_mse, color = 'green', label = 'fedavg_mse', marker = 'o')
            plt.plot(self.conf_as_history[(0, 1, 2, 3, 4)][1], self.conf_as_history[(0, 1, 2, 3, 4)][0], color = 'deepskyblue', label = 'conf_all', marker = 'o')

            if titles:
                plt.title(titles[context])
            else:
                plt.title(context)

            plt.ylim(ylim[0], ylim[1])
            plt.ylabel('Error')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.legend()
            plot += 1

    def confeddi_as_improvement_plots(self, figsize, r, c, ylim = None, top = 0.92, trim_bias = 0):
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Ablation Study: Mt = {self.Mt[0]}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)

        plot = 1
        for run in self.conf_as_history.items():
            #
            err, _ = run[1]
            fig.add_subplot(r, c, plot)

            #
            ratios = np.array(err[1:]) / np.array(self.fedavg_test_mse[1:]) 
            med = np.median(ratios)
            if not trim_bias:
                proper_mean = ratios.mean()
                improvement = 1 - proper_mean
                if improvement < 0: improvement = 0
                title = f'Improvement: {improvement * 100:.2f}%'
            else:
                proper_mean = ratios[ratios < med + trim_bias].mean()
                improvement = 1 - proper_mean
                if improvement < 0: improvement = 0
                title = f'Improvement: {improvement * 100:.2f}%, Median: {med:.2f}'

            plt.plot(ratios, color = 'red', label = 'Conf / Fedavg', marker = 'o')
            plt.title(title)
            plt.ylabel('Error Ratio')
            plt.xlabel('Rounds')
            plt.hlines(1, 0, len(ratios), color = 'green', label = 'Baseline')
            plt.hlines(proper_mean, 0, len(ratios), color = 'blue', label = 'Trimmed Ratio', linestyle = 'dashed')
            plt.grid()
            plt.legend()

            if ylim:
                plt.ylim(ylim[0], ylim[1])

            plot += 1

    # had a norm_features arg for which features to normalize over
    def cross_validation(self, K, score, a, l, rounds = 50, context = [0, 1, 2, 3, 4], args = None):
        tf.keras.utils.set_random_seed(self.data_args['data seed'])
        X = self.StrategyHandler.X.copy()
        y = self.StrategyHandler.y.copy()

        shuffle_idxs = np.arange(len(X))
        np.random.shuffle(shuffle_idxs)

        scaler = StandardScaler()
        
        if self.scheme == 1:
            X = X.to_numpy()[shuffle_idxs]
            y = y.to_numpy()[shuffle_idxs]
            splits = np.linspace(0, len(X), K + 1).astype('int32')

            chunks_X = []
            chunks_y = []
            i = 0
            j = 1
            while j != len(splits):
                chunks_X.append(X[splits[i]:splits[j]])
                chunks_y.append(y[splits[i]:splits[j]])
                i += 1
                j += 1

            for k in range(K):
                print(f'CV Pair {k + 1}')
                GVGT_X = chunks_X[k]
                GVGT_y = chunks_y[k]
                split_idx = int(len(GVGT_X) / 2)

                X_val, y_val = GVGT_X[:split_idx], GVGT_y[:split_idx]
                X_test, y_test = GVGT_X[split_idx:], GVGT_y[split_idx:]

                X_train = np.concatenate([x for idx, x in enumerate(chunks_X) if idx != k])
                y_train = np.concatenate([x for idx, x in enumerate(chunks_y) if idx != k])

                final_data = generate_data(X_train, y_train, seed = self.data_args['data seed'], tolerance = self.data_args['tolerance'])
                X_val = scaler.fit_transform(X_val)
                X_test = scaler.fit_transform(X_test)

                final_data['Client Distances'][self.data_args['distance clients']] += self.data_args['distance augments']

                val = {'Val Data': X_val, 'Val Labels': y_val}
                test = {'Data': X_test, 'Labels': y_test}
                self.fed = FederatedSystem(final_data['Client Data'], final_data['Client Labels'], final_data['Client Distances'])
                self.fed.SetValData(val)
                self.fed.SetTestData(test)

                print('FedAvg')
                w_fedavg, b_fedavg, fedavg_mse, fedavg_log = self.run_fedavg_test(rounds = rounds, frac_clients = 0.5)
                print(f'Time: {fedavg_log[-1]}')
                print('\nConFeddi')
                w, b, conf_mse, conf_log = self.run_confeddi_test(a, l, rounds = rounds, context = context)
                print(f'Time: {conf_log[-1]}', end = '\n\n')

                s1_fedavg, s2_fedavg = self.average_error((w_fedavg, b_fedavg))
                s1, s2 = self.average_error((w, b))

                print(f'FedAvg Avg Test Error: {s1_fedavg}')
                print(f'ConFeddi Avg Test Error: {s1}')
                print(f'FedAvg MSE: {s2_fedavg}')
                print(f'ConFeddi MSE: {s2}', end = '\n\n')

                if k == 0:
                    best_w = w
                    best_b = b
                    best_s1 = s1
                    best_s2 = s2
                    curr_fedavg_mse = fedavg_mse
                    curr_conf_mse = conf_mse
                    curr_fedavg_log = fedavg_log
                    curr_conf_log = conf_log
                    pkg = {'Clients': final_data, 'Validation': val, 'Test': test}
                elif (score == 1) & (s1 < best_s1):
                    best_s1 = s1
                    best_w = w
                    best_b = b
                    curr_fedavg_mse = fedavg_mse
                    curr_conf_mse = conf_mse
                    curr_fedavg_log = fedavg_log
                    curr_conf_log = conf_log
                    pkg = {'Clients': final_data, 'Validation': val, 'Test': test}
                elif (score == 2) & (s2 < best_s2):
                    best_s2 = s2
                    best_w = w
                    best_b = b
                    curr_fedavg_mse = fedavg_mse
                    curr_conf_mse = conf_mse
                    curr_fedavg_log = fedavg_log
                    curr_conf_log = conf_log
                    pkg = {'Clients': final_data, 'Validation': val, 'Test': test}

        return (best_w, best_b), (curr_fedavg_mse, curr_conf_mse), (curr_fedavg_log, curr_conf_log), pkg

                





    def plot_error(self, pairs, colors, labels, ylim):
        for i, p in enumerate(pairs):
            plt.plot(p[0], p[1], color = colors[i], label = labels[i], marker = 'o')

        plt.title('Test MSE')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.ylim(ylim[0], ylim[1])
        plt.grid()
        plt.legend()

    def plot_improvement(self, conf_mse, ylim = None, trim_bias = 0):
        ratios = np.array(conf_mse[1:]) / np.array(self.fedavg_test_mse[1:]) 
        med = np.median(ratios)
        if not trim_bias:
            proper_mean = ratios.mean()
            improvement = 1 - proper_mean
            if improvement < 0: improvement = 0
            title = f'Improvement: {improvement * 100:.2f}%'
        else:
            proper_mean = ratios[ratios < med + trim_bias].mean()
            improvement = 1 - proper_mean
            if improvement < 0: improvement = 0
            title = f'Improvement: {improvement * 100:.2f}%, Median: {med:.2f}'

        plt.plot(ratios, color = 'red', label = 'Conf / Fedavg', marker = 'o')
        plt.hlines(1, 0, len(conf_mse), color = 'green', label = 'Baseline')
        plt.hlines(proper_mean, 0, len(conf_mse), color = 'blue', label = 'Trimmed Ratio', linestyle = 'dashed')
        plt.title(title)
        plt.ylabel('Error Ratio')
        plt.xlabel('Rounds')
        plt.grid()
        plt.legend()

        if ylim:
            plt.ylim(ylim[0], ylim[1])

    def average_error(self, history):
        tf.keras.utils.set_random_seed(self.model_seed)
        model = self.fed.generate_model(history[0], history[1])
        pred = model.predict(self.fed.test_data['Data'], verbose = 0)
        ytest = self.fed.test_data['Labels']
        diff = np.absolute(pred - ytest)
        avg_error = (np.sum(diff) / len(ytest))

        model.compile(optimizer = 'adam', loss = 'mse')
        loss = model.evaluate(self.fed.test_data['Data'], self.fed.test_data['Labels'], verbose = 0, use_multiprocessing = True)
        return avg_error, loss
