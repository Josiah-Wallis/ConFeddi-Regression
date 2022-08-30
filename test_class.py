import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


    def split(self, scheme = 1, args = None):
        """
        Distribute data based on given scheme
        - 1: random sampling
        - 2: correspondence sampling
        - 3: spatial sampling
        """
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
        self.r = len(a_search)
        self.c = len(l_search)
        count = 1

        for a in a_search:
            for l in l_search:
                print(f'Training Model {count}')
                w, b, conf_test_mse, conf_log = self.run_confeddi_test(a, l, lr = lr, rounds = rounds, epochs = epochs, deterministic = deterministic, context = context)
                self.conf_gs_history[(a, l)] = (conf_test_mse, conf_log)
                count += 1
                print()

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
        count = 1

        for context in context_elements:
            print(f'Training Model {count}')
            w, b, mse, log = self.run_confeddi_test(a, l, lr = lr, epochs = epochs, rounds = rounds, deterministic = deterministic, context = context)
            self.conf_as_history[tuple(context)] = (mse, log)
            
            if context == [0, 1, 2, 3, 4]:
                self.conf_as_base_mse = mse
                self.conf_as_base_log = log
            count += 1
            print()

        self.SetDefaultContext()

    def confeddi_as_test_plots(self, figsize, ylim, r, c, top = 0.92, titles = None):
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Ablation Study: Mt = {self.Mt[0]}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)
        
        plot = 1
        for run in list(self.conf_as_history.items())[1:]:
            context = run[0]
            err, time = run[1]
            fig.add_subplot(r, c, plot)
            plt.plot(time, err, color = 'blue', label = 'conf_mse', marker = 'o')
            plt.plot(self.fedavg_log, self.fedavg_test_mse, color = 'green', label = 'fedavg_mse', marker = 'o')
            plt.plot(self.conf_as_base_log, self.conf_as_base_mse, color = 'deepskyblue', label = 'conf_all', marker = 'o')

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
        for run in list(self.conf_as_history.items())[1:]:
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
