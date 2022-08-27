import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from confeddi import FederatedSystem
from distribute_data import generate_data
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONHASHSEED'] = str(50)

class Test():
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args

        self.data_seed = self.data_args['data seed']
        self.model_seed = self.data_args['model seed']
        self.distance_clients = self.data_args['distance clients']
        self.distance_augments = self.data_args['distance augments']
        self.tolerance = self.data_args['tolerance']

        exclude_dtypes = self.data_args['exclude dtypes']
        drop_labels = self.data_args['drop labels']
        target_labels = self.data_args['target labels']
        test_size1 = self.data_args['test size 1']
        test_size2 = self.data_args['test size 2']

        self.data = self.dataset.select_dtypes(exclude = exclude_dtypes)
        X = self.data.drop(columns = drop_labels).to_numpy()
        y = self.data[target_labels].to_numpy()

        print(f'Number of samples: {X.shape[0]}')
        print(f'Features per sample: {X.shape[1]}')

        print(f'Columns:')
        for i in self.data.drop(columns = drop_labels).columns[:-1]:
            if i in target_labels:
                continue
            print(f'{i}, ', end = '')
        print(self.data.drop(columns = drop_labels).columns[-1], end = '\n\n')

        X_split, X_val, y_split, y_val = train_test_split(X, y, test_size = test_size1, random_state = self.data_seed)
        X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size = test_size2, random_state = self.data_seed)

        self.data = generate_data(X_train, y_train, seed = self.data_seed, tolerance = self.tolerance)
        scaler = StandardScaler()
        X_val = scaler.fit_transform(X_val)
        X_test = scaler.fit_transform(X_test)

        self.data['Client Distances'][self.distance_clients] += self.distance_augments

        self.fed = FederatedSystem(self.data['Client Data'], self.data['Client Labels'], self.data['Client Distances'], seed = self.model_seed)
        self.fed.SetValData({'Val Data': X_val, 'Val Labels': y_val})
        self.fed.SetTestData({'Data': X_test, 'Labels': y_test})

        total_x = 0
        total_y = 0
        for a, b in zip(self.data['Client Data'], self.data['Client Labels']):
            total_x += len(a)
            total_y += len(b)
        print(f'Total Client Training Samples: {total_x}')
        print(f'Total Client Training Labels: {total_y}')
        print(f'Total Validation Samples: {len(X_val)}')
        print(f'Total Validation Labels: {len(y_val)}')
        print(f'Total Test Samples: {len(X_test)}')
        print(f'Total Test Labels: {len(y_test)}')

    def SetModel(self, model, default = 0):
        if not default:
            self.fed.SetModel(model)
        else:
            self.fed.DefaultModel()

    def SetDefaultContext(self):
        self.fed.SetContextElements([0, 1, 2, 3, 4])

    def load_baseline_fedavg_data(self, mse_path, log_path):
        self.fedavg_test_mse = np.load(mse_path)
        self.fedavg_log = np.load(log_path)

    def run_fedavg_test(self, lr = 0.001, epochs = 5, frac_clients = 1, rounds = 20):
        w, b = self.fed.FedAvg(lr = lr, epochs = epochs, frac_clients = frac_clients, rounds = rounds)
        fedavg_test_mse = self.fed.test_loss()
        fedavg_log = self.fed.GetLog()
        self.fed.clear_history()

        # add plots

        return w, b, fedavg_test_mse, fedavg_log

    def run_confeddi_test(self, alpha, reg_coeff, lr = 0.001, epochs = 5, rounds =  20, Mt = None, deterministic = 0, context = [0, 1, 2, 3, 4]):
        self.fed.SetContextElements(context)
        w, b = self.fed.ConFeddi(alpha, reg_coeff, lr, epochs, rounds, Mt, deterministic)
        conf_test_mse = self.fed.test_loss()
        conf_log = self.fed.GetLog()
        self.fed.clear_history()
        self.SetDefaultContext()

        # add comparison plots

        return w, b, conf_test_mse, conf_log

    def confeddi_gs(self, a_search, l_search, context = [0, 1, 2, 3, 4], lr = 0.001, epochs = 5, rounds =  20, Mt = None, deterministic = 0):
        self.conf_gs_history = dict()
        self.conf_gs_asearch = a_search
        self.conf_gs_lsearch = l_search
        self.conf_gs_Mt = Mt[0]
        count = 1

        for a in a_search:
            for l in l_search:
                print(f'Training Model {count}')
                w, b, conf_test_mse, conf_log = self.run_confeddi_test(a, l, lr = lr, rounds = rounds, epochs = epochs, Mt = Mt, deterministic = deterministic, context = context)
                self.conf_gs_history[(a, l)] = (conf_test_mse, conf_log)
                count += 1
                print()

    def confeddi_gs_test_plots(self, figsize, ylim1, ylim2, top = 0.95):
        r = len(self.conf_gs_asearch)
        c = len(self.conf_gs_lsearch)
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Test Loss: Mt = {self.conf_gs_Mt}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)

        plot = 1
        for run in self.conf_gs_history.items():
            a, l = run[0]
            err, time = run[1]
            fig.add_subplot(r, c, plot)
            plt.plot(time, err, color = 'blue', label = 'conf_mse', marker = 'o')
            plt.plot(self.fedavg_log, self.fedavg_test_mse, color = 'green', label = 'fedavg_mse', marker = 'o')
            plt.title(f'a = {a}, l = {l}')
            plt.ylim(ylim1, ylim2)
            plt.ylabel('Error')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.legend()

            plot += 1

    def confeddi_gs_improvement_plots(self, figsize, top = 0.95):
        r = len(self.conf_gs_asearch)
        c = len(self.conf_gs_lsearch)
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Improvement: Mt = {self.conf_gs_Mt}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)

        plot = 1
        for run in self.conf_gs_history.items():
            err, _ = run[1]
            fig.add_subplot(r, c, plot)

            ratios = np.array(err[1:]) / np.array(self.fedavg_test_mse[1:]) 
            plt.plot(ratios, color = 'red', label = 'Conf / Fedavg', marker = 'o')

            improvement = 1 - ratios.mean()
            if improvement < 0: improvement = 0
            plt.title(f'Improvement: {improvement * 100:.2f}%')
            plt.ylabel('Error Ratio')
            plt.xlabel('Rounds')
            plt.hlines(1, 0, len(ratios), color = 'green', label = 'Baseline')
            plt.hlines(ratios.mean(), 0, len(ratios), color = 'blue', label = 'Average Ratio', linestyle = 'dashed')
            plt.grid()
            plt.legend()
            plot += 1


    def confeddi_as(self, context_elements, a, l, lr = 0.001, epochs = 5, rounds =  20, Mt = None, deterministic = 0):
        self.conf_as_history = dict()
        self.conf_as_Mt = Mt[0]
        count = 1

        for context in context_elements:
            print(f'Training Model {count}')
            w, b, mse, log = self.run_confeddi_test(a, l, lr = lr, epochs = epochs, rounds = rounds, Mt = Mt, deterministic = deterministic, context = context)
            self.conf_as_history[tuple(context)] = (mse, log)
            
            if context == [0, 1, 2, 3, 4]:
                self.conf_as_base_mse = mse
                self.conf_as_base_log = log
            count += 1
            print()

        self.SetDefaultContext()

    def confeddi_as_test_plots(self, figsize, ylim1, ylim2, r, c, top = 0.92, titles = None):
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Ablation Study: Mt = {self.conf_as_Mt}', fontsize = 'xx-large')
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

            plt.ylim(ylim1, ylim2)
            plt.ylabel('Error')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.legend()
            plot += 1

    def confeddi_as_improvement_plots(self, figsize, r, c, top = 0.92):
        fig = plt.figure(figsize = figsize)
        plt.suptitle(f'Ablation Study: Mt = {self.conf_as_Mt}', fontsize = 'xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top = top)

        plot = 1
        for run in list(self.conf_as_history.items())[1:]:
            err, _ = run[1]
            fig.add_subplot(r, c, plot)

            ratios = np.array(err[1:]) / np.array(self.fedavg_test_mse[1:]) 
            plt.plot(ratios, color = 'red', label = 'Conf / Fedavg', marker = 'o')

            improvement = 1 - ratios.mean()
            if improvement < 0: improvement = 0
            plt.title(f'Improvement: {improvement * 100:.2f}%')
            plt.ylabel('Error Ratio')
            plt.xlabel('Rounds')
            plt.hlines(1, 0, len(ratios), color = 'green', label = 'Baseline')
            plt.hlines(ratios.mean(), 0, len(ratios), color = 'blue', label = 'Average Ratio', linestyle = 'dashed')
            plt.grid()
            plt.legend()

            plot += 1

    def plot_error(self, pairs, colors, labels, ylim1, ylim2):
        for i, p in enumerate(pairs):
            plt.plot(p[0], p[1], color = colors[i], label = labels[i], marker = 'o')

        plt.title('Test MSE')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.ylim(ylim1, ylim2)
        plt.grid()
        plt.legend()

    def plot_improvement(self, conf_mse):
        ratios = np.array(conf_mse[1:]) / np.array(self.fedavg_test_mse[1:])
        improvement = 1 - ratios.mean()
        
        if improvement < 0: improvement = 0
        plt.plot(ratios, color = 'red', label = 'Conf / Fedavg', marker = 'o')
        plt.hlines(1, 0, len(conf_mse), color = 'green', label = 'Baseline')
        plt.hlines(ratios.mean(), 0, len(conf_mse), color = 'blue', label = 'Average Ratio', linestyle = 'dashed')

        plt.title(f'Improvement: {improvement * 100:.2f}%')
        plt.ylabel('Error Ratio')
        plt.xlabel('Rounds')
        plt.grid()
        plt.legend()
