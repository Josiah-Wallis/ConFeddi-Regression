import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Iterable, Any, Union
from sklearn.preprocessing import StandardScaler
from confeddi import FederatedSystem
from distribute_data import generate_data
from dataset import RTTSplitStrategy
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONHASHSEED'] = str(50)

class Test():
    def __init__(self, dataset: pd.DataFrame, data_args: dict, Mt: np.array, model_seed: int) -> None:
        """
        Unpack data and arguments to build test suite.
        Runs FedAvg and ConFeddi Tests: 
        - Single Run
        - Grid Search
        - Ablation Study
        - Cross Validation
        
        Provides analytics:
        - Test error curves (MSE)
        - Comparison graphs between FedAvg and ConFeddi
        - Average error on test set
        """

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

        # For records
        self.fedavg_test_mse = None
        self.fedavg_log = None

        # For CV
        self.data_args = data_args

    def split(self, scheme: int = 1, args: Iterable[Any] = None) -> None:
        """
        Distribute data based on given scheme
        - 1: random sampling
        - 2: correspondence sampling
        - 3: spatial sampling.
        Builds FL System.
        """

        # For records
        self.scheme = scheme

        # Choose split scheme
        split_func = self.opts[scheme]
        pkg = split_func(args)

        # Unpack
        data = pkg['Split Data']
        test = pkg['Test']

        # Build federated system
        self.fed = FederatedSystem(data['Client Data'], data['Client Labels'], data['Client Distances'])
        self.fed.SetTestData(test)

    def display_metadata(self) -> None:
        """
        Displays metadata regarding data distribution
        - Total Samples
        - Number of Clients
        - Training and Test split
        - Percent of total data per client
        """

        self.StrategyHandler.display_metadata()
    
    def display_client_distribution(self) -> None:
        """
        Displays data distribution among clients
        - Data Distribution as a percent of total client samples
        - Client Distance Distribution w.r.t. max distance
        """

        self.StrategyHandler.display_client_distribution()

    def SetModel(self, model: tf.keras.Model, default: Union[int, bool] = 0) -> None:
        """
        Sets FL local models to model, or the default model of default == 1.
        """
        
        if not default:
            self.fed.SetModel(model)
        else:
            self.fed.DefaultModel()

    def SetDefaultContext(self) -> None:
        """
        Sets FL system's context elements to all contexts.
        """

        self.fed.SetContextElements([0, 1, 2, 3])

    def SetMt(self, Mt: np.array) -> None:
        """
        Sets number of clients selected each round.
        """

        self.Mt = Mt

    def SetDataSeed(self, seed: int) -> None:
        """
        Set seed used to distribute and split data
        """

        self.data_args['data seed'] = seed

    def SetFedAvgBaseline(self, fedavg_test_mse: list[float], fedavg_log: list[float]) -> None:
        """
        Stores FedAvg test error and log history for comparisons.
        """

        self.fedavg_test_mse = fedavg_test_mse
        self.fedavg_log = fedavg_log

    def GetDataset(self) -> pd.DataFrame:
        """
        Returns the original RTT dataset without the complex-valued columns
        """

        return self.StrategyHandler.dataset

    def GetDataSeed(self) -> int:
        """
        Returns seed used to split data
        """

        return self.StrategyHandler.data_seed

    def GetModelSeed(self) -> int:
        """
        Returns seed used for average error computation (model.predict).
        """

        return self.model_seed

    def load_baseline_fedavg_data(self, mse_path: str, log_path: str) -> None:
        """
        Loads saved losses and logs of baseline FedAvg model from files.
        """

        self.fedavg_test_mse = np.load(mse_path)
        self.fedavg_log = np.load(log_path)

    def run_fedavg_test(self, lr: float = 0.001, epochs: int = 5, frac_clients: float = 1, rounds: int = 20) -> tuple[list[np.array], list[np.array], list[float], np.array]:
        """
        Runs FedAvg and returns relevant information:
        - time markers
        - test loss at each aggregation process
        """

        w, b = self.fed.FedAvg(lr = lr, epochs = epochs, frac_clients = frac_clients, rounds = rounds)
        fedavg_test_mse = self.fed.test_loss()
        fedavg_log = self.fed.GetLog()
        self.fed.clear_history()

        return w, b, fedavg_test_mse, fedavg_log

    def run_confeddi_test(self, alpha: float, reg_coeff: float, lr: float = 0.001, epochs: int = 5, rounds: int =  20, deterministic: Union[int, bool] = 0, context: list[int] = [0, 1, 2, 3]) -> tuple[list[np.array], list[np.array], list[float], np.array]:
        """
        Runs ConFeddi and returns relevant information:
        - time markers
        - test loss at each aggregation process
        """

        self.fed.SetContextElements(context)
        w, b = self.fed.ConFeddi(alpha, reg_coeff, lr, epochs, rounds, self.Mt, deterministic)
        conf_test_mse = self.fed.test_loss()
        conf_log = self.fed.GetLog()
        self.fed.clear_history()
        self.SetDefaultContext()

        return w, b, conf_test_mse, conf_log

    def confeddi_gs(self, a_search: Iterable[Any], l_search: Iterable[Any], context: list[int] = [0, 1, 2, 3], lr: float = 0.001, epochs: int = 5, rounds: int =  20, deterministic: Union[int, bool] = 0) -> dict:
        """
        Runs a ConFeddi gridsearch.
        Returns the model parameters for each model.
        """

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

    def confeddi_gs_test_plots(self, figsize: tuple[float], ylim: tuple[float], top: float = 0.95) -> None:
        """
        Plots the test error curves of each model (separately) in previously-ran gridsearch.
        Plots curves w.r.t. baseline FedAvg test error curve.
        """

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

    def confeddi_as(self, context_elements: list[list[int]], a: float, l: float, lr: float = 0.001, epochs: int = 5, rounds: int =  20, deterministic: Union[int, bool] = 0) -> dict:
        """
        Runs a ConFeddi ablation study by running ConFeddi with subsets of the full context vector.
        Returns the model parameters for each model.
        """

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

    def confeddi_as_test_plots(self, figsize: tuple[float], ylim: tuple[float], r: int, c: int, top: float = 0.92, titles: list[str] = None) -> None:
        """
        Plots test error curves of each ablation study model (separately) w.r.t. baseline FedAvg test error curve.
        """

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
            plt.plot(self.conf_as_history[(0, 1, 2, 3)][1], self.conf_as_history[(0, 1, 2, 3)][0], color = 'deepskyblue', label = 'conf_all', marker = 'o')

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

    def scheme1_cv(self, X, y, shuffle_idxs, cv_args):
        # Performs cross validation for scheme 1 data split

        tf.keras.utils.set_random_seed(self.data_args['data seed'])

        # Unpack cv args
        K = cv_args['K']
        score = cv_args['score']
        a = cv_args['a']
        l = cv_args['l']
        rounds = cv_args['rounds']
        context = cv_args['context']

        # Shuffle data and generate equally-distanced cuts for validation split
        X = X.to_numpy()[shuffle_idxs]
        y = y.to_numpy()[shuffle_idxs]
        splits = np.linspace(0, len(X), K + 1).astype('int32')
        scaler = StandardScaler()

        # Split data into chunks
        chunks_X = []
        chunks_y = []
        i = 0
        j = 1
        while j != len(splits):
            chunks_X.append(X[splits[i]:splits[j]])
            chunks_y.append(y[splits[i]:splits[j]])
            i += 1
            j += 1

        # Cross Validation
        s1_fa_history = []
        s2_fa_history = []
        s1_conf_history = []
        s2_conf_history = []
        for k in range(K):
            print(f'CV Pair {k + 1}')
            
            # Grab test set
            X_test = chunks_X[k]
            y_test = chunks_y[k]

            # Build training set
            X_train = np.concatenate([x for idx, x in enumerate(chunks_X) if idx != k])
            y_train = np.concatenate([x for idx, x in enumerate(chunks_y) if idx != k])

            # Split training set into clients
            final_data = generate_data(X_train, y_train, seed = self.data_args['data seed'], tolerance = self.data_args['tolerance'], normalize = self.data_args['normalize'], client_num = self.data_args['client num'])
            if self.data_args['normalize']:
                X_test = scaler.fit_transform(X_test)

            # Introduce distance heterogeneity
            final_data['Client Distances'][self.data_args['distance clients']] += self.data_args['distance augments']

            # Format data and build federated system
            test = {'Data': X_test, 'Labels': y_test}
            self.fed = FederatedSystem(final_data['Client Data'], final_data['Client Labels'], final_data['Client Distances'])
            self.fed.SetTestData(test)

            # Run FedAvg and ConFeddi on data split
            print('FedAvg')
            w_fedavg, b_fedavg, fedavg_mse, fedavg_log = self.run_fedavg_test(rounds = rounds, frac_clients = 0.5)
            print(f'Time: {round(fedavg_log[-1], 2)}s')
            print('\nConFeddi')
            w, b, conf_mse, conf_log = self.run_confeddi_test(a, l, rounds = rounds, context = context)
            print(f'Time: {round(conf_log[-1], 2)}s', end = '\n\n')

            # Record scores
            s1_fedavg, s2_fedavg = self.average_error((w_fedavg, b_fedavg))
            s1_fa_history.append(s1_fedavg)
            s2_fa_history.append(s2_fedavg)
            s1, s2 = self.average_error((w, b))
            s1_conf_history.append(s1)
            s2_conf_history.append(s2)

            print(f'FedAvg Avg Test Error: {s1_fedavg}')
            print(f'ConFeddi Avg Test Error: {s1}')
            print(f'FedAvg MSE: {s2_fedavg}')
            print(f'ConFeddi MSE: {s2}', end = '\n\n')

            # Save data of best model based on score
            if k == 0:
                best_w = w
                best_b = b
                best_s1 = s1
                best_s2 = s2
                curr_fedavg_mse = fedavg_mse
                curr_conf_mse = conf_mse
                curr_fedavg_log = fedavg_log
                curr_conf_log = conf_log
                pkg = {'Clients': final_data, 'Test': test}
            elif (score == 1) & (s1 < best_s1):
                best_s1 = s1
                best_w = w
                best_b = b
                curr_fedavg_mse = fedavg_mse
                curr_conf_mse = conf_mse
                curr_fedavg_log = fedavg_log
                curr_conf_log = conf_log
                pkg = {'Clients': final_data, 'Test': test}
            elif (score == 2) & (s2 < best_s2):
                best_s2 = s2
                best_w = w
                best_b = b
                curr_fedavg_mse = fedavg_mse
                curr_conf_mse = conf_mse
                curr_fedavg_log = fedavg_log
                curr_conf_log = conf_log
                pkg = {'Clients': final_data, 'Test': test}

        cv_data = {
            'parameters': (best_w, best_b),
            'mse': (curr_fedavg_mse, curr_conf_mse),
            'log': (curr_fedavg_log, curr_conf_log),
            'data': pkg,
            's1 score': (np.array(s1_fa_history).mean(), np.array(s1_conf_history).mean()),
            's2 score': (np.array(s2_fa_history).mean(), np.array(s2_conf_history).mean())
        }

        return cv_data

    def scheme3_cv(self, X, y, shuffle_idxs, cv_args, r, c):
        # Performs cross validation for scheme 3 data split

        tf.keras.utils.set_random_seed(self.data_args['data seed'])

        # Unpack cv args
        K = cv_args['K']
        score = cv_args['score']
        a = cv_args['a']
        l = cv_args['l']
        rounds = cv_args['rounds']
        context = cv_args['context']

        scaler = StandardScaler()

        # Defining box corners
        x_min = X['GroundTruthPositionX[m]'].min()
        x_max = X['GroundTruthPositionX[m]'].max()
        y_min = X['GroundTruthPositionY[m]'].min()
        y_max = X['GroundTruthPositionY[m]'].max()

        # Defining split boundaries
        x_range = x_max - x_min
        x_block = x_range / c
        y_range = y_max - y_min
        y_block = y_range / r

        # Defining slice partitions
        x_cuts, y_cuts = [], []
        for i in range(1, c):
            x_cuts.append(x_min + i * x_block)
        for i in range(1, r):
            y_cuts.append(y_min + i * y_block)
        x_cuts.insert(0, x_min - 0.5)
        x_cuts.append(x_max + 0.5)
        y_cuts.insert(0, y_min - 0.5)
        y_cuts.append(y_max + 0.5)

        # Cross Validation
        GTPX = X['GroundTruthPositionX[m]']
        GTPY = X['GroundTruthPositionY[m]']
        X = X.to_numpy()[shuffle_idxs]
        y = y.to_numpy()[shuffle_idxs]
        s1_fa_history = []
        s2_fa_history = []
        s1_conf_history = []
        s2_conf_history = []
        for k in range(K):
            print(f'CV Pair {k + 1}')
            final_data = {'Client Data': [], 'Client Labels': [], 'Client Distances': []}
            X_test = []
            Y_test = []
            for i in range(len(y_cuts)):
                for j in range(len(x_cuts)):
                    # Conditions for reaching edge of grid
                    jump_idx1 = 0 if j == c else 1
                    jump_idx2 = 0 if i == r else 1
                    if jump_idx1 == 0 or jump_idx2 == 0: continue

                    # Conditions defining a block and block splits
                    condition = (x_cuts[j] < GTPX) & (GTPX < x_cuts[j + jump_idx1]) & (y_cuts[i] < GTPY) & (GTPY < y_cuts[i + jump_idx2])
                    curr_data = X[condition]
                    curr_labs = y[condition]
                    splits = np.linspace(0, len(curr_data), K + 1).astype('int32')

                    # Splitting the block
                    chunks_X = []
                    chunks_y = []
                    a = 0
                    b = 1
                    while b != len(splits):
                        chunks_X.append(curr_data[splits[a]:splits[b]])
                        chunks_y.append(curr_labs[splits[a]:splits[b]])
                        a += 1
                        b += 1

                    # Grab test set of current block
                    x_test = chunks_X[k]
                    y_test = chunks_y[k]
                    
                    # Build test set of current block
                    x_train = np.concatenate([x for idx, x in enumerate(chunks_X) if idx != k])
                    y_train = np.concatenate([x for idx, x in enumerate(chunks_y) if idx != k])

                    if self.data_args['normalize']:
                        x_train = scaler.fit_transform(x_train)

                    # Add to global sets
                    final_data['Client Data'].append(x_train)
                    final_data['Client Labels'].append(y_train)
                    X_test.append(x_test)
                    Y_test.append(y_test)

            # Build full test set
            X_test = np.concatenate([x for x in X_test])
            y_test = np.concatenate([y for y in Y_test])

            if self.data_args['normalize']:
                X_test = scaler.fit_transform(X_test)

            # Simulate client distances from server
            final_data['Client Distances'] = np.random.rand(len(final_data['Client Data'])) / 2
            final_data['Client Distances'][self.data_args['distance clients']] += self.data_args['distance augments']

            # Build FL system
            test = {'Data': X_test, 'Labels': y_test}
            self.fed = FederatedSystem(final_data['Client Data'], final_data['Client Labels'], final_data['Client Distances'])
            self.fed.SetTestData(test)

            # Run FedAvg and ConFeddi models
            print('FedAvg')
            w_fedavg, b_fedavg, fedavg_mse, fedavg_log = self.run_fedavg_test(rounds = rounds, frac_clients = (5 / (r * c)))
            print(f'Time: {round(fedavg_log[-1], 2)}s')
            print('\nConFeddi')
            w, b, conf_mse, conf_log = self.run_confeddi_test(a, l, rounds = rounds, context = context)
            print(f'Time: {round(conf_log[-1], 2)}s', end = '\n\n')

            # Compute scores
            s1_fedavg, s2_fedavg = self.average_error((w_fedavg, b_fedavg))
            s1_fa_history.append(s1_fedavg)
            s2_fa_history.append(s2_fedavg)
            s1, s2 = self.average_error((w, b))
            s1_conf_history.append(s1)
            s2_conf_history.append(s2)

            print(f'FedAvg Avg Test Error: {s1_fedavg}')
            print(f'ConFeddi Avg Test Error: {s1}')
            print(f'FedAvg MSE: {s2_fedavg}')
            print(f'ConFeddi MSE: {s2}', end = '\n\n')

            # Record best model data depending on score
            if k == 0:
                best_w = w
                best_b = b
                best_s1 = s1
                best_s2 = s2
                curr_fedavg_mse = fedavg_mse
                curr_conf_mse = conf_mse
                curr_fedavg_log = fedavg_log
                curr_conf_log = conf_log
                pkg = {'Clients': final_data, 'Test': test}
            elif (score == 1) & (s1 < best_s1):
                best_s1 = s1
                best_w = w
                best_b = b
                curr_fedavg_mse = fedavg_mse
                curr_conf_mse = conf_mse
                curr_fedavg_log = fedavg_log
                curr_conf_log = conf_log
                pkg = {'Clients': final_data, 'Test': test}
            elif (score == 2) & (s2 < best_s2):
                best_s2 = s2
                best_w = w
                best_b = b
                curr_fedavg_mse = fedavg_mse
                curr_conf_mse = conf_mse
                curr_fedavg_log = fedavg_log
                curr_conf_log = conf_log
                pkg = {'Clients': final_data, 'Test': test}

        cv_data = {
            'parameters': (best_w, best_b),
            'mse': (curr_fedavg_mse, curr_conf_mse),
            'log': (curr_fedavg_log, curr_conf_log),
            'data': pkg,
            's1 score': (np.array(s1_fa_history).mean(), np.array(s1_conf_history).mean()),
            's2 score': (np.array(s2_fa_history).mean(), np.array(s2_conf_history).mean())
        }

        return cv_data

    def cross_validation(self, K: int, score: int, a: float, l: float, rounds: int = 50, context: list[int] = [0, 1, 2, 3], args: Any = None) -> dict:
        """
        Wrapper cross validation function for scheme cross validations
        ----------------
        `score` refers to which score criteria will be used to select best model.
        `score == 1` refers to average test error.
        `score == 2` refers to test loss. 
        """

        tf.keras.utils.set_random_seed(self.data_args['data seed'])
        X = self.StrategyHandler.X.copy()
        y = self.StrategyHandler.y.copy()

        #Shuffle dataset
        shuffle_idxs = np.arange(len(X))
        np.random.shuffle(shuffle_idxs)

        # Pack arguments
        cv_args = {
            'K': K,
            'score': score,
            'a': a,
            'l': l,
            'rounds': rounds,
            'context': context
        }
        
        if self.scheme == 1:
           return self.scheme1_cv(X, y, shuffle_idxs, cv_args)
 
        if self.scheme == 3:
            return self.scheme3_cv(X, y, shuffle_idxs, cv_args, args[0], args[1])
                
    def plot_error(self, pairs: list[tuple], colors: list[str], labels: list[str], ylim: tuple[float]) -> None:
        """
        Plots test error curves (in one graph) with any number of test curves.
        """

        for i, p in enumerate(pairs):
            plt.plot(p[0], p[1], color = colors[i], label = labels[i], marker = 'o')

        plt.title('Test MSE')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.ylim(ylim[0], ylim[1])
        plt.grid()
        plt.legend()
        plt.show()

    def average_error(self, parameters: tuple[list[np.array]]) -> tuple[float]:
        """
        Computes the average error of a model on test set, and computes the loss on the test set.
        """

        # Compute average error on test set
        tf.keras.utils.set_random_seed(self.model_seed)
        model = self.fed.generate_model(parameters[0], parameters[1])
        pred = model.predict(self.fed.test_data['Data'], verbose = 0)
        ytest = self.fed.test_data['Labels']
        diff = np.absolute(pred - ytest)
        avg_error = (np.sum(diff) / len(ytest))

        # Compute loss on test set
        model.compile(optimizer = 'adam', loss = 'mse')
        loss = model.evaluate(self.fed.test_data['Data'], self.fed.test_data['Labels'], verbose = 0, use_multiprocessing = True)
        return avg_error, loss
