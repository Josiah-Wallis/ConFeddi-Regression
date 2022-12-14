import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from distribute_data import generate_data
os.environ['PYTHONHASHSEED'] = str(50)

class RTTSplitStrategy():
    def __init__(self, dataset, data_args):
        # Unpacks necessary arguments for data split strategies

        self.data_seed = data_args['data seed']
        self.distance_clients = data_args['distance clients']
        self.distance_augments = data_args['distance augments']
        self.tolerance = data_args['tolerance']
        self.exclude_dtypes = data_args['exclude dtypes']
        self.drop_labels = data_args['drop labels']
        self.target_labels = data_args['target labels']
        self.test_size = data_args['test size']
        self.normalize = data_args['normalize']
        self.client_num = data_args['client num']

        # CONST, drop signal data (complex)
        self.dataset = dataset.select_dtypes(exclude = self.exclude_dtypes)
        self.X = self.dataset.drop(columns = self.drop_labels)
        self.y = self.dataset[self.target_labels]
        self.total_samples = self.dataset.shape[0]

        # Normalizer
        self.scaler = StandardScaler()

        # data after distribution among clients
        self.final_data = None

    def display_metadata(self):
        # see Test.display_metadata

        # Backwards computing total data
        total_x = 0
        total_y = 0
        for a, b in zip(self.final_data['Client Data'], self.final_data['Client Labels']):
            total_x += len(a)
            total_y += len(b)

        print(f'Number of samples: {self.total_samples}')
        print(f'Features per sample: {self.X_test.shape[1]}', end = '\n\n')

        print(f'Columns:')
        for i in self.dataset.drop(columns = self.drop_labels).columns[:-1]:
            print(f'{i}, ', end = '')
        print(self.dataset.drop(columns = self.drop_labels).columns[-1], end = '\n\n')
    
        print(f"Clients: {len(self.final_data['Client Data'])}")
        print(f'Total Client Training Samples: {total_x} ({total_x * 100/ self.total_samples:.2f}%)')
        print(f'Total Client Training Labels: {total_y}')
        print(f'Total Test Samples: {len(self.X_test)} ({len(self.X_test) * 100/ self.total_samples:.2f}%)')
        print(f'Total Test Labels: {len(self.y_test)}')

    def display_client_distribution(self):
        # see Test.display_client_distribution

        total_x = 0
        for x in self.final_data['Client Data']:
            total_x += len(x)

        print('Data Distribution')
        for i, x in enumerate(self.final_data['Client Data']):
            print(f'Client {i + 1}: {len(x) * 100 / total_x:.2f}%')
        print()

        print('Distance Distribution w.r.t. Max Distance')
        max_distance = self.final_data['Client Distances'].max()
        for i, x in enumerate(self.final_data['Client Distances']):
            print(f'Client {i + 1}: {x * 100 / max_distance:.2f}%')

    def random(self, args):
        # Randomly sample pairs of GTP and FTM responder pairs from same office venue

        # Format X and y
        X = self.X.to_numpy()
        y = self.y.to_numpy()

        # Split into model training/val set + global validation set not used for selection criteria
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.data_seed)


        # Distribute among 10 clients (default)
        self.final_data = generate_data(X_train, y_train, seed = self.data_seed, tolerance = self.tolerance, normalize = self.normalize, client_num = self.client_num)
        if self.normalize:
            self.X_test = self.scaler.fit_transform(self.X_test)

        # Introduce distance heterogeneity
        self.final_data['Client Distances'][self.distance_clients] += self.distance_augments

        # Format return
        return {
            'Split Data': self.final_data, 
            'Test': {'Data': self.X_test, 'Labels': self.y_test}
        }

    def correspondence(self, args):
        #Distribute samples associated with FTM responder X to client X    
           
        # Initialize containers
        self.final_data = {'Client Data': [], 'Client Labels': [], 'Client Distances': []}
        X_test = []
        Y_test = []

        # Split data s.t. Client X has data pairs associated with FTM Responder X
        for i in range(1, 13):
            condition = (self.X['AP_index'] == i)
            curr_data = self.X[condition].to_numpy()
            curr_labs = self.y[condition].to_numpy()
            
            x_train, x_test, y_train, y_test = train_test_split(curr_data, curr_labs, test_size = self.test_size, random_state = self.data_seed)
            self.final_data['Client Data'].append(x_train)
            self.final_data['Client Labels'].append(y_train)
            X_test.append(x_test)
            Y_test.append(y_test)

        # Format for federated system
        self.X_test = np.concatenate([x for x in X_test])
        self.y_test = np.concatenate([y for y in Y_test])

        # Introduce distance heterogeneity
        np.random.seed(self.data_seed)
        self.final_data['Client Distances'] = np.random.rand(len(self.final_data['Client Data'])) / 2
        self.final_data['Client Distances'][self.distance_clients] += self.distance_augments

        # Format return
        return {
            'Split Data': self.final_data, 
            'Test': {'Data': self.X_test, 'Labels': self.y_test}
        }
        
    def spatial(self, args):
        # Distribute samples based on equal partition of office venue w.r.t. FTM resonders

        # r x c grid to split office venue into
        r = args[0]
        c = args[1]

        # Initialize containers
        self.final_data = {'Client Data': [], 'Client Labels': [], 'Client Distances': []}
        X_test = []
        Y_test = []

        # Locate corners of grid
        x_min = self.X['GroundTruthPositionX[m]'].min()
        x_max = self.X['GroundTruthPositionX[m]'].max()
        y_min = self.X['GroundTruthPositionY[m]'].min()
        y_max = self.X['GroundTruthPositionY[m]'].max()

        # Define length/width of grid, and how large boxes are
        x_range = x_max - x_min
        x_block = x_range / c
        y_range = y_max - y_min
        y_block = y_range / r

        # Split grid and save cut indices
        x_cuts, y_cuts = [], []
        for i in range(1, c):
            x_cuts.append(x_min + i * x_block)
        for i in range(1, r):
            y_cuts.append(y_min + i * y_block)
        x_cuts.insert(0, x_min - 0.5)
        x_cuts.append(x_max + 0.5)
        y_cuts.insert(0, y_min - 0.5)
        y_cuts.append(y_max + 0.5)

        # Starting from the bottom left (column to column), section off grid blocks as clients
        GTPX = self.X['GroundTruthPositionX[m]']
        GTPY = self.X['GroundTruthPositionY[m]']
        for i, _ in enumerate(y_cuts):
            for j, _ in enumerate(x_cuts):
                # Determine when to stop iterating along columns
                jump_idx1 = 0 if j == c else 1
                jump_idx2 = 0 if i == r else 1
                if jump_idx1 == 0 or jump_idx2 == 0: continue

                # Grab data that's within a grid block
                condition = (x_cuts[j] < GTPX) & (GTPX < x_cuts[j + jump_idx1]) & (y_cuts[i] < GTPY) & (GTPY < y_cuts[i + jump_idx2])
                curr_data = self.X[condition].to_numpy()
                curr_labs = self.y[condition].to_numpy()
                
                # Split data into train/val/test split
                x_train, x_test, y_train, y_test = train_test_split(curr_data, curr_labs, test_size = self.test_size, random_state = self.data_seed)

                if self.normalize:
                    x_train = self.scaler.fit_transform(x_train)
                    x_test = self.scaler.fit_transform(x_test)

                self.final_data['Client Data'].append(x_train)
                self.final_data['Client Labels'].append(y_train)
                X_test.append(x_test)
                Y_test.append(y_test)

        # Format for federated system
        self.X_test = np.concatenate([x for x in X_test])
        self.y_test = np.concatenate([y for y in Y_test])

        # Introduce distance heterogeneity
        np.random.seed(self.data_seed)
        self.final_data['Client Distances'] = np.random.rand(len(self.final_data['Client Data'])) / 2
        self.final_data['Client Distances'][self.distance_clients] += self.distance_augments

        # Format return
        return {
            'Split Data': self.final_data, 
            'Test': {'Data': self.X_test, 'Labels': self.y_test}
        }