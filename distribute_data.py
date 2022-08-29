import numpy as np
from sklearn.preprocessing import StandardScaler

def check_tolerance(idxs, N, tolerance):
    start = 0
    for idx in idxs:
        if (idx - start) <= tolerance:
            return False
        start = idx

    if (N - start) < tolerance:
        return False
    else:
        return True

def validate_distribution(split_idxs, N, tolerance, client_num, seed):
    count = 0

    while True:
        if check_tolerance(split_idxs, N, tolerance):
            return split_idxs
        else:
            count += 1

            if count == 10000:
                print('The program is having trouble fitting the specified tolerance.\nPlease try a smaller tolerance. Exiting with error code -1...')
                return -1
            
            split_idxs = np.random.uniform(0, N, client_num - 1)
            split_idxs = np.sort(split_idxs).astype('int32')

def split_among_clients(X, y, split_idxs):
    clients_X = []
    clients_y = []
    start = 0
    for end in split_idxs:
        data = X[start : end]
        labels = y[start : end]

        clients_X.append(data)
        clients_y.append(labels)

        start = end

    data = X[start:]
    labels = y[start:]

    clients_X.append(data)
    clients_y.append(labels)

    return clients_X, clients_y


def generate_data(X, y, client_num = 10, tolerance = 1000, seed = 1):
    N = X.shape[0]

    np.random.seed(seed)
    split_idxs = np.random.uniform(0, N, client_num - 1)
    split_idxs = np.sort(split_idxs).astype('int32')
    split_idxs = validate_distribution(split_idxs, N, tolerance, client_num, seed)

    assert type(split_idxs) != int

    scaler = StandardScaler()
    clients_X, clients_y = split_among_clients(X, y, split_idxs)
    for i in range(len(clients_X)):
        clients_X[i] = scaler.fit_transform(clients_X[i])

    np.random.seed(seed)
    distances = np.random.rand(client_num) / 2

    return {'Client Data': clients_X, 'Client Labels': clients_y, 'Client Distances': distances}


