# ConFeddi Regression
**ConFeddi** is a **federated learning** algorithm designed to handle settings with **heterogeneous data** using a **contextual multi-armed bandit**[^1]. Heterogeneous data can take on many forms, but my implementaion focuses on three common heterogeneities in federated learning:
- Non-IIDness of local datasets
- Spatial Distribution of clients
- Noisy Data

Below is a summary of the work I've done, and poster presented at the UCR NC4 Review in Sept. 2022. I refactored ConFeddi for a regressive task on the [RTT dataset][dataset].
![summary.png](https://www.dropbox.com/s/bdmpo6azpu9gjdf/summary.png?dl=0&raw=1)

# Implementation
### Overview
I compare ConFeddi against [FedAvg][paper1], a standard federated learning algorithm. The `FederatedSystem()` class contains my implementation of `FedAvg()` and `ConFeddi()` as well as the fundamental methods for local client training and model aggregation. The ConFeddi algorithm contains the implementation of the selection criteria as a combination of `UCB_scores()` for scoring the worst and best clients for selection, and [C2UCB][paper2] which is implicitly written into `ConFeddi()` for cost-effective updates of `UCB_scores()` parameters. Model aggregation timestamps can be obtained using `Test.GetLog()` while the test loss history, or test loss of each model aggregation, can be obtained using `Test.test_loss()`.\
\
The `Test()` class provides an all-inclusive test suite for testing and comparing both FedAvg and ConFeddi[^2]. Several methods are provided to display relevant distribution data as well as displaying and comparing model performance. `Test.run_fedavg_test()` and `Test.run_confeddi_test()` train federated systems with the respective algorithm then return the model parameters, the test loss, and the log history. The test suite also provides gridsearch, ablation study, and cross validation methods with graphing utilities.\
\
Please refer to the code for full documentation. Public methods contain docstrings while helper or private methods contain small comments.

### Package Versions
I implemented ConFeddi using:
- Python 3.10.6
- Numpy 1.23.2
- Pandas 1.4.3
- Tensorflow 2.9.1
- Keras 2.9.0

### Usage
To run a standard FedAvg or ConFeddi test, shown below is a run setup.
```python
import numpy as np
import pandas as pd
import os
from test_class import Test

def main():
    dataset = pd.read_csv('RTT_data.csv')
    data_args = {
        'data seed': 3,                                 # seed used for all data splitting procedures
        'distance clients': [0, 2, 3, 6],               # select clients to add distance to for spatial heterogeneity
        'distance augments': [0.5, 0.5, 0.5, 0.5],      # how much to add to client distances (in kilometers)
        'tolerance': 5,                                 # minimum number of samples a client can have
        'exclude dtypes': 'object',                     # type of columns to remove (complex values in RTT dataset case)
        'drop labels': ['GroundTruthRange[m]'],         # feature(s) to drop for data
        'target labels': ['GroundTruthRange[m]'],       # feature(s) to predict
        'test size': 0.2,                               # global test set size
        'normalize': True,                              # whether to normalize client data locally, and global test set
        'client num': 10                                # number of clients to distribute data for
    }
    
    rounds = 50                                         # number of training rounds
    Mt = (np.ones(rounds) * 5).astype('int32')          # number of clients to select each round for ConFeddi
    model_seed = 50                                     # operations seed for tf
    test = Test(dataset, data_args, Mt, model_seed)     # instantiate test suite
    test.split(scheme = 1)                              # randomly sample data into 10 (default) clients
    
    w, b, fedavg_loss, fedavg_log = test.run_fedavg_test(rounds = rounds, frac_clients = 0.5)
    
    w, b, conf_loss, conf_log = test.run_confeddi_test(1000, 1, rounds = 50, context = [0])
    
    test.plot_error([(fedavg_log, fedavg_loss), (conf_log, conf_loss)], ['green', 'blue'], ['FedAvg MSE', 'ConFeddi MSE'], (0.1, 0.5))
    
if __name__ == '__main__':
    main()
```

Calling `test.plot_error()` produces the graph below:
![output.png](https://www.dropbox.com/s/18xk14oyatlgr3f/output.png?dl=0&raw=1)

Additionally, the setup for cross validation, similar to the setup used for **First Setup** and **Second Setup** in the summary poster, are provided in scheme1_CV.ipynb (random sampling) and scheme3a_CV.ipynb (grid sampling).

### Relevant Parameters
Two parameters unique to ConFeddi are `a` or `alpha`, and `l` or `reg_coeff`. 
- `a` or `alpha` represent **exploration strength**. A lower exploration strength decreases the rate at which the selection algorithm selects clients it hasn't selected before, while a higher exploration strength increases this rate. 
- `l` or `reg_coeff` acts as a client regularization coefficent, penalizing clients with very large coefficients or high impacts on the global model.

# Contact
Please send any questions, comments, or inquiries to jwall014@ucr.edu.


[^1]: Paper is still a work in progress. Contact jwall014@ucr.edu or cxian008@ucr.edu for more info. The arxiv link will be included once it is ready.

[^2]: As this build was designed for the RTT dataset, the `Test()` implementation caters specifically to this dataset and will not work with other datasets. This test suite implementation will be generalized in the future.

[dataset]: <https://www.researchgate.net/publication/329887019_A_Machine_Learning_Approach_for_Wi-Fi_RTT_Ranging>

[paper1]: <https://arxiv.org/abs/1602.05629>

[paper2]: <https://epubs.siam.org/doi/abs/10.1137/1.9781611973440.53>
