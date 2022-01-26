import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

import pickle
import numpy as np
from numpy.lib.arraypad import pad
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

import argparse

TRAINED_WEIGHT_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, output_size)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch = 1000, loss_fn = nn.MSELoss(), lr = 0.05, batch_size = 16):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super(Regressor, self).__init__()

        X, _ = self._preprocessor(x, training = True)
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.input_size = X.shape[1]
        self.output_size = 1
        self.model = NeuralNetwork(self.input_size, self.output_size).to(DEVICE)
        self.loss_fn = loss_fn.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Return preprocessed x and y, return None for y if it was None
        x = pd.DataFrame.fillna(x, method="pad")

        x['num_rooms'] = x['total_rooms'] / x["households"]
        x['num_bedrooms'] = x['total_bedrooms'] / x["households"]
        x['people_per_house'] = x['population'] / x["households"]
        x = x.drop(columns=['total_rooms', 'total_bedrooms', 'population', "households"])

        if training:
            # Numerical encoding for textual data
            self.lb = preprocessing.LabelBinarizer()
            self.lb.fit(x["ocean_proximity"])
            
            # Use mean and std to normalize test data
            self.mean = x.mean(numeric_only=True)
            self.std = x.std(numeric_only=True)


        # Appends the binary label encoding of the ocean proximity to the features and
        # drops the ocean_proximity column.
        ocean_proximities = self.lb.transform(x["ocean_proximity"])
        x = x.drop(columns=["ocean_proximity"])

        x = (x - self.mean) / self.std

        x = x.join(
            pd.DataFrame(ocean_proximities, columns=self.lb.classes_, index=x.index)
        )

        return x.values, (y.values if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget  

        train = data_utils.TensorDataset(torch.Tensor(X).to(DEVICE), torch.Tensor(Y).to(DEVICE))
        train_loader = data_utils.DataLoader(train, batch_size=self.batch_size, shuffle = True)
        size = len(train_loader.dataset)
        self.model.train()

        for i in range(self.nb_epoch):
            for batch, (batchX, batchY) in enumerate(train_loader):
                batchX = batchX.to(DEVICE)
                batchY = batchY.to(DEVICE)

                # Compute prediction error
                pred = self.model(batchX)
                loss = self.loss_fn(pred, batchY)


                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i % 10 == 0:
                print(f"epoch: {i} loss: {loss:>7f}")
                
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training = False) # Do not forget
        self.model.eval()
        np_pred = torch.from_numpy(X).float().to(DEVICE)
        np_result = self.model(np_pred).detach().cpu().numpy()

        return np_result



        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        
        self.model.eval()
        np_x = torch.from_numpy(X).float().to(DEVICE)
        np_y = torch.from_numpy(Y).float().to(DEVICE)

        result = self.model(np_x)
        loss_fn = nn.MSELoss().to(DEVICE)
        ret = loss_fn(np_y, result).item() 

        return 1 / np.sqrt(ret)


        #######################################################################           
        #                       ** END OF YOUR CODE **
        #######################################################################

def save_regressor(regressor): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(regressor, target)
    print("\nSaved model in part2_model.pickle\n")
    torch.save(regressor.model.state_dict(), TRAINED_WEIGHT_PATH)


def load_regressor(model = "part2_model.pickle"):
    """ 
    Utility function to load the trained regressor model in part2_model.pickle or other existing models.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open(model, 'rb') as target:
        trained_regressor = pickle.load(target)
    trained_model = NeuralNetwork(trained_regressor.input_size, trained_regressor.output_size)
    trained_model.load_state_dict(torch.load(TRAINED_WEIGHT_PATH))
    trained_regressor.model = trained_model
        
    print(f"\nLoaded model in {model}\n")
    
    return trained_regressor

class RegressionEstimator(BaseEstimator):
    def __init__(self, subestimator=None, loss_fn = "MSE", lr=0.05, batch_size=16):
        self.subestimator = subestimator
        self.loss_fn_map = {"MSE": nn.MSELoss(), "Mean": nn.L1Loss()}
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, x, y):
        self.regressor = Regressor(x, 1000,  self.loss_fn_map[self.loss_fn], self.lr, self.batch_size)
        self.regressor.fit(x, y)
    
    def predict(self, x):
        return self.regressor.predict(x)

    def score(self, x, y):
        return self.regressor.score(x, y)

    def get_params(self, deep=False):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"loss_fn": self.loss_fn, "batch_size": self.batch_size, "lr": self.lr}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def RegressorHyperParameterSearch(x, y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:training:
            # N
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    parameters = {
        'loss_fn': ["MSE", "Mean"],
        'batch_size': [64, 128, 256, 512, 1024], 
        'lr': [1e-2, 1e-3, 1e-4, 1e-5],
        # 'epoch': [25, 50, 100, 150, 500, 1000] # e.g. adding epoch as part of the hyper-parameter tuning
    }
    estimator = RegressionEstimator()
    clf = GridSearchCV(estimator, parameters)
    clf.fit(x, y)
    return clf.cv_results_# Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def example_main(args):
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv(args.dataset) 

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if args.hptuning:
        print("Performing hyper parameter search...")
        result = RegressorHyperParameterSearch(x_train, y_train)
        df = pd.DataFrame(data=result)
        print("Search completed. Results saved into tuning.pkl (pandas data frame) and tuning.csv (csv format)")
        df.to_pickle("tuning.pkl")
        df.to_csv("tuning.csv")
        print("Below is a printed version of the hyper parameter search result")
        print(df)
    else:
        regressor = Regressor(x_train)
        if args.loadmodel:
            print("Using pre-trained/existing model to make predictions...")
            regressor = load_regressor(args.loadmodel)
        else:
            print(f"Training the model with provided dataset at {args.dataset}...")
            regressor.fit(x_train, y_train)
            save_regressor(regressor)
            print("Training completed. Below is the prediction on test dataset:")
        
        print(regressor.predict(x_test))

        # Error
        print("Below is the regressor error:")
        error = regressor.score(x_test, y_test)
        print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='House Value Regression')
    parser.add_argument('--dataset', type=str, default='./housing.csv', help='Specify which dataset to use')
    parser.add_argument('--hptuning', action="store_true", help="Performing seaerch for optimal hyper-parameters when this flag is turned on")
    parser.add_argument('--loadmodel', type=str, default="", help="Loading existing model and use it to make predictions (no training)")
    args = parser.parse_args()
    example_main(args)
