# Intro To Machine Learning CW2: Neural Networks

This repo contains the model, training scripts and hyper-parameter search we have performed in this coursework.

#### Relevant packages used

To install relevant packages used, simply run

`pip3 install scikit-learn torch pandas pickle numpy`

#### To train and see the result, simply run

`python3 part2_house_value_regression.py` (Using the existing `./housing.csv` dataset)

or

`python3 part2_house_value_regression.py --dataset <path to other dataset>`

The training will be performed with 1000 epochs on batch size 512, with loss function being `nn.MSELoss()` and learning rate of 10^-3. To change these parameters, go to line 42 and change the default setting and then run the command above.

#### To use existing model to make prediction, simply run

`python3 part2_house_value_regression.py --loadmodel <path to model>`

This will load a trained/existing model and use it to make predictions on test dataset. (no training happens)

#### To run hyper-parameter tuning, simply run

`python3 part2_house_value_regression.py --hptuning`

This will start `GridSearch`ing the best hyperparameters and save the searching result into `tuning.pkl` (pandas Data Frame) and `tuning.csv`. To adjust the parameters you want to search, simply changing the `parameters` variable in function `RegressorHyperParameterSearch` and add corresponding variables to the `RegressionEstimator`, and then change the constructor(`__init__`) of the `Regressor` class.
