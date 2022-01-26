# Introduction to ML - Decision Tree Coursework

### Set Up

For this program to run, a working python 3 environment is needed. 

Run: 
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

### Running the code

To execute the code, simply run `python main.py [dataset] [is_pruned]`, where the `[dataset]` can either be `clean` or `noisy` and `[is_pruned]` can be `pruned` or `unpruned`. For instance

```
python main.py clean unpruned  // will run on clean dataset and print unpruned tree
python main.py clean pruned    // will run on clean dataset and print pruned tree
python main.py noisy unpruned  // will run on noisy dataset and print unpruned tree
python main.py noisy pruned    // will run on noisy dataset and print pruned tree
```

This will run a 10-fold cross validation of the decision tree, and output the result.

The above command prints out the visualisation for the tree generated for fold 0.