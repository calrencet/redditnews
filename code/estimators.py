import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

class EstimatorChecker:

    def __init__(self, estimators, params):
        """
        Initialize an instance of the class.
        Inputs: `models`: The dictionary of estimators to use (e.g. {'etc': ExtraTreesClassifier(), 'rfc': RandomForestClassifier()})
        `params`: The dictionary of parameters for the estimators (e.g. {'etc': {'n_estimators': [5,10,20]})
        Outputs: An instance of the EstimatorChecker class with instance variables `estimators`, `params`, `keys` and `grid_searches`.
        """
        # Checks if all estimators given have been given parameters by matching the sets of keys
        if not set(estimators.keys()).issubset(set(params.keys())):
            # Creates a list of estimators which are missing parameters and informs user
            missing_params = list(set(estimators.keys()) - set(params.keys()))
            raise ValueError(f"Some estimators are missing parameters: {missing_params}")
        # Set the estimators to an instance variable
        self.estimators = estimators
        
        # Set the parameters to an instance variable
        self.params = params
        
        # Set the keys to an instance variable
        self.keys = estimators.keys()
        
        # Starts an empty dict of grid_searches
        self.grid_searches = {}

    def fit(self, X, y, cv=5, n_jobs=-1, verbose=1, scoring=None, refit=False):
        """
        Method to fit the datasets `X` and `y` through a GridSearchCV for each estimator in the instance.
        Parameters `cv`, `n_jobs`, `verbose`, `scoring`, and `refit` can be changed by passing the setting you want.
        Output: Updates the `self.grid_searches` dict with the fitted gs for each estimator."""

        # Iterate through the keys to fit the data through GridSearch for each Estimator
        for key in self.keys:
            print(f"Running GridSearchCV for {key}.")
            estimator = self.estimators[key]
            params = self.params[key]
            # Instantiate GridSearchCV with the estimator and params, returning train_score
            gs = GridSearchCV(estimator, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        """
        Method to create the summary of scores each estimator and parameter for the estimators that were put through GridSearchCV in the `fit` method.
        Input: `sort_by`: Column by which you want to sort the DataFrame
        Output: DataFrame containing the scores for the estimators, sorted according to the `sort_by` method given.
        """
        def row(key, scores, params):
            """
            Adds the scores to a dictionary for the estimator and outputs it into a pandas series
            """
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})
        
        # Instantiate an empty list for rows
        rows = []
        for k in self.grid_searches:
        # Iterates through the estimators in the grid_searches dictionary
            print(k)
            # Sets the params of the estimator for the grid_search
            params = self.grid_searches[k].cv_results_['params']
            
            # Instantiate an empty list for scores
            scores = []
            
            # Iterates through the cross_val number of the estimator and creates a key to append the test score results to the scores list
            for index in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(index)
                r = self.grid_searches[k].cv_results_[key]   
                
                # Append the result in the scores list as a single column
                scores.append(r.reshape(len(params),1))

            # Stack the scores array column-wise
            all_scores = np.hstack(scores)
            
            # Iterate through the a zip of the params and all_scores list to run the row function and create a dictionary containing all the estimators, parameters, and results, and appends them to the rows list.
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))
        
        # Creates a DataFrame using the rows list, transposes it, and sorts it according to the `sort_by` column.
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        
        # Sets the columns to be shown
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        # Returns the DataFrame filtered to the columns
        return df[columns]