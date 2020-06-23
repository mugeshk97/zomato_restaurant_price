from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from ebrain_train.Logger import Log
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle


class Tuner:
    def __init___(self):
        self.logger = Log()

    def decision_tree(self, x, y, parameters):
        self.X = x
        self.y = y
        param_grid = {"criterion": ["mse", "friedman_mse", "mae"],
                      "splitter": ["best", "random"],
                      "max_features": ["auto", "sqrt", "log2"],
                      'max_depth': range(2, 16, 2),
                      'min_samples_split': range(2, 16, 2)}
        grid = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
        grid.fit(self.X, self.y)
        criterion = grid.best_params_['criterion']
        splitter = grid.best_params_['splitter']
        max_feature = grid.best_params_['max_features']
        max_depth = grid.best_params_['max_depth']
        min_samples_split = grid.best_params_['min_samples_split']
        self.dtree = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_features=max_feature,
                                           max_depth=max_depth, min_samples_split=min_samples_split)
        self.dtree.fit(self.X, self.y)
        if parameters == True:
            print(grid.best_params_)
        else:
            pass
        return self.dtree

    def random_forest(self, x, y, parameters):
        self.X = x
        self.y = y
        param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['mse', 'mae', ], "max_depth": range(2, 4, 1),
                      "max_features": ['auto', 'log2', 'sqrt']}

        grid = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
        grid.fit(self.X, self.y)
        criterion = grid.best_params_['criterion']
        max_feature = grid.best_params_['max_features']
        max_depth = grid.best_params_['max_depth']
        n_estimators = grid.best_params_['n_estimators']
        self.randomforest = RandomForestRegressor(criterion=criterion, max_features=max_feature, max_depth=max_depth,
                                                  n_estimators=n_estimators)
        self.randomforest.fit(self.X, self.y)
        if parameters == True:
            print(grid.best_params_)
        else:
            pass
        return self.randomforest

    def xg_boost(self, x, y, parameters):
        self.X = x
        self.y = y
        param_grid = {'learning_rate': [0.5, 0.1, 0.01, 0.001], 'max_depth': [3, 5, 10, 20],
                      'n_estimators': [10, 50, 100, 200]}
        grid = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror'), param_grid=param_grid, cv=5,
                            verbose=1, n_jobs=-1)
        grid.fit(self.X, self.y)
        learning_rate = grid.best_params_['learning_rate']
        max_depth = grid.best_params_['max_depth']
        n_estimators = grid.best_params_['n_estimators']
        self.xgb = XGBRegressor(objective='reg:squarederror', learning_rate=learning_rate, max_depth=max_depth,
                                n_estimators=n_estimators)
        self.xgb.fit(self.X, self.y)
        if parameters == True:
            print(grid.best_params_)
        else:
            pass
        return self.xgb


class Model_Selector(Tuner):
    def __init__(self):
        self.logger = Log()

    def best_model(self, X_train, X_test, y_train, y_test):
        log_file = open('training_logs/04-best_model.txt', 'a+')
        try:
            dtree = Tuner.decision_tree(self, X_train, y_train, parameters=False)
            dtree_prediction = dtree.predict(X_test)
            dtree_score = r2_score(y_test, dtree_prediction)
            self.logger.log(log_file, 'Training Finished - Decision Tree -%s' % str(dtree_score))
        except Exception as e:
            self.logger.log(log_file, 'Failed to Train Decision Tree %s' % e)
        try:
            rforest = Tuner.random_forest(self, X_train, y_train, parameters=False)
            rforest_prediction = rforest.predict(X_test)
            rforest_score = r2_score(y_test, rforest_prediction)
            self.logger.log(log_file, 'Training Finished - Random Forest -%s' % str(rforest_score))
        except Exception as e:
            self.logger.log(log_file, 'Failed to Train Random Forest %s' % e)
        try:
            xgboost = Tuner.xg_boost(self, X_train, y_train, parameters=False)
            xgboost_prediction = xgboost.predict(X_test)
            xgboost_score = r2_score(y_test, xgboost_prediction)
            self.logger.log(log_file, 'Training Finished - XGboost -%s' % str(xgboost_score))
        except Exception as e:
            self.logger.log(log_file, 'Failed to Train Xgboost %s' % e)
        try:
            if rforest_score > xgboost_score:
                with open('models/model-' + str(rforest_score) + '.pkl', 'wb') as file:
                    pickle.dump(rforest, file)
                self.logger.log(log_file, 'Model saved - RandomForestRegressor')
                log_file.write('-' * 150 + '\n')
                return rforest
            else:
                with open('models/model-' + str(xgboost_score) + '.pkl', 'wb') as file:
                    pickle.dump(xgboost, file)
                self.logger.log(log_file, 'Model saved - XGBRegressor')
                log_file.write('-' * 150 + '\n')
                return xgboost
        except Exception as e:
            self.logger.log(log_file, 'Exception %s' % e)
            log_file.write('-' * 150 + '\n')
