from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
import os
import joblib
import helper_functions
import plotting
import numpy as np
import pandas as pd
import ray
from ax.service.managed_loop import optimize
from sklearn.model_selection import cross_val_score


def standardize_values(train_df, test_df=None):
    """Standardizes the supplied training and test sets.

    Parameters
    ----------
    train_df: pandas.core.frame.DataFrame, training set input
    test_df: pandas.core.frame.DataFrame, test set input

    Returns
    -------
    If one dataframe is provided then returns its standardized version otherwise a list containing the standardized training and test dataframes in that order.
    """

    scaler = StandardScaler()
    scaler.fit(train_df)
    train_df[train_df.columns] = scaler.transform(train_df)

    if test_df is not None:
        test_df[test_df.columns] = scaler.transform(test_df)
        return [train_df, test_df]
    else:
        return train_df


def get_error_metrics_on_set(estimator, set_X, set_Y):
    """Calculates the MAE, MSE, RMSE, R2 for the given (training/test) set.  

    Parameters
    ----------
    estimator: sklearn.base.BaseEstimator, an object that inherits from 'BaseEstimator'
    set_X: pandas.core.frame.DataFrame, set input
    set_Y: pandas.core.frame.DataFrame, set output

    Returns
    -------
    A dict where each metric is key and value its value.
    """

    predictions = estimator.predict(set_X)
    MAE = mean_absolute_error(set_Y, predictions)
    MSE = mean_squared_error(set_Y, predictions)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(set_Y, predictions)

    return {'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE, 'R2': R2}


def hyperparameter_tuning(columns_to_exclude, model_type, location_type, location, output, aggregated_data_path):
    """Performs hyperparameter tuning using Bayesian optimization and returns the best parameters.  

    Parameters
    ----------
    columns_to_exclude: list, column names which will not be used in training
    model_type: str, one of 'local' 'regional' 'global'
    location_type: str, one of 'known' 'unknown'
    location: str, one of 'Clim1' 'Clim2' 'Clim3' 'Clim4' 'Clim5' 'Clim6' 'Clim7' 'Clim8' 
    output: str, name of the target variable column
    aggregated_data_path: str, the path where the results of the preprocessing are saved

    Returns
    -------
    Returns a dict where each key is a parameter of the estimator and value the optimized value.
    """

    trainingDF = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(
        aggregated_data_path + '/' + model_type + '/' + location_type + '/train/' + location + '.csv', "*.csv")))
    trainingDF = trainingDF.drop(columns_to_exclude, axis=1)
    X_training_standardizedDF = standardize_values(
        trainingDF.drop(output, axis=1))
    Y_trainingDF = trainingDF[output]

    # the objective function which will be used by Bayesian optimization
    def objective(parameterization):
        return np.mean(np.abs(cross_val_score(RandomForestRegressor(**parameterization), X_training_standardizedDF, Y_trainingDF, cv=3, scoring='r2')))

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "n_estimators", "type": "range", "bounds": [
                10, 15], "log_scale": False, "value_type": 'int'},
            {"name": "max_depth", "type": "range", "bounds": [
                2, 3], "log_scale": False, "value_type": 'int'},
            {"name": "n_jobs", "type": "fixed", "value": 35}
        ],
        evaluation_function=objective,
        minimize=False,
        total_trials=2,
        arms_per_trial=1)

    print('Best parameters for ' + model_type + ', ' +
          location_type + ', ' + location + ' :', best_parameters)

    return best_parameters


def train_and_validate_model(hyperparameters, model_type, location_type, location, test_location_testDF, Y_test_location_testDF, columns_to_exclude, output, aggregated_data_path):
    """Trains a model using Random Forest, validates it on training and test sets, and makes plots for both cases.

    Parameters
    ----------
    hyperparameters: dict, contains the tuned parameters for the estimator, each key is a name of a parameter
    model_type: str, one of 'local' 'regional' 'global'
    location_type: str, one of 'known' 'unknown'
    location: str, one of 'Clim1' 'Clim2' 'Clim3' 'Clim4' 'Clim5' 'Clim6' 'Clim7' 'Clim8' 
    test_location_testDF: pandas.core.frame.DataFrame, the test data for this location including (input and output)
    Y_test_location_testDF: pandas.core.frame.DataFrame, the output of the test set for this location
    columns_to_exclude: list, column names which will not be used in training
    output: str, name of the target variable column
    aggregated_data_path: str, the path where the results of the preprocessing are saved

    Returns
    -------
    A list with the trained model, training metrics, test metrics, test set predictions and locations included in the training of the model in that order.
    list[sklearn.ensemble._forest.RandomForestRegressor, dict, dict, numpy.ndarray, list]
    """

    trainingDF = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(
        aggregated_data_path + '/' + model_type + '/' + location_type + '/train/' + location + '.csv', "*.csv")))

    # to be used for verification purposes
    locations = trainingDF['Weather'].unique()

    trainingDF = trainingDF.drop(columns_to_exclude, axis=1)
    # we standardize only the input not the output values
    X_training_standardizedDF, X_test_location_standardized_testDF = standardize_values(
        trainingDF.drop(output, axis=1), test_location_testDF.drop(output, axis=1))
    Y_trainingDF = trainingDF[output]

    model = RandomForestRegressor(**hyperparameters)
    model.fit(X_training_standardizedDF, Y_trainingDF)

    training_results = get_error_metrics_on_set(
        model, X_training_standardizedDF, Y_trainingDF)
    test_results = get_error_metrics_on_set(
        model, X_test_location_standardized_testDF, Y_test_location_testDF)
    model_predictions_testing = model.predict(
        X_test_location_standardized_testDF)  # on test set

    plotting.make_jointplot(pd.DataFrame({'Predicted_'+output: model.predict(
        X_training_standardizedDF), output: Y_trainingDF}), 'train', output, model_type, location)
    plotting.make_jointplot(pd.DataFrame({'Predicted_'+output: model.predict(
        X_test_location_standardized_testDF), output: Y_test_location_testDF}), 'test', output, model_type, location)

    return [model, training_results, test_results, model_predictions_testing, locations]


@ray.remote
def process_for_location(location, location_type, output, columns_to_exclude, aggregated_data_path):
    """Convenience function around 'train_and_validate_model' to do it for all 3 models and store the results for later use.

    Parameters
    ----------
    location: str, one of 'Clim1' 'Clim2' 'Clim3' 'Clim4' 'Clim5' 'Clim6' 'Clim7' 'Clim8' 
    location_type: str, one of 'known' 'unknown'
    output: str, name of the target variable column
    columns_to_exclude: list, column names which will not be used in training
    aggregated_data_path: str, the path where the results of the preprocessing are saved

    Returns
    -------
    A list with the results of the validation for the 3 models, their predictions, and the trained models.
    list[dict, dict, dict]
    """

    # initialize result structures
    trained_models_dict = {}
    error_metrics_dict = {'training': {}, 'testing': {}}
    model_predictions_dict = {}
    model_predictionsDF_testing = pd.DataFrame()

    # read test df once
    test_location_testDF_original = pd.concat(pd.read_csv(f) for f in glob.glob(
        os.path.join(aggregated_data_path + '/location/test/' + location + '.csv', "*.csv")))
    test_location_testDF = test_location_testDF_original.drop(
        columns_to_exclude, axis=1)
    # ground truth for the specific location, this doesn't change between the global, regional, local models
    Y_test_location_testDF = test_location_testDF[output]

    # global
    global_model_hyperparameters = hyperparameter_tuning(
        columns_to_exclude, 'global', location_type, location, output, aggregated_data_path)
    global_model, global_training_results, global_test_results, global_model_predictions_testing, global_locations = train_and_validate_model(
        global_model_hyperparameters, 'global', location_type, location, test_location_testDF, Y_test_location_testDF, columns_to_exclude, output, aggregated_data_path)
    # regional
    regional_model_hyperparameters = hyperparameter_tuning(
        columns_to_exclude, 'regional', location_type, location, output, aggregated_data_path)
    regional_model, regional_training_results, regional_test_results, regional_model_predictions_testing, regional_locations = train_and_validate_model(
        regional_model_hyperparameters, 'regional', location_type, location, test_location_testDF, Y_test_location_testDF, columns_to_exclude, output, aggregated_data_path)
    # local
    local_model_hyperparameters = hyperparameter_tuning(
        columns_to_exclude, 'local', location_type, location, output, aggregated_data_path)
    local_model, local_training_results, local_test_results, local_model_predictions_testing, local_locations = train_and_validate_model(
        local_model_hyperparameters, 'local', location_type, location, test_location_testDF, Y_test_location_testDF, columns_to_exclude, output, aggregated_data_path)

    error_metrics_dict['training'][location +
                                   '_global'] = global_training_results
    error_metrics_dict['testing'][location + '_global'] = global_test_results
    error_metrics_dict['training'][location +
                                   '_regional'] = regional_training_results
    error_metrics_dict['testing'][location +
                                  '_regional'] = regional_test_results
    error_metrics_dict['training'][location +
                                   '_local'] = local_training_results
    error_metrics_dict['testing'][location + '_local'] = local_test_results

    trained_models_dict[location] = {'global_model': global_model}
    trained_models_dict[location]['regional_model'] = regional_model
    trained_models_dict[location]['local_model'] = local_model

    # this helps when we add columns to model_predictionsDF_testing to not get 'cannot reindex from a duplicate axis'
    test_location_testDF_original = test_location_testDF_original.reset_index()

    model_predictionsDF_testing['global_model'] = global_model_predictions_testing
    model_predictionsDF_testing['regional_model'] = regional_model_predictions_testing
    model_predictionsDF_testing['local_model'] = local_model_predictions_testing
    model_predictionsDF_testing['target_var'] = test_location_testDF_original['target_var']
    model_predictionsDF_testing['Year'] = test_location_testDF_original['Year']
    model_predictionsDF_testing['SoilWater'] = test_location_testDF_original['SoilWater']
    model_predictionsDF_testing['SoilFertility'] = test_location_testDF_original['SoilFertility']
    model_predictionsDF_testing['Irrigation'] = test_location_testDF_original['Irrigation']
    model_predictionsDF_testing['FertMonth'] = test_location_testDF_original['FertMonth']
    model_predictionsDF_testing['FertDay'] = test_location_testDF_original['FertDay']
    model_predictionsDF_testing['FertRate'] = test_location_testDF_original['FertRate']
    model_predictions_dict[location] = model_predictionsDF_testing

    print('location: ' + location + ', global model [' + ' '.join(sorted(global_locations)) + ']' + ', regional [' + ' '.join(
        regional_locations) + ']' + ', local [' + ' '.join(sorted(local_locations)) + ']' + ' --> done')

    return [error_metrics_dict, model_predictions_dict, trained_models_dict]


def ml_pipeline(scenario, location_type, output, aggregated_data_path):
    """Orchestrates the models' training and archives the results. Initializes ray, send the computational burden for each testing location in a different process, gets the results and saves them.

    Parameters
    ----------
    scenario: str, the name for this set of trained models and validations, used to name the folder where the results are saved
    location_type: str, one of 'known' 'unknown'
    output: str, name of the target variable column
    aggregated_data_path: str, the path where the results of the preprocessing are saved

    Returns
    -------
    Returns None. 
    """

    ray.init(num_cpus=10)

    columns_to_exclude = ['Weather', 'Year', 'FertDay']
    locations = ['Clim1', 'Clim2', 'Clim3',
                 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8']

    # create worker processes for the models' training in each location
    futures = [process_for_location.remote(
        location, location_type, output, columns_to_exclude, aggregated_data_path) for location in locations]
    # block until all the processes have finished
    results = ray.get(futures)
    # results [
    #            [error_metrics_dict, model_predictions_dict, trained_models_dict],
    #            ....,
    #            ....
    #            ]

    ray.shutdown()

    print('Combining results')
    # combine results
    for result in results[1:]:
        helper_functions.merge_nested_dict(results[0][0], result[0])
        helper_functions.merge_nested_dict(results[0][1], result[1])
        helper_functions.merge_nested_dict(results[0][2], result[2])

    error_metrics_dict = results[0][0]
    model_predictions_dict_testing = results[0][1]
    trained_models_dict = results[0][2]

    # -------------------------Save results----------------------------------------------------------------------------------------------
    # save error metrics
    print('Saving error metrics')
    helper_functions.results_to_excel(
        error_metrics_dict, 'error_metrics_'+scenario+'.xlsx')
    # save predictions
    print('Saving predictions')
    for location_name in model_predictions_dict_testing:
        model_predictions_dict_testing[location_name].to_csv(
            'predictions_testing_'+location_name+'.csv', index=False)
    # save models
    print('Saving trained models')
    for location_key in trained_models_dict:
        for model_type_key in trained_models_dict[location_key]:
            joblib.dump(
                trained_models_dict[location_key][model_type_key], location_key+'_'+model_type_key+'.model')
