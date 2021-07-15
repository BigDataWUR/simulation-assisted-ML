import os
import pandas as pd


def flatten(xs):
    """Flattens an arbitrarily nested list. Taken from https://stackoverflow.com/a/4590652.

    Parameters
    ----------
    xs: list, a list containing nested lists

    Returns
    -------
    Returns a flattened list.
    """
    res = []

    def loop(ys):
        for i in ys:
            if isinstance(i, list):
                loop(i)
            else:
                res.append(i)
    loop(xs)
    return res


def create_path(path):
    """Creates a path of multiple levels if it exist already, otherwise it prints that the path exists.

    Parameters
    ----------
    path: str, a path which may contain multiple levels

    Returns
    -------
    Returns None.
    """
    
    try:
        os.makedirs(path)
    except:
        print(path + ' exists')


def results_to_excel(validation_results, filename):
    """Converts a dataframe of validation results to excel and saves it.

    Parameters
    ----------
    validation_results: dict, 
    filename: str, the name of the excel file to be created

    Returns
    -------
    Returns None.
    """

    train_result_df = pd.DataFrame(validation_results['training'])
    test_result_df = pd.DataFrame(validation_results['testing'])
    with pd.ExcelWriter(filename) as writer:
        train_result_df.to_excel(writer, sheet_name='training')
        test_result_df.to_excel(writer, sheet_name='testing')


def merge_nested_dict(d1, d2):
    """Merges the supplied dictionaries, in place, on 'd1'.

    Parameters
    ----------
    d1: dict
    d2: dict 

    Returns
    -------
    Returns None.
    """

    for k in d2:
        if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
            merge_nested_dict(d1[k], d2[k])
        else:
            d1[k] = d2[k]


def combine_known_unknown_predictions(ml_results_path, scenario_name):
    """Combines the predictions for all models, model types, location types and simulation parameters into a single dataframe.

    Parameters
    ----------
    ml_results_path: str, the path where the results are saved
    scenario_name: str, this along with 'predictions_base_path' will point to the folder where the predictions are

    Returns
    -------
    Returns a pandas.core.frame.DataFrame.
    """

    totalDF = pd.DataFrame()
    for known_unknown in ['known', 'unknown']:
        for location in ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8']:

            predictions_path = ml_results_path + '/' + scenario_name + '/' + \
                known_unknown + '/predictions_testing_' + location + '.csv'
            predictionsDF = pd.read_csv(predictions_path)
            for model in ['local', 'regional', 'global']:
                df = pd.DataFrame()
                df['Year'] = predictionsDF['Year']
                df['FertMonth'] = predictionsDF['FertMonth']
                df['target_var'] = predictionsDF['target_var']
                df['NRR'] = predictionsDF[model + '_model']
                df['Residual'] = (df['target_var'] - df['NRR']).abs()
                df['Metamodel type'] = model
                df['Test type'] = known_unknown
                df['Location'] = location
                df['Irrigation'] = predictionsDF['Irrigation']

                totalDF = totalDF.append(df, ignore_index=True)

    totalDF['Location'] = totalDF['Location'].map({'Clim1': 'Waiotu', 'Clim2': 'Ruakura', 'Clim3': 'Wairoa',
                                                  'Clim4': 'Marton', 'Clim5': 'Mahana', 'Clim6': 'Kokatahi', 'Clim7': 'Lincoln', 'Clim8': 'Wyndham'})

    return totalDF
