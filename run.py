import preprocessing_clover
import ML_clover
import helper_functions
import plotting
import os
import shutil


def main():

    # DEFINE PATHS
    simulations_path = '/mnt/guanabana/raid/home/pylia001/NZ/data/grassClover/FullMixedMethodsGrassCloverCSVs_16Files'
    weather_path = '/mnt/guanabana/raid/home/pylia001/NZ/data/WeatherDataAll.csv'
    nitrogen_path = '/mnt/guanabana/raid/home/pylia001/NZ/data/fixed_NRespAll.csv'

    scenario_name = 'NZ_study_trial_1'
    preprocessing_results_path = '/mnt/guanabana/raid/home/pylia001/NZ/data/grassClover/'
    preprocessing_results_path = preprocessing_results_path + scenario_name

    ml_results_path = '/mnt/guanabana/raid/home/pylia001/NZ/data/grassClover/ML_results'
    helper_functions.create_path(
        ml_results_path + '/' + scenario_name + '/' + 'known')
    helper_functions.create_path(
        ml_results_path + '/' + scenario_name + '/' + 'unknown')

    # PREPROCESSING
    preprocessing_clover.preprocessing_pipeline(False,
                                                [7],
                                                -28,
                                                -1,
                                                ['AboveGroundWt', 'NetGrowthWt', 'NetPotentialGrowthWt',
                                                    'SoilWater300', 'SoilTemp300', 'SoilTemp050', 'AboveGroundNConc'],
                                                True,
                                                True,
                                                simulations_path,
                                                weather_path,
                                                nitrogen_path,
                                                preprocessing_results_path)

    print('Preprocessing done')

    # ML
    for location_type in ['known', 'unknown']:
        ML_clover.ml_pipeline(scenario_name + '_' + location_type,
                              location_type,
                              'target_var',
                              preprocessing_results_path)

        # move .png .csv .xlsx .model files to another directory
        os.system('find . \( -name "*.png" -o -name "*.csv" -o -name "*.xlsx" -o -name "*.model" \) -exec mv "{}" ' +
                  ml_results_path + '/' + scenario_name + '/' + location_type + ' \;')

    print('ML done')

    # RESIDUAL PLOTTING
    combined_predictionsDF = helper_functions.combine_known_unknown_predictions(
        ml_results_path, scenario_name)

    for location_type in ['known', 'unknown']:
        for x_axis in ['FertMonth', 'Year']:
            plotting.make_boxplots(
                location_type, combined_predictionsDF, x_axis)
            for irrigation in [0, 1]:
                plotting.make_boxplots(
                    location_type, combined_predictionsDF, x_axis, irrigation)

        # move .png .csv .xlsx .model files to another directory
        os.system('find . -name "*.png" -exec mv "{}" ' + ml_results_path +
                  '/' + scenario_name + '/' + location_type + ' \;')

    print('Residual plotting done')

    # ARCHIVING
    shutil.make_archive(ml_results_path + '/' + scenario_name,
                        'zip', ml_results_path + '/' + scenario_name)

    # CLEANING UP
    os.system('rm -rf ' + ml_results_path + '/' + scenario_name)

    print('All tasks finished')


if __name__ == "__main__":
    main()
