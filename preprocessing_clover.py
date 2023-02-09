from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
import helper_functions
import location_match
import schemas


# PROCESS DATA

def combine_input_datasets(WeatherDataAllDF, simulationsDF, biophysicals):
    """Creates a dataframe with the simulations output and the corresponding weather for each simulation. Only the biophysical variables contained in 'biophysicals' are preserved from the simulations dataframe.

    Parameters
    ----------
    WeatherDataAllDF: pyspark.sql.dataframe.DataFrame, dataframe containing the weather data
    simulationsDF: pyspark.sql.dataframe.DataFrame, dataframe containing the simulations
    biophysicals: list, biophysical variables of the simulations

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame.
    """

    WeatherDataAllDF = WeatherDataAllDF.select(*[column for column in WeatherDataAllDF.columns if (column != 'Date' and column != 'Weather')],
                                               F.to_date(
                                                   F.col("Date"), "dd/MM/yyyy").alias('Date'),
                                               F.substring(F.col('Weather'), 0, 5).alias('Weather'))

    simulationsDF = simulationsDF.select(*['File', 'FertiliserApplied', 'HerbageCut'],
                                         *biophysicals,
                                         F.to_date(
                                             F.col("Date"), "dd/MMM/yyyy").alias('Date'),
                                         F.substring(F.col('File'), -5, 5).alias('Weather'),
                                         F.col('DaysRelative').cast("Int").alias('DaysRelative'))

    return simulationsDF.join(WeatherDataAllDF, ['Date', 'Weather'])

def extract_columns_from_filename(dataDF):
    """Extracts the simulation parameters from the 'File' column strings and adds them as columns.

    Parameters
    ----------
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame.
    """

    dataDF = dataDF.withColumn('parts', F.split(F.col('File'), '_'))\
                   .withColumn('SoilFertility', F.substring(F.element_at('parts', 6), 9, 4))\
                   .withColumn('Irrigation', F.substring(F.element_at('parts', 7), 11, 3))\
                   .select(*dataDF.columns,
                           F.substring(F.element_at('parts', 1), 2, 4).cast(
                               IntegerType()).alias('Year'),
                           F.substring(F.element_at('parts', 2), 2, 2).cast(
                               IntegerType()).alias('FertMonth'),
                           F.substring(F.element_at('parts', 3), 2, 2).cast(
                               IntegerType()).alias('FertDay'),
                           F.substring(F.element_at('parts', 4), 11, 3).cast(
                               IntegerType()).alias('FertRate'),
                           F.substring(F.element_at('parts', 5), 5, 3).cast(
                               IntegerType()).alias('SoilWater'),
                           F.when(F.col('SoilFertility') == 'Low', 1).otherwise(
                               F.when(F.col('SoilFertility') == 'Med', 2).otherwise(3)).alias('SoilFertility'),
                           F.when(F.col('Irrigation') == 'Off', 0).otherwise(F.when(F.col('Irrigation') == 'On', 1)).alias('Irrigation'))

    return dataDF


def calculate_baseline_herbage_cuts(dataDF):
    """Calculates the herbage cuts for the baseline case and adds them to the supplied dataframe.

    Parameters
    ----------
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame
    """

    # group by simulation parameters
    baseline_herbagecut_aggDF = dataDF.filter(F.col('FertRate') == 0)\
                                      .groupBy(['Weather', 'Year', 'SoilWater',
                                               'SoilFertility', 'Irrigation', 'FertMonth',
                                                'FertDay'])\
                                      .agg(F.sum(
                                          F.when(
                                              # take the sum in 0<days<65 to include the 2 cuts
                                              (F.col('DaysRelative') >= 0) & (
                                                  F.col('DaysRelative') <= 65),
                                              F.col('HerbageCut'))).alias('baseline_0_65_HerbageCut'))

    dataDF = dataDF.join(baseline_herbagecut_aggDF, [
                         'Weather', 'Year', 'SoilWater', 'SoilFertility', 'Irrigation', 'FertMonth', 'FertDay'])

    return dataDF


def add_target_variable(dataDF, NitrogenResponseRateDF):
    """Adds the nitrogen response rate column to the supplied dataframe.

    Parameters
    ----------
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    NitrogenResponseRateDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the nitrogen response rates of the simulations

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame
    """
    # put target variable
    return dataDF.join(NitrogenResponseRateDF, ['File'])\
        .withColumnRenamed('NResp', 'target_var')


def remove_sims_herbagecut_0(dataDF):
    """Removes the simulations where the herbage cut of the baseline 0

    Parameters
    ----------
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame
    """
    return dataDF.filter(F.col('baseline_0_65_HerbageCut') != 0)


def remove_baseline_simulations(dataDF):
    """Removes the baseline simulations

    Parameters
    ----------
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame
    """
    return dataDF.filter(F.col('FertRate') != 0)


def keep_relevant_days(dataDF, downlimit, uplimit):
    """Retains only the rows of the days which are within the specified limits

    Parameters
    ----------
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    downlimit: int, the earliest day (relative to fertilization) to be contained in our retained data
    uplimit: int, the latest day (relative to fertilization) to be contained in our retained data

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame
    """
    return dataDF.filter((F.col('DaysRelative') >= downlimit) &
                         (F.col('DaysRelative') < uplimit))


# FEATURE GENERATION

def avg_per_days(per_days, total_days, feature):
    """Creates the expressions to calculate the average of 'feature' every 'per_days' days

    Parameters
    ----------
    per_days: int, the interval in days for the calculation of the average
    total_days: int, the total number of days in our time window
    feature: str, the name of the feature

    Returns
    -------
    Returns a list where each element is an expression which will be used later for aggregation.
    """

    expressions = []

    for i in range(-total_days, -per_days + 1, per_days):
        expressions.append(
            F.avg
            (F.when(
                (F.col('DaysRelative') >= i)
                &
                (F.col('DaysRelative') < i + per_days), F.col(feature)))
            .alias('avg_'+feature+'_'+str(i)+'_'+str(i + per_days - 1)))

    return expressions

def create_aggregated_dataset(dataDF, timeframes, biophysicals, avg_weather, avg_biophysical, downlimit, uplimit):
    """Aggregates the simulations based on the provided timeframes. Before the aggregation one row of 'dataDF' is one day in a simulation, after the aggregation one row is one simulation.

    Parameters
    ----------
    dataDF: int, pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    timeframes: list, contains in the length of the windows for which the weather/biophysical averages will be calculated
    biophysicals: list, biophysical variables of the simulations which we wish to retain
    avg_weather: bool, flag to indicate whether we want to have the average weather features calculated
    avg_biophysical: bool, flag to indicate whether we want to have the average biophysical features calculated
    downlimit: int, the earliest day (relative to fertilization) to be contained in our retained data
    uplimit: int, the latest day (relative to fertilization) to be contained in our retained data

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame
    """
    total_days = uplimit - downlimit + 1
    weather_factors = ['Rain', 'MaxT', 'MinT',
                       'RH', 'VP', 'Wind', 'Radn', 'PET']

    average_weather = [avg_per_days(timeframe, total_days, factor)
                       for timeframe in timeframes for factor in weather_factors]
    average_biophysical = [avg_per_days(
        timeframe, total_days, factor) for timeframe in timeframes for factor in biophysicals]

    expressions = []

    if avg_weather:
        expressions = expressions + average_weather
    if avg_biophysical:
        expressions = expressions + average_biophysical

    # add target variable expression
    expressions = expressions + \
        [F.first(F.col('target_var')).alias('target_var')]

    # aggregate based on the expressions inside 'expressions'
    feature_aggDF = dataDF.groupBy(['Weather', 'Year', 'SoilWater',
                                    'SoilFertility', 'Irrigation', 'FertMonth',
                                    'FertDay', 'FertRate'])\
        .agg(*helper_functions.flatten(expressions))

    return feature_aggDF


# FEATURE GENERATION PIPELINE

def preprocessing_pipeline(remove_herbagecut, timeframes, downlimit, uplimit, biophysicals, avg_weather, avg_biophysical, simulations_path, weather_path, nitrogen_path, base_path):
    """Orchestrates the preprocessing of the data and saves the results. Initializes spark, reads the simulation/weather/nitrogen data, processes them, saves them

    Parameters
    ----------
    remove_herbagecut: bool, flag that shows if simulations which have in their corresponding baseline simulation herbage cut = 0 should be removed or not
    timeframes: list, contains in the length of the windows for which the weather/biophysical averages will be calculated
    downlimit: int, earliest (inclusive) relative day before fertilization for which we assume to have data
    uplimit: int, latest (inclusive) relative day before fertilization for which we assume to have data
    biophysicals: list, biophysical variables of the simulations which we wish to retain
    avg_weather: bool, flag to indicate whether we want to have the average weather features calculated
    avg_biophysical: bool, flag to indicate whether we want to have the average biophysical features calculated
    simulations_path: str, the path to the csvs with the simulations
    weather_path: str, the path to the csvs with the weather
    nitrogen_path: str, the path to the cvs with the calculated nitrogen respose rates
    base_path: str, the path where the aggregated datasets will be saved

    Returns
    -------
    Returns None.
    """

    # create a spark cluster
    spark = SparkSession.builder.master("local[25]")\
        .config('spark.driver.memory', '40g')\
        .config('spark.executor.memory', '8g')\
        .config('spark.network.timeout', '800s')\
        .config('spark.sql.legacy.timeParserPolicy', 'LEGACY')\
        .getOrCreate()

    # read data
    simulationsDF = spark.read.csv(
        simulations_path, header=True, mode='FAILFAST', schema=schemas.clover_simulations_schema)
    WeatherDataAllDF = spark.read.csv(
        weather_path, header=True, mode='FAILFAST', schema=schemas.weather_schema)
    NitrogenResponseRateDF = spark.read.csv(
        nitrogen_path, header=True, mode='FAILFAST', schema=schemas.nitrogen_response_rate_schema)

    NitrogenResponseRateDF = NitrogenResponseRateDF.filter(F.col(
        'PastureType') == 'GrassClover')  # keep only the values that refer to clover

    # setting up data before aggregations
    dataDF = combine_input_datasets(
        WeatherDataAllDF, simulationsDF, biophysicals)
    dataDF = extract_columns_from_filename(dataDF)
    dataDF = calculate_baseline_herbage_cuts(dataDF)
    dataDF = add_target_variable(dataDF, NitrogenResponseRateDF)
    if remove_herbagecut:
        dataDF = remove_sims_herbagecut_0(dataDF)
    dataDF = remove_baseline_simulations(dataDF)
    dataDF = keep_relevant_days(dataDF, downlimit, uplimit)

    dataDF.cache()

    # split data to avoid information leaks later
    train_years = list(range(1979, 2011))
    test_years = list(range(2011, 2019))
    train_dataDF = dataDF.filter(F.col('Year').isin(train_years))
    test_dataDF = dataDF.filter(F.col('Year').isin(test_years))

    for df_name, df in {'train': train_dataDF, 'test': test_dataDF}.items():

        featureDF = create_aggregated_dataset(
            df, timeframes, biophysicals, avg_weather, avg_biophysical, downlimit, uplimit)

        if df_name == 'train':
            for location_type in ['known', 'unknown']:
                for location in ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8']:
                    for model_type in ['local', 'regional', 'global']:
                        if location_type == 'known':
                            # if the location is known, use locations from climate matching
                            locations = location_match.climate_match(
                                model_type, location_type, location)
                            df = featureDF.filter(
                                F.col('Weather').isin(locations))
                        else:
                            # if the location is unknown, use the nearest locations
                            if model_type == 'local':
                                closest_locations = location_match.get_n_closest_locations(1)[
                                    location]
                            elif model_type == 'regional':
                                closest_locations = location_match.get_n_closest_locations(3)[
                                    location]
                            else:
                                closest_locations = location_match.get_n_closest_locations(7)[
                                    location]
                            df = featureDF.filter(
                                F.col('Weather').isin(closest_locations))

                        # create the path where the test sets will be saved
                        save_path = base_path + '/' + model_type + \
                            '/' + location_type + '/' + df_name + '/'
                        helper_functions.create_path(save_path)
                        # save results
                        df.write.csv(save_path + location +
                                     '.csv', header=True)
                        print('Finished: ' + model_type + ',' +
                              location_type + ',' + location)
        else:
            # create the path where the test sets will be saved
            helper_functions.create_path(base_path + '/location/test/')

            for location in ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8']:
                df = featureDF.filter(F.col('Weather') == location)

                save_path = base_path + '/location/' + df_name + '/'
                helper_functions.create_path(save_path)
                # save results
                df.write.csv(save_path + location + '.csv', header=True)
                print('Finished location: ' + location + ' ' + df_name)
