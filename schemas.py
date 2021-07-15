from pyspark.sql.types import StructField, StructType, StringType, DoubleType

clover_simulations_schema = StructType([
    StructField("File", StringType(), True),
    StructField("DaysRelative", DoubleType(), True),
    StructField("Date", StringType(), True),
    StructField("FertiliserApplied", DoubleType(), True),
    StructField("AboveGroundWt", DoubleType(), True),
    StructField("HerbageCut", DoubleType(), True),
    StructField("HerbageNCut", DoubleType(), True),
    StructField("Volatilisation", DoubleType(), True),
    StructField("Denitrification", DoubleType(), True),
    StructField("DenitN2O", DoubleType(), True),
    StructField("FixedN", DoubleType(), True),
    StructField("HerbageME", DoubleType(), True),
    StructField("NetPotentialGrowthWt", DoubleType(), True),
    StructField("NetPotentialGrowthAfterWaterWt", DoubleType(), True),
    StructField("NetPotentialGrowthAfterNutrientWt", DoubleType(), True),
    StructField("NetGrowthWt", DoubleType(), True),
    StructField("NetGrowthN", DoubleType(), True),
    StructField("AboveGroundNConc", DoubleType(), True),
    StructField("GlfWaterSupply", DoubleType(), True),
    StructField("GlfTemperature", DoubleType(), True),
    StructField("GlfNSupply", DoubleType(), True),
    StructField("GlfNContent", DoubleType(), True),
    StructField("SoilTemp050", DoubleType(), True),
    StructField("SoilTemp300", DoubleType(), True),
    StructField("SoilWater300", DoubleType(), True),
    StructField("SoilMinN300", DoubleType(), True),
    StructField("EvapDemand", DoubleType(), True),
    StructField("EvapActual", DoubleType(), True)
])

nitrogen_response_rate_schema = StructType([
    StructField("File", StringType(), True),
    StructField("PastureType", StringType(), True),
    StructField("NResp", DoubleType(), True)
])

weather_schema = StructType([
    StructField("Weather", StringType(), True),
    StructField("Date", StringType(), True),
    StructField("MaxT", DoubleType(), True),
    StructField("MinT", DoubleType(), True),
    StructField("Rain", DoubleType(), True),
    StructField("PET", DoubleType(), True),
    StructField("Radn", DoubleType(), True),
    StructField("VP", DoubleType(), True),
    StructField("RH", DoubleType(), True),
    StructField("Wind", DoubleType(), True)
])
