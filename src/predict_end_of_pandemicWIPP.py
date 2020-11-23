from pyspark import SparkConf, SparkContext, SparkFiles
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)


# Load in the case data 
sc.addFile("https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_canada/cases_timeseries_canada.csv")
data = sqlContext.read.format("csv") \
    .options(header='true', inferschema='true') \
    .load(SparkFiles.get("cases_timeseries_canada.csv"))
data.cache()
data.printSchema()

