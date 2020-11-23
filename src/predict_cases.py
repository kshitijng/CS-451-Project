from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

sc= SparkContext()
sqlContext = SQLContext(sc)

# Load in Data
df = sqlContext.read.format('com.databricks.spark.csv') \
               .options(header='true', inferschema='true').load('data/ml_input.csv')

df.cache()
df.printSchema()

# Assemble data for ML (everything input is features)
vectorAssembler = VectorAssembler(
    inputCols = ['new_cases_last_week', 'new_cases_2w_ago', 'active_cases',
                 'G_anti-mask-law', 'G_coronavirus', 'G_cough', 'G_covid',
                 'G_fatigue', 'G_fever', 'G_flu', 'G_headache', 'G_lockdown',
                 'G_mask', 'G_sick', 'G_symptoms', 'G_tired', 'G_virus'],
    outputCol = 'features'
)
v_df = vectorAssembler.transform(df)
v_df = v_df.select(['features', 'actual_cases'])

# Split into a training and a testing model
splits = v_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

# Run a Linear Regression model training on this data
lr = LinearRegression(featuresCol = 'features', labelCol='actual_cases', maxIter=100, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# Some stats about the training model
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

train_df.describe().show()

# Making predictions on test data and evaluating it
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","actual_cases","features").show(100)

lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="actual_cases",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
