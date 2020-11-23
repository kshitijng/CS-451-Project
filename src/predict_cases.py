from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def getrows(df, rownums=None):
    return df.rdd.zipWithIndex().filter(lambda x: x[1] in rownums).map(lambda x: x[0])

sc= SparkContext()
sqlContext = SQLContext(sc)

# Load in Data
df = sqlContext.read.format('com.databricks.spark.csv') \
               .options(header='true', inferschema='true').load('data/ml_input.csv')

# df.cache()
# df.printSchema()

# train_df = sqlContext.read.format('com.databricks.spark.csv') \
#                .options(header='true', inferschema='true').load('data/ml_input.csv')
# test_df = sqlContext.read.format('com.databricks.spark.csv') \
#                .options(header='true', inferschema='true').load('data/ml_last10.csv')

# Assemble data for ML (everything input is features)
vectorAssembler = VectorAssembler(
    inputCols = [
        'new_cases_last_week',
        'new_cases_2w_ago',
        'active_cases',
        # 'G_anti-mask-law',
        # 'G_coronavirus',
        'G_cough',
        'G_covid',
        'G_fatigue',
        'G_fever',
        'G_flu',
        'G_headache',
        'G_lockdown',
        # 'G_mask',
        'G_sick',
        'G_symptoms',
        'G_tired',
        'G_virus'
    ],
    outputCol = 'features'
)
v_df = vectorAssembler.transform(df)
v_df = v_df.select(['features', 'actual_cases'])

# Split into a training and a testing model
splits = v_df.randomSplit([0.98, 0.02])
train_df = splits[0]
test_df = splits[1]
# train_df = vectorAssembler.transform(train_df)
# train_df = train_df.select(['features', 'actual_cases'])
# test_df = vectorAssembler.transform(test_df)
# test_df = test_df.select(['features', 'actual_cases'])

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
lr_predictions.select("prediction","actual_cases","features").show(5)

lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="actual_cases",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

test_result = lr_model.evaluate(test_df)
print("LR Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

# Try with a decision tree
from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'actual_cases')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_predictions.select("prediction","actual_cases","features").show(5)
dt_evaluator = RegressionEvaluator(
    labelCol="actual_cases", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("DT Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# Try with Gradient-boosted Tree Regression
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'actual_cases', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'actual_cases', 'features').show(5)

gbt_evaluator = RegressionEvaluator(
    labelCol="actual_cases", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("GBTR Root Mean Squared Error (RMSE) on test data = %g" % rmse)
