from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import csv

OUTPUT_FILE = "data/ml_result.csv"

def output_data(data, file=OUTPUT_FILE):
    print("\nGood Stuff:")
    j = 0
    with open(file, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(data)):
            w.writerow(data[i])
            if i > 0 and (data[i][2] < 25000 or data[i][4] < 25000):
                print("'{}',".format(data[i][0]))
                j += 1
    print("Total: {}".format(j))


def run_linear_regression(train_df, test_df, features_col="features",
                          label_col="actual_cases"):
    print("  Running Linear Regression")
    # Run a Linear Regression model training on this data
    lr = LinearRegression(featuresCol = features_col, labelCol=label_col,
                          maxIter=100, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)
    # print("Coefficients: " + str(lr_model.coefficients))
    # print("Intercept: " + str(lr_model.intercept))

    # Some stats about the training model
    trainingSummary = lr_model.summary
    r2 = trainingSummary.r2
    # print("    Training r^2: %f" % trainingSummary.r2)
    # print("    Training RMSE: %f" % trainingSummary.rootMeanSquaredError)

    # train_df.describe().show()

    # Making predictions on test data and evaluating it
    lr_predictions = lr_model.transform(test_df)
    # lr_predictions.select("prediction",label_col,features_col).show(5)

    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol=label_col,metricName="r2")
    # print("    Test r^2: %g" % lr_evaluator.evaluate(lr_predictions) )

    test_result = lr_model.evaluate(test_df)
    rmse = test_result.rootMeanSquaredError
    # print("    Test RMSE: %g" % rmse)

    return (r2, rmse)


def run_decision_tree(train_df, test_df, features_col="features",
                      label_col="actual_cases"):
    print("  Running Decision Tree")
    # Try with a decision tree
    from pyspark.ml.regression import DecisionTreeRegressor
    dt = DecisionTreeRegressor(featuresCol=features_col, labelCol=label_col)
    dt_model = dt.fit(train_df)
    dt_predictions = dt_model.transform(test_df)
    # dt_predictions.select("prediction",label_col,features_col).show(5)

    dt_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = dt_evaluator.evaluate(dt_predictions)
    # print("    Test RMSE: %g" % rmse)

    return rmse


def run_gb_tree(train_df, test_df, features_col="features",
                label_col="actual_cases"):
    print("  Running Gradient Boosted Decision Tree")
    # Try with Gradient-boosted Tree Regression
    from pyspark.ml.regression import GBTRegressor
    gbt = GBTRegressor(featuresCol=features_col, labelCol=label_col, maxIter=10)
    gbt_model = gbt.fit(train_df)
    gbt_predictions = gbt_model.transform(test_df)
    # gbt_predictions.select('prediction', label_col, features_col).show(5)

    gbt_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = gbt_evaluator.evaluate(gbt_predictions)
    # print("    Test RMSE: %g" % rmse)

    return rmse


def test_inputs(train_df, test_df, input_cols,
                features_col="features", label_col="actual_cases"):
    vectorAssembler = VectorAssembler(
        inputCols = input_cols,
        outputCol = features_col
    )
    test_df = vectorAssembler.transform(test_df).select([features_col, label_col])
    train_df = vectorAssembler.transform(train_df).select([features_col, label_col])

    # print(str(test_df.collect()))
    # print(str(train_df.collect()))

    lr_r2, lr_rmse = run_linear_regression(train_df, test_df)
    dt_rmse = run_decision_tree(train_df, test_df)
    gbdt_rmse = run_gb_tree(train_df, test_df)

    return [lr_r2, lr_rmse, dt_rmse, gbdt_rmse]


def test_cols(train_df, test_df, cols):
    result = [['column', 'lr_training_r2', "lr_rmse", "dt_rmse", 'gbdt_rmse']]
    # cols = ['T_CU_quarantine', 'T_CU_positive']
    for c in cols:
        print("Checking {}".format(c))
        row = test_inputs(train_df, test_df, [c])
        result.append([c] + row)
        # break

    print("RESULT: {}".format(result))
    output_data(result)


def test_col_combos(train_df, test_df):
    KEEP = [
        'new_cases_last_week',      # 979.76
        'T_CU_new',                 # 598.68
        'T_GU_restrictions',        # 446.57
        'T_CB_covid19-pandemic',    # 425.98
        'T_CU_deaths',              # 338.46
        'T_CB_covid-cases',         # 240.34
        'T_GB_tested-positive',     # 212.15
        'T_CU_tests',               # 203.40
        'G_symptoms',               # 195.30
    ]
    cols = train_df.columns[2:]
    result = [['columns', 'lr_training_r2', "lr_rmse", "dt_rmse", 'gbdt_rmse']]
    for i in range(len(cols)):
        if cols[i] in KEEP:
            continue
        # for j in range(i+1, (len(COLS))):
        test_cols = KEEP + [cols[i]]
        print("Checking {}".format(test_cols))
        row = test_inputs(train_df, test_df, test_cols)
        result.append([str(test_cols)] + row)
            # print("{},{}".format(i,j))

    output_data(result)

def main():
    sc= SparkContext()
    sqlContext = SQLContext(sc)

    # Load in Data
    df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load('data/ml_input.csv')

    # df.cache()
    # df.printSchema()

    train_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load('data/ml_input.csv')
    test_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load('data/ml_test.csv')

    # Get all the non date/feature columns
    cols = train_df.columns[2:]

    # test_cols(train_df, test_df, cols)

    test_col_combos(train_df, test_df)

if __name__ == "__main__":
    main()

# Assemble data for ML (everything input is features)
# vectorAssembler = VectorAssembler(
#     inputCols = [
#         'new_cases_last_week',
#         'new_cases_2w_ago',
#         'active_cases',
#         # 'G_anti-mask-law',
#         # 'G_coronavirus',
#         'G_cough',
#         'G_covid',
#         'G_fatigue',
#         'G_fever',
#         'G_flu',
#         'G_headache',
#         'G_lockdown',
#         # 'G_mask',
#         'G_sick',
#         'G_symptoms',
#         'G_tired',
#         'G_virus'
#     ],
#     outputCol = 'features'
# )
# v_df = vectorAssembler.transform(df)
# v_df = v_df.select(['features', 'actual_cases'])
#
# # Split into a training and a testing model
# #splits = v_df.randomSplit([0.7, 0.3])
# known = df[:-1]
# last = df[-1]
#
# train_df = known #splits[0]
# test_df = last #splits[1]
# # train_df = vectorAssembler.transform(train_df)
# # train_df = train_df.select(['features', 'actual_cases'])
# # test_df = vectorAssembler.transform(test_df)
# # test_df = test_df.select(['features', 'actual_cases'])
