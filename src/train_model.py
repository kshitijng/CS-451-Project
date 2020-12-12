from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor
import csv
import os

OUTPUT_DIR = "data/ml_result"

LR_MODEL = []

DT_MODEL = []

GBDT_MODEL = []


def output_data(data, file):
    dest = os.path.join(OUTPUT_DIR, file)
    print("Outputting to {}".format(dest))
    with open(dest, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(data)):
            w.writerow(data[i])


def run_linear_regression(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''Run a Linear Regression model on this data'''
    # print("  Running Linear Regression")
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


def run_decision_tree(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''Run a decision tree.'''

    # print("  Running Decision Tree")
    dt = DecisionTreeRegressor(featuresCol=features_col, labelCol=label_col)
    dt_model = dt.fit(train_df)
    dt_predictions = dt_model.transform(test_df)
    # dt_predictions.select("prediction",label_col,features_col).show(5)

    dt_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = dt_evaluator.evaluate(dt_predictions)
    # print("    Test RMSE: %g" % rmse)

    return rmse


def run_gb_tree(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''Run a Gradient-boosted Tree Regression'''
    # print("  Running Gradient Boosted Decision Tree")
    gbt = GBTRegressor(featuresCol=features_col, labelCol=label_col, maxIter=10)
    gbt_model = gbt.fit(train_df)
    gbt_predictions = gbt_model.transform(test_df)
    # gbt_predictions.select('prediction', label_col, features_col).show(5)

    gbt_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = gbt_evaluator.evaluate(gbt_predictions)
    # print("    Test RMSE: %g" % rmse)

    return rmse


def test_inputs(train_df, test_df, input_cols, features_col="features", label_col="actual_cases"):
    '''Test all models on inputs'''
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
    result = [] # [['column', 'lr_training_r2', "lr_rmse", "dt_rmse", 'gbdt_rmse']]
    # cols = ['T_CU_quarantine', 'T_CU_positive']
    for c in cols:
        print("Checking {}".format(c))
        row = test_inputs(train_df, test_df, [c])
        result.append([c] + row)
        # break

    print("RESULT: {}".format(result))
    output_data(result)


def assemble_vectors(train_df, test_df, input_cols, features_col="features", label_col="actual_cases"):
    '''Test all models on inputs'''
    vectorAssembler = VectorAssembler(
        inputCols = input_cols,
        outputCol = features_col
    )
    train2_df = vectorAssembler.transform(train_df).select([features_col, label_col])
    test2_df = vectorAssembler.transform(test_df).select([features_col, label_col])

    return (train2_df, test2_df)


def test_col_combos(train_df, test_df, lr=True, dt=True, gbdt=True):
    cols = train_df.columns[2:]
    lr_result = []
    dt_result = []
    gbdt_result = []

    print("ITERATION {}:".format(1+max(len(LR_MODEL), len(DT_MODEL), len(GBDT_MODEL))))

    for i in range(len(cols)):
        if lr and (cols[i] not in LR_MODEL):
            model_inputs = LR_MODEL + [cols[i]]
            train2_df, test2_df = assemble_vectors(train_df, test_df, model_inputs)

            # print("Checking LR {}".format(model_inputs))
            lr_r2, lr_rmse = run_linear_regression(train2_df, test2_df)
            lr_result.append(model_inputs + [lr_rmse])

        if dt and (cols[i] not in DT_MODEL):
            model_inputs = DT_MODEL + [cols[i]]
            train2_df, test2_df = assemble_vectors(train_df, test_df, model_inputs)

            # print("Checking DT {}".format(model_inputs))
            dt_rmse = run_decision_tree(train2_df, test2_df)
            dt_result.append(model_inputs + [dt_rmse])

        if gbdt and (cols[i] not in GBDT_MODEL):
            model_inputs = GBDT_MODEL + [cols[i]]
            train2_df, test2_df = assemble_vectors(train_df, test_df, model_inputs)

            # print("Checking GBDT {}".format(model_inputs))
            gbdt_rmse = run_gb_tree(train2_df, test2_df)
            gbdt_result.append(model_inputs + [gbdt_rmse])

    if lr:
        output_data(lr_result, "lr_{:02d}.csv".format(len(LR_MODEL)+1))
    if dt:
        output_data(dt_result, "dt_{:02d}.csv".format(len(DT_MODEL)+1))
    if gbdt:
        output_data(gbdt_result, "gbdt_{:02d}.csv".format(len(GBDT_MODEL)+1))

    return (lr_result, dt_result, gbdt_result)


def find_best_col(res):
    best_col = res[0][-2]
    best_val = res[0][-1]
    for i in range(1, len(res)):
        if res[i][-1] < best_val:
            best_col = res[i][-2]
            best_val = res[i][-1]

    print("    BEST COL: {} - {}".format(best_col, best_val))

    return best_col, best_val

def main():
    sc= SparkContext()
    sqlContext = SQLContext(sc)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Load in Data
    # df = sqlContext.read.format('com.databricks.spark.csv') \
    #                .options(header='true', inferschema='true').load('data/ml_input.csv')

    train_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load('data/ml_input.csv')
    test_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load('data/ml_test.csv')

    # Get all the non date/feature columns
    # cols = train_df.columns[2:]
    # test_cols(train_df, test_df, cols)

    lr, dt, gbdt = True, True, True
    lr_best, dt_best, gbdt_best = 1000000, 1000000, 1000000

    while (lr or dt or gbdt):
        lr_res, dt_res, gbdt_res = test_col_combos(train_df, test_df, lr, dt, gbdt)

        if lr:
            print("  LR:")
            lr_col, lr_best_new = find_best_col(lr_res)
            if lr_best_new < lr_best:
                lr_best = lr_best_new
                LR_MODEL.append(lr_col)
                # print("  ADD TO LR: {}".format(lr_col))
            else:
                lr = False
                print("  DONE WITH LR AT SIZE: {}".format(len(LR_MODEL)))

        if dt:
            print("  DT:")
            dt_col, dt_best_new = find_best_col(dt_res)
            if dt_best_new < dt_best:
                dt_best = dt_best_new
                DT_MODEL.append(dt_col)
                # print("  ADD TO DT: {}".format(dt_col))
            else:
                dt = False
                print("  DONE WITH DT AT SIZE: {}".format(len(DT_MODEL)))

        if gbdt:
            print(" GBDT:")
            gbdt_col, gbdt_best_new = find_best_col(gbdt_res)
            if gbdt_best_new < gbdt_best:
                gbdt_best = gbdt_best_new
                GBDT_MODEL.append(gbdt_col)
                # print("  ADD TO GBDT: {}".format(gbdt_col))
            else:
                gbdt = False
                print("  DONE WITH GBDT AT SIZE: {}".format(len(GBDT_MODEL)))

    print("")
    print("--------------------------------------------------")
    print("FINAL RESULTS:")
    print("")
    print("LR_MODEL = [\n'{}']".format("', \n'".join(LR_MODEL)))
    print("")
    print("DT_MODEL = [\n'{}']".format(", \n".join(DT_MODEL)))
    print("")
    print("GBDT_MODEL = [\n'{}']".format(", \n".join(GBDT_MODEL)))

if __name__ == "__main__":
    main()
