from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor
import csv
import os

OUTPUT_DIR = "data/ml_result"
TRAINING_DATA = "data/ml_training/ml_input.csv"
TEST_DATA = "data/ml_training/ml_test.csv"

LR_MODEL = []
DT_MODEL = []
GBDT_MODEL = []

TRAIN_LR = True
TRAIN_DT = False
TRAIN_GBDT = False


def output_data(data, file):
    '''
    Output results of model testing to a csv file.

    Args:
        data: A list of lists containing the model data
        file: The file path to which data is output
    '''
    dest = os.path.join(OUTPUT_DIR, file)
    print("Outputting to {}".format(dest))
    with open(dest, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(data)):
            w.writerow(data[i])


def run_linear_regression(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''
    Run a Linear Regression model on given training and test data.

    Args:
        train_df: A dataframe containing training data for the model.
        test_df: A dataframe containing test data for the model.
        features_col: The column in the dataframe containing the features.
        label_col: The column in the dataframe containing the predicted value.

    Return:
        r2: r^2 error of the model.
        rmse: Root Mean Squared Error of the model.
    '''
    lr = LinearRegression(featuresCol = features_col, labelCol=label_col,
                          maxIter=100, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)

    # r^2 value of the model on the training data
    trainingSummary = lr_model.summary
    r2 = trainingSummary.r2

    # Making predictions on test data and evaluating it
    lr_predictions = lr_model.transform(test_df)

    # Show the prediction for some rows
    # lr_predictions.select("prediction",label_col,features_col).show(5)

    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol=label_col,metricName="r2")

    test_result = lr_model.evaluate(test_df)
    rmse = test_result.rootMeanSquaredError

    return (r2, rmse)


def run_decision_tree(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''
    Run a decision tree model on given training and test data.

    Args:
        train_df: A dataframe containing training data for the model.
        test_df: A dataframe containing test data for the model.
        features_col: The column in the dataframe containing the features.
        label_col: The column in the dataframe containing the predicted value.

    Return:
        rmse: Root Mean Squared Error of the model.
    '''
    dt = DecisionTreeRegressor(featuresCol=features_col, labelCol=label_col)
    dt_model = dt.fit(train_df)
    dt_predictions = dt_model.transform(test_df)

    # Show the prediction for some rows
    # dt_predictions.select("prediction",label_col,features_col).show(5)

    dt_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = dt_evaluator.evaluate(dt_predictions)

    return rmse


def run_gb_tree(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''
    Run a Gradient-boosted Tree Regression model on given training and test data.

    Args:
        train_df: A dataframe containing training data for the model.
        test_df: A dataframe containing test data for the model.
        features_col: The column in the dataframe containing the features.
        label_col: The column in the dataframe containing the predicted value.

    Return:
        rmse: Root Mean Squared Error of the model.
    '''
    gbt = GBTRegressor(featuresCol=features_col, labelCol=label_col, maxIter=10)
    gbt_model = gbt.fit(train_df)
    gbt_predictions = gbt_model.transform(test_df)

    # Show the prediction for some rows
    # gbt_predictions.select('prediction', label_col, features_col).show(5)

    gbt_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = gbt_evaluator.evaluate(gbt_predictions)

    return rmse


def assemble_vectors(train_df, test_df, input_cols, features_col="features", label_col="actual_cases"):
    '''
    Given a training and test dataframe, extract the specified input_cols to
    create featuers and the label_col to create a label that can be used by
    ML models.

    Args:
        train_df: A dataframe containing training data for the model.
        test_df: A dataframe containing test data for the model.
        input_cols: The columns from the full dataframe that will be used as
                    features for the ML models.
        features_col: The name of the newly created features column.
        label_col: The name of the column in the data that is to be predicted
                   by the ML models.

    Returns:
        train2_df: A traning dataframe with the selected features and label
        test2_df: A test dataframe with the selected features and label
    '''
    vectorAssembler = VectorAssembler(
        inputCols = input_cols,
        outputCol = features_col
    )
    train2_df = vectorAssembler.transform(train_df).select([features_col, label_col])
    test2_df = vectorAssembler.transform(test_df).select([features_col, label_col])

    return (train2_df, test2_df)


def test_col_combos(train_df, test_df, lr=True, dt=True, gbdt=True):
    '''
    Given a training dataframe and test dataframe, test all the unused columns
    along with the already selected columns (in LR_MODEL, DT_MODEL,
    or GBDT_MODEL) in the respective ML model.
    Return a summary of all the results.

    Args:
        train_df: A dataframe containing training data for the model.
        test_df: A dataframe containing test data for the model.
        lr: Boolean indicator denoting whether to test linear regression.
        dt: Boolean indicator denoting whether to test decision tree.
        gbdt: Boolean indicator denoting whether to test gradient boosted
              decision tree.

    Returns:
        lr_result: A list of lists containing the results of linear regression
                   testing.
        dt_result: A list of lists containing the results of decision tree
                   testing.
        gbdt_result: A list of lists containing the results of gradient boosted
                     decision tree testing.
    '''
    cols = train_df.columns[2:]
    lr_result = []
    dt_result = []
    gbdt_result = []

    print("ITERATION {}:".format(1+max(len(LR_MODEL), len(DT_MODEL), len(GBDT_MODEL))))

    for i in range(len(cols)):
        if lr and (cols[i] not in LR_MODEL):
            model_inputs = LR_MODEL + [cols[i]]
            train2_df, test2_df = assemble_vectors(train_df, test_df, model_inputs)

            lr_r2, lr_rmse = run_linear_regression(train2_df, test2_df)
            lr_result.append(model_inputs + [lr_rmse])

        if dt and (cols[i] not in DT_MODEL):
            model_inputs = DT_MODEL + [cols[i]]
            train2_df, test2_df = assemble_vectors(train_df, test_df, model_inputs)

            dt_rmse = run_decision_tree(train2_df, test2_df)
            dt_result.append(model_inputs + [dt_rmse])

        if gbdt and (cols[i] not in GBDT_MODEL):
            model_inputs = GBDT_MODEL + [cols[i]]
            train2_df, test2_df = assemble_vectors(train_df, test_df, model_inputs)

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
    '''
    Given the result of testing a model, determine which test had the lowest
    root mean squared error. Return the column tested and the error value.

    Args:
        res: list of lists containing the result of tests.

    Returns:
        best_col: Name of the column which had the lowest error.
        best_val: The error value of the best_col.
    '''
    best_col = res[0][-2]
    best_val = res[0][-1]
    for i in range(1, len(res)):
        if res[i][-1] < best_val:
            best_col = res[i][-2]
            best_val = res[i][-1]

    print("    BEST COL: {} - {}".format(best_col, best_val))

    return best_col, best_val

def main():
    '''
    Read a training and test dataframe from specified input files. Create a
    model for linear regression, decision tree, and gradient boosted decision
    tree strategies using the following strategy:

    Create a model with each column and find the one with the lowest error.
    Create every model with two columns where the first is the best single
    column. Choose the best model from this step. Add another column by testing
    every remaining column and choosing the lowest error. Repeat this until
    adding a new column does not decrease the error.

    Outputs:
        The result of every test into csv files as denoted by the logging.

    Prints:
        The list of columns that achieved the lowest error with the above
        strategy. This is for each ML model.
    '''

    sc= SparkContext()
    sqlContext = SQLContext(sc)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Load in Data

    train_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load(TRAINING_DATA)
    test_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load(TEST_DATA)

    lr, dt, gbdt = TRAIN_LR, TRAIN_DT, TRAIN_GBDT
    lr_best, dt_best, gbdt_best = 1000000, 1000000, 1000000

    while (lr or dt or gbdt):
        lr_res, dt_res, gbdt_res = test_col_combos(train_df, test_df, lr, dt, gbdt)

        if lr:
            print("  LR:")
            lr_col, lr_best_new = find_best_col(lr_res)
            if lr_best_new < lr_best:
                lr_best = lr_best_new
                LR_MODEL.append(lr_col)
            else:
                lr = False
                print("  DONE WITH LR AT SIZE: {}".format(len(LR_MODEL)))

        if dt:
            print("  DT:")
            dt_col, dt_best_new = find_best_col(dt_res)
            if dt_best_new < dt_best:
                dt_best = dt_best_new
                DT_MODEL.append(dt_col)
            else:
                dt = False
                print("  DONE WITH DT AT SIZE: {}".format(len(DT_MODEL)))

        if gbdt:
            print(" GBDT:")
            gbdt_col, gbdt_best_new = find_best_col(gbdt_res)
            if gbdt_best_new < gbdt_best:
                gbdt_best = gbdt_best_new
                GBDT_MODEL.append(gbdt_col)
            else:
                gbdt = False
                print("  DONE WITH GBDT AT SIZE: {}".format(len(GBDT_MODEL)))

    # Print a summary of the results
    print("")
    print("--------------------------------------------------")
    print("FINAL RESULTS:")
    print("")
    print("LR_MODEL = [\n'{}']".format("', \n'".join(LR_MODEL)))
    print("")
    print("DT_MODEL = [\n'{}']".format("', \n'".join(DT_MODEL)))
    print("")
    print("GBDT_MODEL = [\n'{}']".format("', \n'".join(GBDT_MODEL)))

if __name__ == "__main__":
    main()
