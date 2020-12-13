from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor
import csv
import os

TRAINING_DATA = "data/prediction_input/predict_input.csv"
TEST_DATA = "data/prediction_input/predict_test.csv"

TEST_LR = True
TEST_DT = False
TEST_GBDT = False

# NEW MODEL FOR 5 - 11 # 45968 (0.91 %)
# LR_MODEL = [
# 'new_cases_last_week',
# 'T_GB_tested-positive',
# 'T_CB_wave-of',
# 'T_CU_deaths',
# 'T_CU_corona',
# 'T_CU_tests',
# 'T_CU_positive',
# 'T_GB_new-cases',
# 'T_GB_coronavirus-in',
# 'G_lockdown',
# 'T_CU_pandemic',
# 'G_fever',
# 'T_CB_new-covid19',
# 'T_CB_covid19-cases',
# 'T_GB_the-virus',
# 'T_CU_ontario',
# 'T_CU_toronto',
# 'T_CU_ottawa',
# 'T_GB_spread-of',
# 'T_CU_case',
# 'new_cases_2w_ago',
# 'T_CU_cases',
# 'T_GB_covid19-testing',
# 'T_CU_covid',
# 'G_fatigue',
# 'T_GB_a-pandemic',
# 'T_GU_corona',
# 'T_CB_tested-positive',
# 'T_GU_restrictions',
# 'T_CU_quarantine']

# MODEL USED FOR DECEMBER 11 - 17
# USING 15 DAYS IN TRAINING GETS 50709
LR_MODEL = [
'new_cases_last_week',
'T_CB_wave-of',
'T_GB_spread-of',
'T_GU_restrictions',
'T_GB_tested-positive',
'T_CB_covid-cases',
'T_GU_deaths',
'new_cases_2w_ago',
'T_GB_covid19-pandemic',
'G_cough',
'T_GU_death',
'T_CU_death',
'T_CU_pandemic',
'T_GB_face-masks']

DT_MODEL = [
'active_cases',
'T_CU_lockdown',
'new_cases_2w_ago',
'G_headache']

GBDT_MODEL = [
'active_cases',
'T_CU_lockdown',
'T_GB_social-distancing',
'new_cases_2w_ago',
'T_GB_a-pandemic',
'T_CU_testing',
'G_lockdown',
'T_CB_covid-cases']


def run_linear_regression(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''Run a Linear Regression model on this data'''
    print("  Running Linear Regression")
    lr = LinearRegression(featuresCol = features_col, labelCol=label_col,
                          maxIter=100, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)

    # Some stats about the training model
    trainingSummary = lr_model.summary
    r2 = trainingSummary.r2

    # Making predictions on test data and evaluating it
    lr_predictions = lr_model.transform(test_df)

    lr_predictions.select('prediction', 'date', features_col).show(20)


def run_decision_tree(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''Run a decision tree.'''

    print("  Running Decision Tree")
    dt = DecisionTreeRegressor(featuresCol=features_col, labelCol=label_col)
    dt_model = dt.fit(train_df)
    dt_predictions = dt_model.transform(test_df)

    dt_predictions.select('prediction', 'date', features_col).show(20)


def run_gb_tree(train_df, test_df, features_col="features", label_col="actual_cases"):
    '''Run a Gradient-boosted Tree Regression'''
    print("  Running Gradient Boosted Decision Tree")
    gbt = GBTRegressor(featuresCol=features_col, labelCol=label_col, maxIter=10)
    gbt_model = gbt.fit(train_df)
    gbt_predictions = gbt_model.transform(test_df)
    gbt_predictions.select('prediction', 'date', features_col).show(20)


def assemble_vectors(train_df, test_df, input_cols, features_col="features", label_col="actual_cases"):
    '''Test all models on inputs'''
    vectorAssembler = VectorAssembler(
        inputCols = input_cols,
        outputCol = features_col
    )
    train2_df = vectorAssembler.transform(train_df).select([features_col, label_col])
    test2_df = vectorAssembler.transform(test_df).select([features_col, 'date'])

    return (train2_df, test2_df)


def predict_cases(train_df, test_df):
    # LR Check
    if TEST_LR:
        train_lr_df, test_lr_df = assemble_vectors(train_df, test_df, LR_MODEL)
        run_linear_regression(train_lr_df, test_lr_df)

    # DT Check
    if TEST_DT:
        train_dt_df, test_dt_df = assemble_vectors(train_df, test_df, DT_MODEL)
        run_decision_tree(train_dt_df, test_dt_df)

    # GBDT Check
    if TEST_GBDT:
        train_gbdt_df, test_gbdt_df = assemble_vectors(train_df, test_df, GBDT_MODEL)
        run_gb_tree(train_gbdt_df, test_gbdt_df)

def main():
    sc= SparkContext()
    sqlContext = SQLContext(sc)

    # Load in Data
    train_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load(TRAINING_DATA)
    test_df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true').load(TEST_DATA)

    predict_cases(train_df, test_df)



if __name__ == "__main__":
    main()
