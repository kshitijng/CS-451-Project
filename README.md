# Predicting Covid-19 cases next week and at the end of pandemic


### Initial setup

To run what we currently have, first install the python libs needed using:
`pip install -r requirements.txt`

the python version we use is python3 (so make relevant changes i.e pip3 etc)

If there are issues running pySpark, make sure you have the following env variables
`$PYSPARK_PYTHON` is set to where your python3 e.g.: `/usr/local/bin/python3`
and `$PYSPARK_DRIVER_PYTHON` is `ipython3`


### Hydrating the Twitter Data
Batch the tweets by month to load into the hydrator more quickly. Raw tweets are in `data/ieee_raw`, the monthly batches are output to `data/ieee_raw_monthly`

`python3 src/batch_by_month.py "apr,may,jun,jul,aug,sep,oct,nov,dec"`

Install the [hydrator application](https://github.com/DocNow/hydrator). (note that we've already hydrated data up to December 11)

Follow the steps in the hydrator repo to load the monthly batches and hydrate them. Output the hydrated .csv files into `data/ieee_hydrated`.

### Extract Bigrams and Unigrams from Tweets

`src/create_raw_features.py` creates the counts, bigrams etc. needed for the model for Canada.
Specifically it looks for tweets in ["CANADA", "ONTARIO", "VANCOUVER", "TORONTO", "OTTAWA", "MONTREAL", "WATERLOO", "KITCHENER"]. It will also provide the counts and bigrams globally, as it could be useful to analyze global trends.

Now we must process the results into usable csv files. To do this run `python3 src/extract_twitter_data.py`

### Prepare Google Trends Data and Government Data

The Google search trends data is in `data/google_trends`. Ensure that the data is as up to date as needed by going to https://trends.google.com/trends. The data is filtered to Canada from April 1, 2020 to present.

The Covid cases data is from https://github.com/ishaberry/Covid19Canada. It is available in `data/covid_cases`. Specifically, update the files for [active cases in Canada](https://github.com/ishaberry/Covid19Canada/blob/master/timeseries_canada/active_timeseries_canada.csv) and [new cases in Canada](https://github.com/ishaberry/Covid19Canada/blob/master/timeseries_canada/cases_timeseries_canada.csv).

### Compile all Data

All the data can now be compiled into a single csv file for input into the model. To do this, run `python3 src/compile_daily_data.py`

### Train Model

Adjust `data/ml_training/ml_input.csv` and `data/ml_training/ml_test.csv` as needed. The first file is what is being trained, and the second file is what is used to gauge error.

Run `python3 src/train_model.py 2> log.txt` to iterate through all the possible inputs to the model and find what is best. The `2> log.txt` will just direct excessive pyspark logs into a file so the output is more legible. This took about 3-5 minutes per iteration, and iterations continue until models stop improving.

### Predict the Next Week's Cases

Set up all the available data in `data/prediction_input/predict_input.csv`. Provide the data for the first day of the week to predict in `data/prediction_input/predict_test.csv`. For example, If predicting the week of December 5 to 11, provide the data for December 5.

Run `python3 src/predict_next_week.py 2> log.txt` to see the result from the three different models.

### Predict End of Pandemic

Run `python3 src/predict_end_of_pandemic.py`
