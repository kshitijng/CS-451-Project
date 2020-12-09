# Predicting cases next week and end of pandemic

To run what we currently have, first install the python libs needed using:
`pip install -r requirements.txt`

the python version we use is python3 (so make relevant changes i.e pip3 etc)

If there are issues running pySpark, make sure you have the following env variables
`$PYSPARK_PYTHON` is set to where your python3 e.g.: `/usr/local/bin/python3`
and `$PYSPARK_DRIVER_PYTHON` is `ipython3`

`src/create_raw_features.py` creates the counts, bigrams etc. needed for the model for Canada.
Specifically it looks for tweets in ["CANADA", "ONTARIO", "VANCOUVER", "TORONTO", "OTTAWA", "MONTREAL", "WATERLOO", "KITCHENER"]


