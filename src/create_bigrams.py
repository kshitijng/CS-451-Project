from pyspark.ml.feature import NGram
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from functools import reduce
from pyspark.sql.functions import udf
from pyspark.sql.functions import array
from pyspark.sql.functions import split


sc = SparkContext('local')
spark = SparkSession(sc)

def to_upper(s):
    return s.upper()

to_upper_udf = udf(lambda z: to_upper(z))
pattern_list_cap = ["CANADA", "ONTARIO", "VANCOUVER", "TORONTO", "OTTAWA", "MONTREAL", "WATERLOO", "KITCHENER"]

df = spark.read.csv('data/ieee_hydrated/04_apr.csv', header=True)
non_null_df = df.filter(df.user_location.isNotNull()).withColumn("cap_loc", to_upper_udf(df["user_location"]))

canada_df = non_null_df.where(
    reduce(lambda a, b: a|b, (non_null_df['cap_loc'].like('%'+pat+"%") for pat in pattern_list_cap))
).withColumn("arr", split(df['text'], ' '))

ngram = NGram(n=2, inputCol="arr", outputCol="bigrams")
bigram_df = ngram.transform(canada_df)

# Prints 1 bigram for April currently.
bigram_df.select("bigrams").show(1, truncate=False)



