from pyspark.ml.feature import NGram
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from functools import reduce
from pyspark.sql.functions import udf
from pyspark.sql.functions import array
from pyspark.sql.functions import split
import string

translator = str.maketrans('', '', string.punctuation)

sc = SparkContext('local')
spark = SparkSession(sc)

def to_upper(s):
    return s.upper()

def split_date(s):
    split_string = s.split(" ")
    if (len(split_string) < 3):
        return s
    return split_string[1] + split_string[2]

def count_bigrams(s):
    bigrams_lst = list(s[1])
    bigram_count = {}
    for row in bigrams_lst:
        raw_bigrams_list = row[1]
        for raw_bigram in raw_bigrams_list:
            # TODO remove if done early.
            raw_bigram = raw_bigram.translate(translator)
            if raw_bigram in bigram_count:
                bigram_count[raw_bigram] += 1
            else:
                bigram_count[raw_bigram] = 1
    return str(bigram_count)

to_upper_udf = udf(lambda z: to_upper(z))
date_udf = udf(lambda z: split_date(z))

pattern_list_cap = ["CANADA", "ONTARIO", "VANCOUVER", "TORONTO", "OTTAWA", "MONTREAL", "WATERLOO", "KITCHENER"]

input_file_list = ["04_apr.csv", "05_may.csv", "06_jun.csv", "07_jul.csv", "08_aug.csv", "09_sep.csv", "10_oct.csv", "11_nov.csv", "12_dec.csv"]

for month in input_file_list:
    df = spark.read.csv('data/ieee_hydrated/' + month, header=True)

    # 1. Get global tweets with non-null normalized location and dates.
    non_null_df = df.filter(df.user_location.isNotNull()).withColumn("cap_loc", to_upper_udf(df["user_location"])).withColumn("new_text", to_upper_udf(df["text"])).withColumn("new_date", date_udf(df["created_at"])).select("cap_loc", "new_date", "new_text")

    # 2. Count number of Global Tweets per day.
    global_tweets_per_day = non_null_df.groupBy("new_date").count().select("new_date", "count").withColumnRenamed("count", "global_count")

    # 3. Get tweets in Canada.
    canada_df = non_null_df.where(
        reduce(lambda a, b: a|b, (non_null_df['cap_loc'].like('%'+pat+"%") for pat in pattern_list_cap))
    )

    # 4. Count number of Canada tweets per day.
    canada_tweets_per_day = canada_df.groupBy("new_date").count().select("new_date", "count").withColumnRenamed("count", "canada_count")

    result_df = canada_tweets_per_day.join(global_tweets_per_day, "new_date")

    # 5. Prep for Bigrams
    unigram_gen = NGram(n=1, inputCol="arr", outputCol="unigrams")
    bigram_gen = NGram(n=2, inputCol="arr", outputCol="bigrams")

    # 6. Canada Bigrams
    canada_df = canada_df.withColumn("arr", split(canada_df['new_text'], ' '))
    canada_bigram_count = bigram_gen.transform(canada_df).select("new_date", "bigrams").rdd.groupBy(lambda x: x[0]).map(lambda s: (s[0], count_bigrams(s)))
    canada_unigram_count = unigram_gen.transform(canada_df).select("new_date", "unigrams").rdd.groupBy(lambda x: x[0]).map(lambda s: (s[0], count_bigrams(s)))

    can_uni = spark.createDataFrame(canada_unigram_count).toDF("date", "unigram_counts")
    can_uni.orderBy("date").coalesce(1).write.format('csv').save('tst_result/canada_unigrams/' + month[0:6])

    can_bi = spark.createDataFrame(canada_bigram_count).toDF("date", "bigram_counts")
    can_bi.orderBy("date").coalesce(1).write.format('csv').save('tst_result/canada_bigrams/' + month[0:6])


    # 7. Global Bigrams
    non_null_df = non_null_df.withColumn("arr", split(non_null_df['new_text'], ' '))
    global_bigram_count = bigram_gen.transform(non_null_df).select("new_date", "bigrams").rdd.groupBy(lambda x: x[0]).map(lambda s: (s[0], count_bigrams(s)))
    global_unigram_count = unigram_gen.transform(non_null_df).select("new_date", "unigrams").rdd.groupBy(lambda x: x[0]).map(lambda s: (s[0], count_bigrams(s)))

    glo_uni = spark.createDataFrame(global_unigram_count).toDF("date", "unigram_counts")
    glo_uni.orderBy("date").coalesce(1).write.format('csv').save('tst_result/global_unigrams/' + month[0:6])

    glo_bi = spark.createDataFrame(global_bigram_count).toDF("date", "bigram_counts")
    glo_bi.orderBy("date").coalesce(1).write.format('csv').save('tst_result/global_bigrams/' + month[0:6])

    # 8. Tweet Content
    # Mentioned Canada
    mention_canada_list = ["CANADA", "ONTARIO", "VANCOUVER", "TORONTO", "OTTAWA", "MONTREAL", "WATERLOO", "KITCHENER"]
    mentioned_canada_tweets_per_day = non_null_df.where(
        reduce(lambda a, b: a|b, (non_null_df['new_text'].like('%'+pat+"%") for pat in mention_canada_list))
    ).groupBy("new_date").count().select("new_date", "count").withColumnRenamed("count", "mentioned_count")

    result_df = result_df.join(mentioned_canada_tweets_per_day, "new_date")
    result_df.orderBy("new_date").coalesce(1).write.format('csv').save('tst_result/' + month[0:6])
