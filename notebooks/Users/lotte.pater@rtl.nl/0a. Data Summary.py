# Databricks notebook source
import numpy as np
import pandas as pd
import sys
import operator
import math
import pyspark.mllib 
import pyspark.sql.functions as f
from matplotlib import pyplot as plt
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, BooleanType, StructField, LongType, DateType,TimestampType, FloatType
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel, LogisticRegression, LogisticRegressionModel, LinearRegression, LinearRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline

# COMMAND ----------

market_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_train.parquet/")
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")
bids_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/")
bids_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_test.parquet/")

# COMMAND ----------

market = market_train.unionAll(market_test)
bids = bids_train.unionAll(bids_test)
market.cache()
bids.cache()

# COMMAND ----------

market_with_features = spark.read.parquet("s3://rtl-databricks-datascience/lpater/market_sample_2020-02-24_processed.parquet/").withColumn('hour', f.substring('ts', 13, 2).cast('integer'))

# COMMAND ----------

#success and fill rate

bids.groupBy("win").count().show()
#+-----+-------+
#|  win|  count|
#+-----+-------+
#|false|3213896|
#| true|1868350|
#+-----+-------+

succes_rate = 1868350/(1868350+3213896)
fill_rate = 1868350/market.count()

print(succes_rate,fill_rate)

# COMMAND ----------

#bid rates by deal_id
deal_ids = bids.groupBy("deal_id").count()
deal_ids.orderBy('count', ascending=False).show(200)

# COMMAND ----------

#density plots of the bids
bids.select("max_bid").toPandas().plot(kind='density',xlim=(0,40))


# COMMAND ----------

#create a list of deal ids to plot densities for
active_deals = list(deal_ids.orderBy('count', ascending=False).select("deal_id").toPandas()[0:20]["deal_id"])

# COMMAND ----------

deal_bids = bids.filter(f.col("deal_id")==active_deals[i]).select("max_bid").toPandas()
deal_bids.plot(kind='density',xlim=(0,40))

# COMMAND ----------

fig, axs = plt.subplots(4, 5, sharex=True,sharey=True)
fig.suptitle("Density plots for the most active deals")
#axs[0:5], axs[5:10], axs[10:15], axs[15:20] = axs

for i in range(0,20):
  deal_bids = bids.filter(f.col("deal_id")==active_deals[i]).select("max_bid").toPandas()
  #axs[i//5,i % 5].plot(bids.filter(f.col("deal_id")==active_deals[i]).select("max_bid").toPandas()),kind='density',xlim=(0,40)
  axs[i//5,i % 5].hist(deal_bids,
                       xlim=(0,40),
                       ylim=(0,0.5),
                       density=True,
                       bins=np.arange(-0.5,39.5,2))
  
for ax in axs.flat:
    ax.label_outer()
#bids.select("max_bid").toPandas().plot(kind='density',xlim=(0,40),by=bids.select("deal_id").toPandas())

# COMMAND ----------

LinearRegression

# COMMAND ----------

#bids.agg({"max_bid": "min"}).collect()[0][0]

# COMMAND ----------

features = ["device_type","hour","pos","cnt","zender","domain","branddelichannel"]

for feature in features:
  market_with_features.groupBy(feature).count().orderBy('count', ascending=False).show(200)