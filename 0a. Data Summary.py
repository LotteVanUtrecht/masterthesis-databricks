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
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel, LogisticRegression, LogisticRegressionModel
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
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
bids.select("max_bid").toPandas().hist(density=True,
                                       bins=np.arange(-0.5,39.5,2))
                                       #height=0.5)


# COMMAND ----------

active_deals = deal_ids.orderBy('count', ascending=False).select("count").toPandas()[0:20]
active_deals.describe()
148756.95*20/bids.count()

# COMMAND ----------

#create a list of deal ids to plot densities for
active_deals = list(deal_ids.orderBy('count', ascending=False).select("deal_id").toPandas()[0:20]["deal_id"])

# COMMAND ----------

fig, axs = plt.subplots(4, 5, sharex=True,sharey=True,figsize=(15,12))
fig.suptitle("Density plots for the most active deals",size=30)

for i in range(0,20):
  deal_bids = bids.filter(f.col("deal_id")==active_deals[i]).select("max_bid").toPandas().to_numpy()
  axs[i//5,i % 5].hist(deal_bids,
                       density=True,
                       bins=np.arange(-0.5,39.5,2))
  axs[i//5,i % 5].set_title(active_deals[i])
  
for ax in axs.flat:
    ax.label_outer()
    

# COMMAND ----------

fig3 = plt.figure(constrained_layout=True,figsize=(18*0.8,13*0.8))
fig3.suptitle("Bid height densities for the 20 most active deals and all deals together",size=20)
gs = fig3.add_gridspec(4, 6)#,sharex=True,sharey=True)

f3_axs = axs

for i in range(0,20):
    Xfrom = i//5
    Yfrom = i % 5
    if i in range(0,6): Xto,Yto=0,i
    elif i in range(6,8): Xto,Yto=1,i-6
    elif i in range(8,10):  Xto,Yto=1,i-4
    elif i in range(10,12): Xto,Yto=2,i-10
    elif i in range(12,14): Xto,Yto=2,i-8
    elif i in range(14,20): Xto,Yto=3,i-14
    
    f3_axs[Xfrom,Yfrom] = fig3.add_subplot(gs[Xto, Yto],ylim=(0,0.5))
    deal_bids = bids.filter(f.col("deal_id")==active_deals[i]).select("max_bid").toPandas().to_numpy()
    f3_axs[Xfrom,Yfrom].hist(deal_bids,
                       density=True,
                       bins=np.arange(-0.5,39.5,2))
    f3_axs[Xfrom,Yfrom].xaxis.set_ticks([])
    f3_axs[Xfrom,Yfrom].yaxis.set_ticks([])
    f3_axs[Xfrom,Yfrom]

f3_axs_all_deals = fig3.add_subplot(gs[1:3, 2:4],ylim=(0,0.5))    
deal_bids = bids.select("max_bid").toPandas().to_numpy()
f3_axs_all_deals.hist(deal_bids,
                       density=True,
                       bins=np.arange(-0.5,39.5,2),
                       color="black")
f3_axs_all_deals.set_title("All deals together")

fig3.subplots_adjust(top=12/13)

#for i in range(5,6):
#    f3_axs[1,0] = fig3.add_subplot(gs[0, i])
#    deal_bids = bids.filter(f.col("deal_id")==active_deals[i]).select("max_bid").toPandas().to_numpy()
#    f3_axs[1,0].hist(deal_bids,
#                       density=True,
#                       bins=np.arange(-0.5,39.5,2))

# COMMAND ----------

deal_means_train = bids_train.groupBy("deal_id").mean()
deal_means_train.show(200)
global_means_train = float(bids_train.select("max_bid").describe().toPandas().to_numpy()[1][1])
global_means_train

# COMMAND ----------

VAF_by_deals = bids_test\
.join(deal_means_train,"deal_id", how='left')\
.withColumn('variance',np.square(f.col("max_bid")-global_means_train))\
.withColumn('MSE',np.square(f.col("max_bid")-f.col("avg(max_bid)")))\
.withColumn('squared_deal_difference',np.square(global_means_train-f.col("avg(max_bid)")))

# COMMAND ----------

VAF_by_deals.select("variance","MSE",'squared_deal_difference').describe().show()

# COMMAND ----------

features = ["device_type","hour","pos","cnt","zender","domain","branddelichannel"]

for feature in features:
  market_with_features.groupBy(feature).count().orderBy('count', ascending=False).show(200)