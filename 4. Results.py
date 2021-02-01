# Databricks notebook source
# MAGIC %md #Results

# COMMAND ----------

import numpy as np
import pandas as pd
import sys
import operator
import math
import random
import pyspark.mllib 
import pyspark.sql.functions as f
import numpy as np 
from matplotlib import pyplot as plt
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, BooleanType, StructField, LongType, DateType,TimestampType, FloatType, ArrayType
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel, LogisticRegression, LogisticRegressionModel,DecisionTreeClassificationModel,DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder #Estimator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline

# COMMAND ----------

market_predictions_lr = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_predictions_lr.parquet/")
market_predictions_trees = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_predictions_trees.parquet/")

# COMMAND ----------

#fill rate

#add boolean column for whether there was a simulated fill or not
flt_array_type = ArrayType(FloatType())
market_predictions_lr = market_predictions_lr.withColumn("filled_lr",f.col("max_bid").isNotNull())
market_predictions_trees = market_predictions_trees.withColumn("filled_trees",f.col("max_bid").isNotNull())

#join the fill columns together
market_filled = market_predictions_lr.select("market_guid","filled_lr","true_winning_deal_id","true_winning_max_bid")\
.join(market_predictions_trees.select("market_guid","filled_trees"),"market_guid",how="inner")\
.withColumn("filled_true",f.col("true_winning_max_bid").isNotNull())


# COMMAND ----------

market_predictions_trees.select("max_bid").summary().show()
market_predictions_trees.groupBy("filled_trees").count().orderBy('count', ascending=False).show()

# COMMAND ----------

#fill rates and fill correlations

fill_summary = market_filled.groupBy("filled_true","filled_lr","filled_trees").count().orderBy('count', ascending=False)
fill_summary.show(20,False)

market_filled.groupBy("filled_true","filled_lr").count().orderBy('count', ascending=False).show()
market_filled.groupBy("filled_true","filled_trees").count().orderBy('count', ascending=False).show()

market_filled.groupBy("filled_true").count().orderBy('count', ascending=False).show()
market_filled.groupBy("filled_trees").count().orderBy('count', ascending=False).show()
market_filled.groupBy("filled_lr").count().orderBy('count', ascending=False).show()

market_filled.groupBy("filled_trees","filled_lr").count().orderBy('count', ascending=False).show()
#conclusion: independent bidding assumption doesn't hold up

# COMMAND ----------

print(187514/568894,218755/568894,202017/568894)
print(104542/187514,114213/(568894-187514))
print(96665/187514,105352/(568894-187514))

print(116733/218755,85284/350139)

# COMMAND ----------

#distributions

market_predictions_lr = market_predictions_lr\
.withColumnRenamed("max_bid","lr_max_bid")\
.withColumnRenamed("bidders","lr_bidders")\
.withColumnRenamed("bids","lr_bids")

market_predictions_trees = market_predictions_trees\
.withColumnRenamed("max_bid","trees_max_bid")\
.withColumnRenamed("bidders","trees_bidders")\
.withColumnRenamed("bids","trees_bids")\
.drop("true_winning_deal_id","true_winning_max_bid")

market_predictions = market_predictions_lr\
.join(market_predictions_trees,"market_guid",how="outer")

# COMMAND ----------

fig, axs = plt.subplots(1, 2, sharex=True,sharey=True)
#market_predictions.select("lr_max_bid").toPandas().hist(density=True,bins=np.arange(-0.5,39.5,2))


true_bids_plotprocessed = market_predictions.select("true_winning_max_bid")\
              .dropna()\
              .withColumnRenamed("true_winning_max_bid","true")\
              .toPandas()\
              .to_numpy()

simulated_bids_plotprocessed = market_predictions.select("trees_max_bid")\
              .dropna()\
              .withColumnRenamed("trees_max_bid","simulated")\
              .toPandas()\
              .to_numpy

axs[0,0].hist(true_bids_plotprocessed,\
              density=True,\
              bins=np.arange(-0.5,39.5,2))
axs[0,0].set_title("true")


axs[0,1].hist(simulated_bids_plotprocessed,\
              density=True,\
              bins=np.arange(-0.5,39.5,2))
axs[0,1].set_title("simulated")

fig.show()

# COMMAND ----------

market_predictions\
.withColumn("lr_diff", f.col("lr_max_bid")-f.col("true_winning_max_bid"))\
.select("lr_diff")\
.toPandas()\
.hist(density=True,bins=np.arange(-25,25,2))

market_predictions\
.withColumn("trees_diff", f.col("trees_max_bid")-f.col("true_winning_max_bid"))\
.select("trees_diff")\
.toPandas()\
.hist(density=True,bins=np.arange(-25,25,2))

#conclusie: het overschat vaker dan het onderschat? wat raar

# COMMAND ----------

#plot the number of bidders
def get_length(bidders):
  length = bidders.__len__()
  return(length)

number_of_bids = udf(lambda z: get_length(z), IntegerType())

simulated_NoB = market_predictions\
.withColumn("number_of_bidders",number_of_bids("trees_bidders"))\
.groupBy("number_of_bidders")\
.count()\
.orderBy("count",ascending=False)\
.withColumnRenamed("count","simulation")

bids_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_test.parquet/")
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")

#filter(f.col("price_floor_net")<f.col("max_bid"))\
true_NoB = bids_test\
.join(market_test.select("market_guid","price_floor_net"),
     "market_guid",
     how="left")\
.groupBy("market_guid").count()\
.orderBy("count",ascending=False)\
.withColumnRenamed("count","number_of_bidders")\
.groupBy("number_of_bidders").count()\
.orderBy("count",ascending=False)\
.withColumnRenamed("count","true")

number_of_bidders = simulated_NoB.join(true_NoB,"number_of_bidders",how="outer").toPandas()
number_of_bidders.at[0,"true"] = 381380
number_of_bidders = number_of_bidders.fillna(0)

# COMMAND ----------

number_of_bidders["true"] = number_of_bidders["true"].apply(lambda x: x/(number_of_bidders.sum()["true"])).cumsum()
number_of_bidders["simulation"] = number_of_bidders["simulation"].apply(lambda x: x/(number_of_bidders.sum()["simulation"])).cumsum()

# COMMAND ----------

ax = number_of_bidders[["true","simulation"]].plot(ylim=(0,1),xlim=(0,15))
ax.legend()

# COMMAND ----------

number_of_bidders[["true","simulation"]]

# COMMAND ----------

market_predictions\
.withColumn("lr_length",number_of_bids("lr_bidders"))\
.withColumn("trees_length",number_of_bids("trees_bidders"))\
.select("lr_length","trees_length")\
.describe()\
.show()

# COMMAND ----------

market_predictions\
.select("lr_max_bid","trees_max_bid","true_winning_max_bid")\
.describe()\
.show()

# COMMAND ----------

#total values
market_predictions\
.select("lr_max_bid","trees_max_bid","true_winning_max_bid")\
.describe()\
.show()

# COMMAND ----------

print(218755*17.99)
print(202017*19.73)
print(187514*16.52)

# COMMAND ----------

#unused function
def get_market_value(market_items):
  #input: a set of market predictions:
  #output: average market value, fill rate, total market value
  bids = market_items.select("max_bid")
  items = bids.count()
  fill_rate = round(bids.dropna().count()/items, 3)
  total_value = round(bids.agg({"max_bid":"sum"}).collect()[0]["sum(max_bid)"])
  average_value = round(total_value/items,3)
  average_max_bid = round(average_value/fill_rate,3)
  
  results_simulated = {"market items":items, "fill rate":fill_rate,"total market value":total_value,"average market value":average_value,"average sell price": average_max_bid}
  
  return(results)

# COMMAND ----------

market_items_features = spark.read.parquet("s3://rtl-databricks-datascience/lpater/market_sample_2020-02-24_processed.parquet/")\
.withColumn('hour', f.substring('ts', 13, 2).cast('integer'))\
.join(market_predictions_trees,"market_guid",how="right")\
.select("market_guid","trees_max_bid","true_winning_max_bid","hour","pos","cnt","device_type","zender","domain","branddelichannel")

# COMMAND ----------

for feature in ["hour","pos","cnt","device_type","zender","domain","branddelichannel"]:
  categories = market_items_features\
  .select(feature).\
  groupBy(feature).\
  count().orderBy('count', ascending=False)\
  .select(feature).toPandas()[feature]
  
  for category in categories:
    market_subset = market_items_features.filter(f.col(feature)==category)
    
    try: true_value = market_subset.agg({"true_winning_max_bid":"sum"}).collect()[0]["sum(true_winning_max_bid)"] / market_subset.count() 
    except: true_value = 0
    
    try: sim_value = market_subset.agg({"max_bid":"sum"}).collect()[0]["sum(max_bid)"] / market_subset.count() 
    except: sim_value = 0
  
    print("Feature:",feature,"Category:",category,"True market price:", round(true_value,2),"Simulated market price:",round(sim_value,2))

# COMMAND ----------

hour_prices = pd.DataFrame(np.array([[21, 4.56, 7.91],
[20, 6.15, 8.58],
[19, 6.41, 7.82],
[18, 5.43, 6.92],
[17, 5.32, 6.6],
[16, 4.98, 6.3],
[15, 5.02, 6.0],
[22, 4.47, 8.2],
[14, 5.58, 6.26],
[11, 6.01, 6.77],
[12, 5.89, 6.47],
[13, 5.68, 6.37],
[10, 6.2, 6.95],
[9, 5.87, 7.1],
[8, 4.93, 6.09],
[7, 4.61, 5.24],
[23, 5.35, 8.04],
[6, 4.45, 5.18],
[5, 4.91, 5.11],
[0, 5.95, 8.17],
[1, 6.41, 8.34],
[4, 5.73, 5.93],
[2, 6.39, 7.45],
[3, 7.29, 6.5]]),
columns=['Hour', 'true','simulation'])\
.astype({'Hour': 'int32'})\
.sort_values(by="Hour")

# COMMAND ----------

ax = hour_prices[["true","simulation"]].plot(use_index=False,xlim=(0,23),ylim=(0,10))
ax.legend()

# COMMAND ----------

market_predictions\
.select("lr_max_bid","trees_max_bid","true_winning_max_bid")\
.describe()\
.show()

# COMMAND ----------

deal_id = "e86f7061c"
model_trees = DecisionTreeClassificationModel.load(f"/mnt/lotte/decision_trees/{deal_id}/")
print(model_trees.toDebugString)
#deal_id = "48f06d9af"