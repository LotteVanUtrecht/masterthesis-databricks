# Databricks notebook source
# MAGIC %md #Bids Simulator

# COMMAND ----------

import numpy as np
import pandas as pd
import sys
import operator
import math
import random
import pyspark.mllib 
import pyspark.sql.functions as f
from matplotlib import pyplot as plt
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, BooleanType, StructField, LongType, DateType,TimestampType, FloatType, ArrayType
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel, LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder #Estimator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline

# COMMAND ----------

#import data
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")

#create list of deal_ids
deal_ids = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/").select("deal_id").distinct() #maybe add test?
deal_ids_list = list(deal_ids.select("deal_id").toPandas()["deal_id"])

#import bid height models => put them in a dictionary
bid_height_models = {deal_id : [float(row.max_bid) for row in spark.read.parquet("s3://rtl-databricks-datascience/lpater/bid_heights/"+deal_id+".parquet/").collect()] for deal_id in deal_ids_list}

# COMMAND ----------

#simulate a set of bidders for each deal_id & market

market_predictions = market_test.select("features","market_guid")
biddingprob=udf(lambda v:float(v[1]),FloatType())

for deal_id in deal_ids_list:
  model = LogisticRegressionModel.load(f"/mnt/lotte/logistic_regression/{deal_id}/") #load the model for deal_id
  #model.transform(market_predictions).select("probability").show()
  
  market_predictions = model.transform(market_predictions) #for each impression, estimate the probability that deal_id will bid
  
  market_predictions = market_predictions\
  .withColumn("rand",f.rand())\
  .withColumn(deal_id, 
              f.when(f.col("rand") < biddingprob(f.col("probability")),1)\
              .otherwise(0))\
  .drop("rawPrediction","prediction","probability","rand")#deal_id will bid with the predicted probability from the model

# COMMAND ----------

#gather all the bids in one column
deal_ids_schnam = market_predictions.schema.names

gather_bidders = udf(lambda row: [deal_ids_schnam[x] for x in range(len(deal_ids_schnam)) if row[x] == 1], ArrayType(StringType()))

market_predictions = market_predictions.withColumn("bidders", gather_bidders(f.struct([market_predictions[x] for x in market_predictions.columns])))

# COMMAND ----------

#simulate the bid heights for each bid
def simulate(bidders):
  bids = []
  for bidder in bidders:
    bids.append(random.choice(bid_height_models[bidder]))
  return(bids)

simulate_bids = udf(lambda z: simulate(z), ArrayType(FloatType()))

market_predictions = market_predictions.withColumn("bids", simulate_bids("bidders"))

# COMMAND ----------



# COMMAND ----------

#select the max bid
def max_bid_try_except(bids):
  try: max_bid = max(bids)
  except: max_bid = 0
  return(max_bid)

max_bid= udf(lambda z: max_bid_try_except(z), FloatType())

market_predictions = market_predictions.withColumn("max_bid", max_bid("bids"))

# COMMAND ----------

#get the true winning bids and winning bidders
bids_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_test.parquet/")\
  .filter(f.col("win")=="true")\
  .withColumnRenamed("deal_id","true_winning_deal_id")\
  .withColumnRenamed("max_bid","true_winning_max_bid")\
  .drop("win")

#join these with the simulations
market_predictions = market_predictions\
  .join(bids_test,"market_guid", how='outer')

# COMMAND ----------

market_predictions.select("bidders","bids","max_bid","true_winning_deal_id","true_winning_max_bid").show(20,False)

# COMMAND ----------

market_predictions.select("bids","max_bid").show(100,False)

# COMMAND ----------

market_predictions.select("bidders","bids","max_bid","true_winning_deal_id","true_winning_max_bid").write.mode('overwrite').parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_predictions.parquet/")
