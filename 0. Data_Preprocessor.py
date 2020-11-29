# Databricks notebook source
# MAGIC %md # Custom_Fields Processor
# MAGIC 
# MAGIC The goal of this notebook is to a) do some analysis on the Custom_Fields part of the market file and b) to develop a function that takes the custom fields out of their own column and into new attributes.

# COMMAND ----------

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
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline

# COMMAND ----------

spark.read.parquet("s3://rtl-databricks-datascience/lpater/market_sample_2020-02-24.parquet/").count()

# COMMAND ----------

market_sample.count()

# COMMAND ----------

print(spark.read.parquet("s3://rtl-databricks-datascience/lpater/bids_sample_2020-02-24.parquet/").count())

# COMMAND ----------

#load in the relevant Parquet file
market_sample = spark.read.parquet("s3://rtl-databricks-datascience/lpater/market_sample_2020-02-24.parquet/")\
.filter(f.col('custom_fields')!='custom_fields')\
.where(~f.col('custom_fields').contains('%%PATTERN'))\
.where(~f.col('custom_fields').contains("iphone-chromecast"))\
.where(~f.col('custom_fields').contains("android-chromecast"))\
.where(~f.col('custom_fields').contains("ipad-chromecast"))

bids_sample = spark.read.parquet("s3://rtl-databricks-datascience/lpater/bids_sample_2020-02-24.parquet/")

# COMMAND ----------

# MAGIC %md # Custom_Fields Processor

# COMMAND ----------

#creates basic convertor to call upon in the main function

def string_to_dict(string, feature):
  ####In:
  #A string containing a dictionary that has been misinterpreted as a dictionary
  #A feature to extract from this 'dictionary'
  
  ####Out:
  #The value of the desired feature if it exists, or "NA" if it doesn't
  
  string = string.replace("\"\"", "\"").replace("\"{","{").replace("}\"","}")
  dictionary = eval(string)
  
  try: value = dictionary[feature]
  except: value = "No Data Available"
  if value=="":
    value = "No Data Available"
  return value

StringToDict = udf(string_to_dict,StringType())

#note: in case of doubles, this just stores the last occuring value
#to select: at least pos, cnt en environment. maybe zender, network, product, domain, branddelichannel

def custom_fields_processor(data, features):
  ####In:
  #data: a data frame with market data as imported from [FILE PATH HERE]. This has to contain a feature custom_fields, which is a dictionary interpreted as a string
  #features: a list of features to extract from the custom_fields
  
  ####Out:
  #The original data frame, with a column added for every feature in the features list
  for feature in features:
    data = data.withColumn(feature,StringToDict("custom_fields",f.lit(feature)))
  
  return(data)

# COMMAND ----------

features_cfp = ["hour","pos","cnt","environment","zender","domain","branddelichannel"]

market_data = custom_fields_processor(market_sample,features_cfp)

# COMMAND ----------

#data.write.mode('overwrite').parquet("s3://rtl-databricks-datascience/lpater/market_sample_2020-02-24_processed.parquet/")#

# COMMAND ----------

# MAGIC %md # Data Prep

# COMMAND ----------

market_sample = market_data.withColumn('hour', f.substring('ts', 13, 2).cast('integer'))
#add an hour column
bids_sample = bids_sample.withColumn('new_deal_id', f.regexp_replace('deal_id',"\.|18ff3",""))\
.drop('deal_id')\
.withColumnRenamed('new_deal_id','deal_id')
#remove the points from deal_id, since that's what causes problems later. also remove the common start of 18ff3

features = ["hour","pos","cnt","environment","zender","domain","branddelichannel"]

deal_ids = list(bids_sample.select("deal_id").distinct().toPandas()["deal_id"]) #create al list of deal_ids to select from

bids_sample.cache()
market_sample.cache()

# COMMAND ----------

#1. Created a pivot table from the deal_id vector
bids = bids_sample.select("market_guid","deal_id","max_bid","win")\
.withColumn("indicator",f.lit(1))\
.groupBy("market_guid")\
.pivot("deal_id")\
.agg(f.expr("coalesce(first(indicator),0)")\
.cast("integer"))

#2. Join the market with the pivot table to get a data frame containing just the relevant features and outcome variables
market = market_sample\
.select(*features,"market_guid")\
.join(bids.select(*deal_ids,"market_guid"),"market_guid", how='left')\
.fillna(0) #.drop("market_guid")\

# COMMAND ----------

inputSI = ["hour","pos","cnt","environment","zender","domain","branddelichannel"]
outputSI = ["hour_index","pos_index","cnt_index","environment_index","zender_index","domain_index","branddelichannel_index"]
indexer = StringIndexer(inputCols = inputSI, outputCols = outputSI)

outputOHE = ["hour_one_hot","pos_one_hot","cnt_one_hot","environment_one_hot","zender_one_hot","domain_one_hot","branddelichannel_one_hot"]
encoder = OneHotEncoder(inputCols = outputSI, outputCols = outputOHE)

outputVA = "features"
aggregator = VectorAssembler(inputCols = outputOHE, outputCol = outputVA)

#outputVA2 = "label"
#aggregator2 = VectorAssembler(inputCols = deal_ids, outputCol = outputVA2)

stages = [indexer, encoder, aggregator]#, aggregator2]
pipeline = Pipeline(stages=stages)

splits = pipeline.fit(market).transform(market)\
.select(*deal_ids,"features","market_guid")\
.randomSplit([0.9, 0.1], seed = 0)
#.select(*deal_ids,"features")\

market_train = splits[0]
market_test = splits[1]

# COMMAND ----------

bids_train = bids_sample\
.join(market_train.select("market_guid"),"market_guid", how='inner')\
.select("market_guid","deal_id","max_bid","win")

bids_test = bids_sample\
.join(market_test.select("market_guid"),"market_guid", how='inner')\
.select("market_guid","deal_id","max_bid","win")

# COMMAND ----------

market_train.write.mode('overwrite').parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_train.parquet/")
market_test.write.mode('overwrite').parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")
bids_train.write.mode('overwrite').parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/")
bids_test.write.mode('overwrite').parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_test.parquet/")
