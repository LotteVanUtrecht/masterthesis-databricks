# Databricks notebook source
# MAGIC %md # Tree depth tuner
# MAGIC 
# MAGIC This notebook runs a train/test split testing 5 different tree depths for each deal-id. Based on this results, 

# COMMAND ----------

import numpy as np
import pandas as pd
import sys
import operator
import math
import pyspark.mllib 
import pyspark.sql.functions as f
from matplotlib import pyplot as plt
import sklearn.tree as tree
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, BooleanType, StructField, LongType, DateType,TimestampType, FloatType
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline

# COMMAND ----------

#Import data, load in cache for easy working
market_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_train.parquet/")
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")
market_train.cache()
market_test.cache()

variable_names = {dic['idx'] : dic["name"] for dic in market_train.schema["features"].metadata["ml_attr"]["attrs"]["binary"]} #stores the variable names to use later
deal_ids = list(spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/").groupBy("deal_id").count().orderBy('count', ascending=False).select("deal_id").toPandas()["deal_id"]) #create a list of deal_ids to select from, ordered by how common they are

# COMMAND ----------

#create the objects

tree = DecisionTreeClassifier()

paramGrid = ParamGridBuilder()\
    .addGrid(tree.maxDepth, [4, 5, 6, 7, 8]) \
    .build()

tvs = TrainValidationSplit(estimator=tree, 
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# COMMAND ----------

# Run TrainValidationSplit, and choose the best regularization paramer.
for deal_id in deal_ids:
  training_data = market_train.withColumnRenamed(deal_id,'label')
  #testing_data = market_test.withColumnRenamed(deal_id,'label')
  model = tvs.fit(training_data)
  #print({"accuracy" : model.summary.accuracy})  
  #print({variable_names[variable_number] : model.coefficients[variable_number.item()] for variable_number in model.coefficients.indices})