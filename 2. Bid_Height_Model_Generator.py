# Databricks notebook source
# MAGIC %md # Bid Height Model Generator

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

bids_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/")

# COMMAND ----------

bids_train = bids_train.withColumn('new_deal_id', f.regexp_replace('deal_id',"\.|18ff3",""))\
.drop('deal_id')\
.withColumnRenamed('new_deal_id','deal_id')

bids_train.cache()

# COMMAND ----------

deal_ids = bids_train.select("deal_id").distinct()

deal_ids_list = list(deal_ids.select("deal_id").toPandas()["deal_id"])

# COMMAND ----------

for deal_id in deal_ids_list:
  temp_bids = bids_train.filter(bids_train.deal_id==deal_id).select("max_bid")
  temp_bids.write.mode('overwrite').parquet("s3://rtl-databricks-datascience/lpater/bid_heights/"+deal_id+".parquet/")

# COMMAND ----------

for deal_id in deal_ids_list:
  print(spark.read.parquet("s3://rtl-databricks-datascience/lpater/bid_heights/"+deal_id+".parquet/").rdd.takeSample(False, 1, seed=0))
  
  #to-do: round this