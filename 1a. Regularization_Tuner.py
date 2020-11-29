# Databricks notebook source
# MAGIC %md # Regularization tuner
# MAGIC 
# MAGIC This notebook runs a train/test split testing 5 different regularization parameters for each deal-id. Based on this results, we chose 0.0001 (the second-smallest option) as the regularization parameter.  

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
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder #Estimator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline

# COMMAND ----------

#Import data, load in cache for easy working
market_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_train.parquet/")
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")
market_train.cache()
market_test.cache()

variable_names = {dic['idx'] : dic["name"] for dic in market_train.schema["features"].metadata["ml_attr"]["attrs"]["binary"]} #stores the variable names to use later. actiepunt: check of dit goed gaat
deal_ids = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/").select("deal_id").distinct() #maybe add test?
deal_ids_list = list(deal_ids.select("deal_id").toPandas()["deal_id"])

# COMMAND ----------

#create the objects

lrmodel = LogisticRegression(maxIter=100, elasticNetParam=1, family="binomial")

paramGrid = ParamGridBuilder()\
    .addGrid(lrmodel.regParam, [0.1, 0.01, 0.001, 0.0001, 0.00001]) \
    .build()

tvs = TrainValidationSplit(estimator=lrmodel, #we could also do cross-validation, but that will take 5x longer
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# COMMAND ----------

# Run TrainValidationSplit, and choose the best regularization paramer.
for deal_id in deal_ids_list:
  training_data = market_train.withColumnRenamed(deal_id,'label')
  testing_data = market_test.withColumnRenamed(deal_id,'label')
  model = tvs.fit(training_data)
  print({"regularization" : model._java_obj.getRegParam(),"accuracy" : model.summary.accuracy,"intercept" : model.intercept})  
  print({variable_names[variable_number] : model.coefficients[variable_number.item()] for variable_number in model.coefficients.indices})

# COMMAND ----------

variable_names = {dic['idx'] : dic["name"] for dic in market_train.schema["features"].metadata["ml_attr"]["attrs"]["binary"]} #stores the variable names to use later. actiepunt: check of dit goed gaat