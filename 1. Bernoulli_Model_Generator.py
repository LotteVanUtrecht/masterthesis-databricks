# Databricks notebook source
# MAGIC %md # Logistic Regression & Naive Bayes model generator
# MAGIC 
# MAGIC This notebook currently does the following things:
# MAGIC 1. Loads Parquet files (created using the SpotX data prep notebook and processed in the Custom_Fields_Processor notebook)
# MAGIC 2. Run and save desired models for a single deal_id for any number of desired features
# MAGIC 3. Shows some attributes for all models: the intercept, coefficients, accuracy and area under the Precision-Recall curve.

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
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder#Estimator
from pyspark.ml import Pipeline

# COMMAND ----------

#mount my folder
#dbutils.fs.mount("s3a://rtl-databricks-datascience/lpater/", "/mnt/lotte")

# COMMAND ----------

def naive_bayes_generator(training_data,deal_id):  
  ####In: 
  #A training data set, as generated by data_prep() 
  #The deal_id you want to generate a model for
  
  ####Out
  #The model is saved
  #An update message is outputted
  
  training_data = training_data.withColumnRenamed(deal_id,'label')
  model = NaiveBayes(smoothing=10,modelType="bernoulli")
  model = model.fit(training_data)
  model.write().overwrite().save(f"s3://rtl-databricks-datascience/lpater/naive_bayes/{deal_id}/")
  output_message = "Saved a Naive Bayes model for "+deal_id+"."
  
  #sea also: https://spark.apache.org/docs/latest/ml-classification-regression.html
  return output_message


# COMMAND ----------

def logistic_regression_generator(training_data,deal_id):  
  ####In: 
  #A training data set, as generated by data_prep() 
  #The deal_id you want to generate a model for
  
  ####Out
  #The model is saved
  #An update message is outputted
  
  training_data = training_data.withColumnRenamed(deal_id,'label')
  model = LogisticRegression(maxIter=100, regParam=0.0001, elasticNetParam=1, family="binomial")
  model = model.fit(training_data)
  model.write().overwrite().save(f"s3://rtl-databricks-datascience/lpater/logistic_regression/{deal_id}/")
  output_message = "Saved a Logistic Regression model for "+deal_id+"."
  
  #see also: https://spark.apache.org/docs/latest/ml-classification-regression.html
  
  #note: this currently uses LASSO to select parameters
  return output_message

# COMMAND ----------

def naive_bayes_evaluator(test_data,deal_id):  
  ####In: 
  #A testing data set, as generated by data_prep() 
  #The deal_id you want to test a model for
  #NB: The model has to be already saved to the cloud 
  
  ####Out
  #An update message is outputted
  #an evaluator
  
  
  model = NaiveBayesModel.load(f"/mnt/lotte/naive_bayes/{deal_id}/")
  predictions = model.transform(test_data.withColumnRenamed(deal_id,'label'))
  # compute accuracy on the test set
  evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                metricName="accuracy") #alternatively, use AreaUnderPR to get the precision-recall curve instead of the accuracy
  accuracy = evaluator.evaluate(predictions)
  print("Naive Bayes test accuracy for " + deal_id +  " = " + str(accuracy))
  
  return evaluator

# COMMAND ----------

def logistic_regression_evaluator(test_data,deal_id):  
  ####In: 
  #A testing data set, as generated by data_prep() 
  #The deal_id you want to test a model for
  #NB: The model has to be already saved to the cloud 
  
  ####Out
  #An update message is outputted
  #an evaluator
  
  
  model = LogisticRegressionModel.load(f"/mnt/lotte/logistic_regression/{deal_id}/")
  predictions = model.transform(test_data.withColumnRenamed(deal_id,'label'))
  # compute accuracy on the test set
  evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction",
                                                metricName="areaUnderPR") #alternatively, use areaUnderPR to get the precision-recall curve instead of the accuracy
  
  accuracy = evaluator.evaluate(predictions)
  print("Logistic Regression area under PR " + deal_id +  " = " + str(accuracy))
  
  return evaluator

# COMMAND ----------

market_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_train.parquet/")
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")
market_train.cache()
market_test.cache()

deal_ids = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/").select("deal_id").distinct() #maybe add test?
deal_ids_list = list(deal_ids.select("deal_id").toPandas()["deal_id"])

variable_names = {dic['idx'] : dic["name"] for dic in market_train.schema["features"].metadata["ml_attr"]["attrs"]["binary"]} #stores the variable names to use later. actiepunt: check of dit goed gaat

# COMMAND ----------

#Creates and saves all logistic regression models
#

for deal_id in deal_ids_list:
  logistic_regression_generator(training_data=market_train,deal_id=deal_id)


# COMMAND ----------

#Prints the area under the Precision-Recall curve for every model
for deal_id in deal_ids_list:
  print(logistic_regression_evaluator(test_data=market_test,deal_id=deal_id))

# COMMAND ----------

#prints the intercept and coefficients for each model

for deal_id in deal_ids_list:
  temp_model = LogisticRegressionModel.load("/mnt/lotte/logistic_regression/"+deal_id+"/")
  print({"intercept" : temp_model.intercept})
  print({variable_names[variable_number] : temp_model.coefficients[variable_number.item()] for variable_number in temp_model.coefficients.indices})

# COMMAND ----------

#for deal_id in deal_ids_list:
#  temp_model = LogisticRegressionModel.load("/mnt/lotte/logistic_regression/"+deal_id+"/")
#  print({"intercept" : temp_model.intercept})
#  print({variable_names[variable_number] : temp_model.coefficients[variable_number.item()] for variable_number in temp_model.coefficients.indices})