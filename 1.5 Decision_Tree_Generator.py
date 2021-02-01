# Databricks notebook source
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
from pyspark.ml import Pipeline

# COMMAND ----------

def decision_tree_generator(training_data,deal_id):  
  ####In: 
  #A training data set
  #The deal_id you want to generate a decision tree for
  
  ####Out
  #The tree is saved
  #An update message is outputted
  
  training_data = training_data.withColumnRenamed(deal_id,'label')
  dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",
                              maxDepth=8,impurity="entropy",
                             algo="classification",numClasses=2)
  model = dt.fit(training_data)
  model.write().overwrite().save(f"s3://rtl-databricks-datascience/lpater/decision_trees/{deal_id}/")
  output_message = "Saved a Decision Tree for "+deal_id+"."
  
  return model
#for visualization: https://github.com/parrt/dtreeviz

# COMMAND ----------

def decision_tree_evaluator(test_data,deal_id):  
  ####In: 
  #A testing data set
  #The deal_id you want to test a tree for
  #NB: The model tree to be already saved to the cloud 
  
  ####Out
  #An update message is outputted
  #an evaluator
  
  
  model = DecisionTreeClassificationModel.load(f"s3://rtl-databricks-datascience/lpater/decision_trees/{deal_id}/")
  predictions = model.transform(test_data.withColumnRenamed(deal_id,'label'))
  # compute accuracy on the test set
  evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction",
                                                metricName="areaUnderPR") #alternatively, use areaUnderPR to get the precision-recall curve instead of the accuracy
  
  accuracy = evaluator.evaluate(predictions)
  print("Decision Tree area under PR " + deal_id +  " = " + str(accuracy))
  
  return evaluator

# COMMAND ----------



# COMMAND ----------

market_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_train.parquet/")
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")

deal_ids = list(spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/").groupBy("deal_id").count().orderBy('count', ascending=False).select("deal_id").toPandas()["deal_id"]) #create a list of deal_ids to select from, ordered by how common they are

variable_names = {dic['idx'] : dic["name"] for dic in market_train.schema["features"].metadata["ml_attr"]["attrs"]["binary"]} #stores the variable names to use later. actiepunt: check of dit goed gaat

# COMMAND ----------

for deal in deal_ids:
  decision_tree_generator(market_train,deal)
  decision_tree_evaluator(market_test,deal)

# COMMAND ----------

#defaults
for deal in deal_ids:
  decision_tree_generator(market_train,deal)
  decision_tree_evaluator(market_test,deal)

# COMMAND ----------

#maxDepth=10, maxBins=128,maxMemoryInMB=2048,seed=1
for i in range(5):
  decision_tree_generator(market_train,deal_ids[i])
  decision_tree_evaluator(market_test,deal_ids[i])
  
#worse results than the defaults, probably because of overfitting

# COMMAND ----------

#Creates and saves all trees models



for deal_id in deal_ids:
  market_predictions = market_test.select("features","market_guid")
  model = DecisionTreeClassificationModel.load(f"s3://rtl-databricks-datascience/lpater/decision_trees/{deal_id}/")
  market_predictions = model.transform(market_predictions.withColumnRenamed(deal_id,'label'))
  market_predictions.groupBy("probability").count().show(100,False)

# COMMAND ----------

print(deal_ids)

# COMMAND ----------

#Prints the area under the Precision-Recall curve for every model
for deal_id in deal_ids_list:
  print(logistic_regression_evaluator(test_data=market_test,deal_id=deal_id))