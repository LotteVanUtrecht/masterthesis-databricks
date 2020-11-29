# Custom_Fields Processor 
# The goal of this notebook is to a) do some analysis on the Custom_Fields part of the market file and b) to develop a function that takes the custom fields out of their own column and into new attributes.

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

variable_names = {dic['idx'] : dic["name"] for dic in market_train.schema["features"].metadata["ml_attr"]["attrs"]["binary"]}

# COMMAND ----------

def logistic_regression_calibration(test_data,deal_id):  
  ####In: 
  #A testing data set, as generated by data_prep() 
  #The deal_id you want to test a model for
  #NB: The model has to be already saved to the cloud 
  
  ####Out
  #A list containing:
  #1) the true # of bids in the test set
  #2) the sum of all bid probabilities for the test set
  #3) the # of times the model predicts a probability >.5
  
  #https://stackoverflow.com/questions/54354915/pyspark-aggregate-sum-vector-element-wise
  
  model = LogisticRegressionModel.load(f"/mnt/lotte/logistic_regression/{deal_id}/")
  predictions = model.transform(test_data.withColumnRenamed(deal_id,'label').select("label","features"))
  
  predictions = predictions.select("label",ith("probability", f.lit(1)),"prediction") #the ith function selects the probability of category 1, i.e. the probability of bidding
  col_names = ["true bids","bid probabilities","bid predictions"]
  predictions = predictions.toDF(*col_names)
  
  probabilities = predictions.agg(f.sum("true bids"),f.sum("bid probabilities"),f.sum("bid predictions")).collect()
  
  return probabilities

def ith_(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None

ith = f.udf(ith_, DoubleType())

# COMMAND ----------

#Import data, load in cache for easy working
#bids_train = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_train.parquet/")
market_test = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_test.parquet/")
market_test.cache()

deal_ids = spark.read.parquet("s3://rtl-databricks-datascience/lpater/processed_data/bids_train.parquet/").select("deal_id").distinct() #maybe add test?
deal_ids_list = list(deal_ids.select("deal_id").toPandas()["deal_id"])

# COMMAND ----------

for deal_id in deal_ids_list:
  print([deal_id,logistic_regression_calibration(test_data=market_test,deal_id=deal_id)])
  
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

market_predictions.select("bidders","bids","max_bid","true_winning_deal_id","true_winning_max_bid")\
  .write.mode('overwrite')\
  .parquet("s3://rtl-databricks-datascience/lpater/processed_data/market_predictions.parquet/")
