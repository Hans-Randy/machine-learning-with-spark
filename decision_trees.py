from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("Exercise1_DecisionTree").getOrCreate()

# Load the data
path_to_data = "data/sample_libsvm_data.txt" 
df_hans_randy = spark.read.format("libsvm").load(path_to_data)

# Basic Investigation
print("Number of records:", df_hans_randy.count())
print("Number of columns:", len(df_hans_randy.columns))
df_hans_randy.printSchema()

# StringIndexer
labelIndexer_hans_randy = StringIndexer(inputCol="label", outputCol="indexedLabel_hans_randy").fit(df_hans_randy)

# VectorIndexer
featureIndexer_hans_randy = VectorIndexer(inputCol="features", outputCol="indexedFeatures_hans_randy", maxCategories=4).fit(df_hans_randy)

# Printout Features (Metadata check)
print("Input Column:", featureIndexer_hans_randy.getInputCol())
print("Output Column:", featureIndexer_hans_randy.getOutputCol())
print("Number of Features:", featureIndexer_hans_randy.numFeatures)
print("Category Maps:", featureIndexer_hans_randy.categoryMaps)

# Split Data
# Split data into 65% training and 35% testing
(training_hans_randy, testing_hans_randy) = df_hans_randy.randomSplit([0.65, 0.35])

# Create Decision Tree Estimator
DT_hans_randy = DecisionTreeClassifier(labelCol="indexedLabel_hans_randy", featuresCol="indexedFeatures_hans_randy")

# Create Pipeline
pipeline_hans_randy = Pipeline(stages=[labelIndexer_hans_randy, featureIndexer_hans_randy, DT_hans_randy])

# Fit the Model
model_hans_randy = pipeline_hans_randy.fit(training_hans_randy)

# Predictions
predictions_hans_randy = model_hans_randy.transform(testing_hans_randy)

# Print Schema of Predictions
predictions_hans_randy.printSchema()

# Evaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel_hans_randy", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions_hans_randy)
test_error = 1.0 - accuracy

print("Accuracy: " + str(accuracy))
print("Test Error: " + str(test_error))

# Show Predictions
predictions_hans_randy.select("prediction", "indexedLabel_hans_randy", "features").show(10)