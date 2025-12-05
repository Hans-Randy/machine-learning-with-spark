from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("Exercise1_DecisionTree").getOrCreate()

# ---------------------------------------------------------
# Load the data
# ---------------------------------------------------------
# Loading sample_libsvm_data.txt with "libsvm" format
# Note: Ensure the path matches your specific environment path
path_to_data = "/home/centos/data/sample_libsvm_data.txt" 
df_YourFirstname = spark.read.format("libsvm").load(path_to_data)

# ---------------------------------------------------------
# Basic Investigation
# ---------------------------------------------------------
print("Number of records:", df_YourFirstname.count())
print("Number of columns:", len(df_YourFirstname.columns))
df_YourFirstname.printSchema()

# ---------------------------------------------------------
# StringIndexer
# ---------------------------------------------------------
# Index labels, adding metadata to the label column.
labelIndexer_YourFirstname = StringIndexer(inputCol="label", outputCol="indexedLabel_YourFirstname").fit(df_YourFirstname)

# ---------------------------------------------------------
# VectorIndexer
# ---------------------------------------------------------
# Automatically identify categorical features and index them.
# We set maxCategories to 4.
featureIndexer_YourFirstname = VectorIndexer(inputCol="features", outputCol="indexedFeatures_YourFirstname", maxCategories=4).fit(df_YourFirstname)

# ---------------------------------------------------------
# Printout Features (Metadata check)
# ---------------------------------------------------------
# Investigating the results of the VectorIndexer
print("Input Column:", featureIndexer_YourFirstname.getInputCol())
print("Output Column:", featureIndexer_YourFirstname.getOutputCol())
print("Number of Features:", featureIndexer_YourFirstname.numFeatures)
# Accessing category maps from the metadata
# Note: This might be empty if no features met the maxCategories criteria in this specific dataset
print("Category Maps:", featureIndexer_YourFirstname.categoryMaps)

# ---------------------------------------------------------
# Split Data
# ---------------------------------------------------------
# Split data into 65% training and 35% testing
(training_YourFirstname, testing_YourFirstname) = df_YourFirstname.randomSplit([0.65, 0.35])

# ---------------------------------------------------------
# Create Decision Tree Estimator
# ---------------------------------------------------------
# Using the indexed labels and features
DT_YourFirstname = DecisionTreeClassifier(labelCol="indexedLabel_YourFirstname", featuresCol="indexedFeatures_YourFirstname")

# ---------------------------------------------------------
# Create Pipeline
# ---------------------------------------------------------
# Pipeline stages: LabelIndexer -> FeatureIndexer -> DecisionTree
pipeline_YourFirstname = Pipeline(stages=[labelIndexer_YourFirstname, featureIndexer_YourFirstname, DT_YourFirstname])

# ---------------------------------------------------------
# Fit the Model
# ---------------------------------------------------------
model_YourFirstname = pipeline_YourFirstname.fit(training_YourFirstname)

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------
predictions_YourFirstname = model_YourFirstname.transform(testing_YourFirstname)

# ---------------------------------------------------------
# Print Schema of Predictions
# ---------------------------------------------------------
predictions_YourFirstname.printSchema()

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel_YourFirstname", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions_YourFirstname)
test_error = 1.0 - accuracy

print("Accuracy: " + str(accuracy))
print("Test Error: " + str(test_error))

# ---------------------------------------------------------
# Show Predictions
# ---------------------------------------------------------
# Select example rows to display.
predictions_YourFirstname.select("prediction", "indexedLabel_YourFirstname", "features").show(10)