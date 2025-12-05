from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Initialize Spark Session
spark = SparkSession.builder.appName("Exercise2_KMeans").getOrCreate()

# ---------------------------------------------------------
# Load the Wine Dataset
# ---------------------------------------------------------
# The provided file uses semi-colons ';' as delimiters
wine_YourFirstname1 = spark.read.option("header", True)\
    .option("delimiter", ";")\
    .option("inferSchema", True)\
    .csv("wine.csv") # Ensure wine.csv is in your working directory

# ---------------------------------------------------------
# Initial Investigation
# ---------------------------------------------------------
print("--- Column Names ---")
print(wine_YourFirstname1.columns)

print("\n--- Column Types ---")
wine_YourFirstname1.printSchema()

print("\n--- Basic Statistics (Mean, Stddev, Min, Max) ---")
wine_YourFirstname1.describe().show()

print("\n--- Quartiles (Approximate) ---")
# Calculating quartiles for a sample column (e.g., alcohol) as describe() does not give quartiles
for c in wine_YourFirstname1.columns:
    if c != 'quality': # skipping label if treating as categorical for this print
        quartiles = wine_YourFirstname1.approxQuantile(c, [0.25, 0.5, 0.75], 0.01)
        print(f"{c} Quartiles: 25%={quartiles[0]}, 50%={quartiles[1]}, 75%={quartiles[2]}")

print("\n--- Missing Values Count ---")
# Generating table showing number of missing values for each column
wine_YourFirstname1.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in wine_YourFirstname1.columns]).show()

print("\n--- Distinct Quality Values ---")
wine_YourFirstname1.select("quality").distinct().show()

print("\n--- Mean of Chemical Compositions by Quality ---")
wine_YourFirstname1.groupBy("quality").avg().show()

# ---------------------------------------------------------
# Feature Engineering (Vector Assembler)
# ---------------------------------------------------------
# Create a vector column 'feature_YourFirstname' containing specific columns
assembler = VectorAssembler(
    inputCols=["citric acid", "volatile acidity", "chlorides", "sulphates"],
    outputCol="feature_YourFirstname")

# Transform the dataframe to add the vector column
wine_YourFirstname = assembler.transform(wine_YourFirstname1)

# ---------------------------------------------------------
# Partitions and Caching
# ---------------------------------------------------------
# Spread dataframe across 3 partitions
wine_YourFirstname = wine_YourFirstname.coalesce(3)

# Cache the dataframe for performance during iterative training
wine_YourFirstname.cache()

# ---------------------------------------------------------
# K-Means Clustering (K=6)
# ---------------------------------------------------------
print("\n--- K-Means Clustering (K=6) ---")
kmeans_6 = KMeans().setK(6).setSeed(1).setFeaturesCol("feature_YourFirstname")
model_6 = kmeans_6.fit(wine_YourFirstname)

# Print cluster sizes (count of points in each cluster)
summary_6 = model_6.summary
print("Cluster Sizes (K=6): ", summary_6.clusterSizes)

# Print cluster centroids
print("Cluster Centroids (K=6):")
centers_6 = model_6.clusterCenters()
for center in centers_6:
    print(center)

# ---------------------------------------------------------
# K-Means Clustering (K=4)
# ---------------------------------------------------------
print("\n--- K-Means Clustering (K=4) ---")
kmeans_4 = KMeans().setK(4).setSeed(1).setFeaturesCol("feature_YourFirstname")
model_4 = kmeans_4.fit(wine_YourFirstname)

# Print cluster sizes
summary_4 = model_4.summary
print("Cluster Sizes (K=4): ", summary_4.clusterSizes)

# Print cluster centroids
print("Cluster Centroids (K=4):")
centers_4 = model_4.clusterCenters()
for center in centers_4:
    print(center)

spark.stop()