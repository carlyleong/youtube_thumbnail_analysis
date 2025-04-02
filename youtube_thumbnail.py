import os
import sys
import pymongo
import pandas as pd
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
import json
from pyspark.sql.functions import col as spark_col, udf, expr, when, isnan
from pyspark.sql.types import DoubleType, StringType, ArrayType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Set Python interpreter for PySpark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Set MLflow tracking URI (optional, use if you want to store experiments in a specific location)
mlflow.set_tracking_uri("file:./mlruns")

# Start MLflow experiment
mlflow.set_experiment("YouTube Thumbnail Analysis")

# MongoDB connection
MONGO_URI = 'mongodb+srv://carlyleong:732Rosewood@youtubethumb.d0byf.mongodb.net/'
DATABASE_NAME = 'thumbnail_analyzer'
COLLECTION_NAME = 'raw_videos_duplicate'

client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Fetch data from MongoDB
data_list = list(collection.find({}))

def train_and_evaluate_model(model, model_name, train_df, test_df):
    """Train a model with cross-validation and log metrics with MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Define parameter grid for each model type
        if isinstance(model, LinearRegression):
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam, [0.01, 0.1, 0.3]) \
                .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
                .build()
        elif isinstance(model, RandomForestRegressor):
            param_grid = ParamGridBuilder() \
                .addGrid(model.numTrees, [50, 100, 200]) \
                .addGrid(model.maxDepth, [5, 10, 15]) \
                .build()
        elif isinstance(model, GBTRegressor):
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxDepth, [3, 5, 7]) \
                .addGrid(model.maxIter, [50, 100, 150]) \
                .build()
        
        # Create CrossValidator
        evaluator = RegressionEvaluator(
            labelCol="viewCount", 
            predictionCol="prediction", 
            metricName="rmse"
        )
        
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3
        )
        
        # Fit CrossValidator
        print(f"\nTraining {model_name} with cross-validation...")
        cv_model = cv.fit(train_df)
        
        # Get best model
        model_fitted = cv_model.bestModel
        
        # Make predictions
        predictions = model_fitted.transform(test_df)
        
        # Evaluate metrics
        rmse = evaluator.setMetricName("rmse").evaluate(predictions)
        r2 = evaluator.setMetricName("r2").evaluate(predictions)
        mae = evaluator.setMetricName("mae").evaluate(predictions)
        
        # Log best parameters
        for param in model_fitted.extractParamMap():
            mlflow.log_param(param.name, model_fitted.getOrDefault(param))
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Create a simplified example without vector columns
        example = train_df.select(
            "_id",
            "viewCount",
            "likeCount",
            "commentCount",
            "sentiment_score",
            "like_view_ratio",
            "comment_view_ratio"
        ).limit(1)
        
        # Convert to pandas and drop any problematic columns
        example_pd = example.toPandas()
        
        # Log model with simplified example
        mlflow.spark.log_model(
            model_fitted, 
            f"model_{model_name}",
            input_example=example_pd
        )
        
        print(f"{model_name} Metrics:")
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")
        print(f"MAE: {mae}")
        
        return model_fitted, predictions

# Convert ObjectId to string for proper serialization
for doc in data_list:
    doc["_id"] = str(doc["_id"])

if data_list:
    # Convert to pandas DataFrame
    pdf = pd.json_normalize(data_list)
    
    # Convert complex types to strings
    for col in pdf.columns:
        pdf[col] = pdf[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

    # Convert numeric types
    for col in pdf.select_dtypes(include=["int64", "float64"]).columns:
        pdf[col] = pdf[col].astype(float)

    # Convert object types to strings
    for col in pdf.select_dtypes(include=['object']).columns:
        pdf[col] = pdf[col].astype(str)

    # Create Spark session
    ss = SparkSession.builder \
        .appName("MongoDB-PySpark Connection") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.pyspark.python", sys.executable) \
        .config("spark.pyspark.driver.python", sys.executable) \
        .getOrCreate()

    # Split data into vision and YouTube dataframes
    vision_df = pdf[['_id', 'analysis.text', 'analysis.labels',
                     'analysis.num_faces', 'analysis.dominant_colors']]
    yt_df = pdf.drop(['analysis.text', 'analysis.labels',
                     'analysis.num_faces', 'analysis.dominant_colors'], axis=1)

    # Create Spark DataFrames
    spark_vision_df = ss.createDataFrame(vision_df)
    spark_yt_df = ss.createDataFrame(yt_df)
    
    # Display information about the dataframes
    print("Columns in the DataFrame:")
    print(pdf.columns)
    
    print("\nVision DataFrame Sample:")
    spark_vision_df.show(2, truncate=False)
    
    print("\nYouTube DataFrame Sample:")
    spark_yt_df.show(2, truncate=False)
    
    print("\nYouTube DataFrame Schema:")
    spark_yt_df.printSchema()
    
    print("\nVision DataFrame Schema:")
    spark_vision_df.printSchema()

    # IMPORTANT: Properly select and alias columns
    # The column names have periods, so use backticks and provide aliases
    yt_df_red = spark_yt_df.select(
        spark_col("_id"),
        spark_col("`statistics.viewCount`").alias("viewCount"), 
        spark_col("`statistics.likeCount`").alias("likeCount"), 
        spark_col("`statistics.commentCount`").alias("commentCount")
    )
    
    # Check if we have the correct columns
    print("\nReduced YouTube DataFrame:")
    yt_df_red.printSchema()
    yt_df_red.show(5)

    vision_df_red = spark_vision_df.select(
        spark_col("_id"),
        spark_col("`analysis.labels`").alias("labels")
    )
    
    # Check if we have the correct columns
    print("\nReduced Vision DataFrame:")
    vision_df_red.printSchema()
    vision_df_red.show(5)

    # Define a more useful sentiment analysis UDF
    def sentiment_score(labels_json):
        try:
            # Parse the JSON string to get the actual labels
            labels = json.loads(labels_json)
            
            # Simple sentiment logic - could be improved with a real sentiment model
            positive_words = ["happiness", "active", "toy", "red", "orange", "smile", "beautiful"]
            negative_words = ["electronic device", "mobile phone", "technology", "screenshot"]
            
            # Count positive and negative matches
            pos_count = sum(1 for label in labels if any(pos in label.lower() for pos in positive_words))
            neg_count = sum(1 for label in labels if any(neg in label.lower() for neg in negative_words))
            
            # Calculate score from -1 to 1
            total = pos_count + neg_count
            if total > 0:
                return (pos_count - neg_count) / total
            return 0.0
        except:
            # If there's any error, return a neutral score
            return 0.0

    # Register UDF
    sentiment_udf = udf(sentiment_score, DoubleType())
    
    # Apply sentiment analysis
    vision_df_red = vision_df_red.withColumn("sentiment_score", sentiment_udf(spark_col("labels")))

    # Join dataframes
    print("\nJoining DataFrames...")
    joined_df = yt_df_red.join(vision_df_red, "_id", "inner")
    print(f"Joined DataFrame count: {joined_df.count()}")
    
    # Show the joined data
    print("\nJoined DataFrame Sample:")
    joined_df.show(5)

    # Verify we have numeric data for our model
    print("\nChecking data types:")
    joined_df.printSchema()
    
    # Convert columns to numeric and handle NaN values
    joined_df = joined_df.withColumn("viewCount", spark_col("viewCount").cast("double"))
    joined_df = joined_df.withColumn("likeCount", spark_col("likeCount").cast("double"))
    joined_df = joined_df.withColumn("commentCount", spark_col("commentCount").cast("double"))

    # Create engagement ratio features with safety checks
    joined_df = joined_df.withColumn(
        "like_view_ratio",
        when(spark_col("viewCount") > 0, spark_col("likeCount") / spark_col("viewCount"))
        .otherwise(0.0)
    )
    joined_df = joined_df.withColumn(
        "comment_view_ratio",
        when(spark_col("viewCount") > 0, spark_col("commentCount") / spark_col("viewCount"))
        .otherwise(0.0)
    )

    # Drop any remaining rows with null values
    joined_df = joined_df.na.fill(0.0)  # Fill any remaining NaN values with 0

    # Check for null values
    print("\nChecking for null values:")
    joined_df.select([spark_col(c).isNull().alias(c) for c in joined_df.columns]).show()

    # Optionally, drop rows with null values
    joined_df = joined_df.dropna(subset=["viewCount", "sentiment_score"])
    
    # Validate data before scaling
    print("\nChecking for NaN or null values in features:")
    for column in ["sentiment_score", "likeCount", "commentCount", "like_view_ratio", "comment_view_ratio"]:
        null_count = joined_df.filter(spark_col(column).isNull() | isnan(spark_col(column))).count()
        print(f"Null or NaN values in {column}: {null_count}")

    # Assemble features for ML with handleInvalid parameter
    assembler = VectorAssembler(
        inputCols=[
            "sentiment_score",
            "likeCount",
            "commentCount",
            "like_view_ratio",
            "comment_view_ratio"
        ],
        outputCol="features",
        handleInvalid="skip"  # This will skip any remaining invalid values
    )
    ml_df = assembler.transform(joined_df)
    
    # Verify feature vector creation
    print("\nFeature vector sample:")
    ml_df.select("features").show(5, truncate=False)

    # Add scaling with null check
    scaler = StandardScaler(
        inputCol="features", 
        outputCol="scaledFeatures",
        withStd=True,
        withMean=True
    )

    # Filter out any rows where features is null before scaling
    ml_df = ml_df.filter(spark_col("features").isNotNull())
    scaler_model = scaler.fit(ml_df)
    ml_df = scaler_model.transform(ml_df)

    # Split into training and test sets
    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(
            featuresCol="scaledFeatures",
            labelCol="viewCount",
            maxIter=100
        ),
        "Random Forest": RandomForestRegressor(
            featuresCol="scaledFeatures",
            labelCol="viewCount",
            numTrees=100
        ),
        "Gradient Boosting": GBTRegressor(
            featuresCol="scaledFeatures",
            labelCol="viewCount",
            maxIter=100
        )
    }
    
    # Train and evaluate all models
    results = {}
    for model_name, model in models.items():
        model_fitted, predictions = train_and_evaluate_model(
            model,
            model_name,
            train_df,
            test_df
        )
        results[model_name] = {
            'model': model_fitted,
            'predictions': predictions
        }
    
    # Compare predictions from different models
    print("\nSample Predictions from All Models:")
    for model_name, result in results.items():
        predictions = result['predictions']
        print(f"\n{model_name} Predictions:")
        predictions.select("_id", "viewCount", "prediction").show(5)
    
    # Find best performing model based on R2 score
    best_model = None
    best_r2 = -float('inf')
    evaluator = RegressionEvaluator(labelCol="viewCount", predictionCol="prediction", metricName="r2")
    
    for model_name, result in results.items():
        r2 = evaluator.evaluate(result['predictions'])
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
    
    print(f"\nBest performing model: {best_model} with R2 score: {best_r2}")

else:
    print("No data found in MongoDB collection.")

# Stop Spark session
ss.stop()

