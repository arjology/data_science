from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level="INFO")
log = logging.getLogger('boston_housing')
log.setLevel(20)  # Set logging level to WARN

log.info("Setting up spark context and loading housing dataframe")
sc= SparkContext()
sc.setLogLevel("WARN")
sqlContext = SQLContext(sc)
house_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('file:///home/arjang/data_science/data/boston_housing/train.csv')
house_test_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('file:///home/arjang/data_science/data/boston_housing/test.csv')
house_df.take(1)

house_df.cache()
log.info("house_df schema:")
house_df.printSchema()

log.info("Descrpiptive analytics")
house_df.describe().toPandas().transpose()

log.info("Correlation between multiple variables")
import six
for i in house_df.columns:
    if not( isinstance(house_df.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to MV for ", i, house_df.stat.corr('medv',i))


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = [ 'crim',
			'zn',
			'indus',
			'chas',
			'nox',
			'rm',
			'age',
			'dis',
			'rad',
			'tax',
			'ptratio',
			'black',
			'lstat'], outputCol = 'features')

vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['features', 'medv'])
splits = vhouse_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

log.info("""\n
#------------------------------------------------------------
# Linear Regression 
# 
#------------------------------------------------------------
""")


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='medv', maxIter=20, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

train_df.describe().show()

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","medv","features").show(5)

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",\
        labelCol="medv", metricName="r2")
print("R squared (R2) on test data = %g" %
        lr_evaluator.evaluate(lr_predictions))

test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" %
        test_result.rootMeanSquaredError)

print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()

predictions = lr_model.transform(test_df)
predictions.select("prediction", "medv", "features").show()


log.info("""\n
#------------------------------------------------------------
# Decision tree regression 
#
#------------------------------------------------------------
""")

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'medv')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(
        labelCol="medv", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

print("Feature importance", dt_model.featureImportances)
print(house_df.take(1))


log.info("""\n
#------------------------------------------------------------
# Gradient-boosted tree regression 
# 
#------------------------------------------------------------
""")

from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'medv', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'medv', 'features').show(5)

gbt_evaluator = RegressionEvaluator(labelCol='medv', predictionCol='prediction', metricName='rmse')
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

