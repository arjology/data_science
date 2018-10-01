package arjology

import com.typesafe.config.ConfigFactory

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml.feature.{VectorAssembler, Imputer}
import org.apache.spark.ml.classification.GBTClassifier

import arjology.schema.{trainSchema, testSchema, encodeStringColumns}

import java.io.File

object model {

  def main(args: Array[String]): Unit = {

    val config = ConfigFactory.parseFile(new File("homeDefault.conf"))
    val testIinput = config.getString("spark.test")
    val trainInput = config.getString("spark.train")
    val appName = config.getString("spark.appName")
    val sparkMaster = config.getString("spark.master")

    val conf = new SparkConf().setAppName(appName)

    lazy val spark: SparkSession = {
      SparkSession
        .builder()
        .master(sparkMaster)
        .appName(appName)
        .getOrCreate()
    }
    spark.sparkContext.setLogLevel("WARN")

    val trainDFNoSpaces = spark.read.option("header", "true").schema(trainSchema).csv(trainInput)
    val testDFNoSpaces = spark.read.option("header", "true").schema(testSchema).csv(trainInput)

    val trainIndexedDF = encodeStringColumns(trainDFNoSpaces)
    val testIndexedDF = encodeStringColumns(testDFNoSpaces)

    val imputer = new Imputer().setInputCols(trainIndexedDF.columns)
      .setOutputCols(trainIndexedDF.columns).setStrategy("mean")
    val trainIndexedDFImputed = imputer.fit(trainIndexedDF).transform(trainIndexedDF)
    val testIndexedDFImputed = imputer.fit(testIndexedDF).transform(testIndexedDF)

    val feature_cols = trainIndexedDFImputed.columns.diff(List("SK_ID_CURR", "TARGET"))
    val assembler = new VectorAssembler().setInputCols(feature_cols).setOutputCol("FEATS")
    val full_model_input = assembler.transform(trainIndexedDFImputed.drop("SK_ID_CURR"))
    val classifier = new GBTClassifier().setLabelCol("TARGET").setFeaturesCol("FEATS").setMaxDepth(2)
      .setSubsamplingRate(0.8)
      .setLossType("logistic")
      .setSeed(7)
    val Array(training, test) = full_model_input.randomSplit(Array(0.7, 0.3))

    val t0 = System.currentTimeMillis()
    val model = classifier.fit(training)
    val t1 = System.currentTimeMillis()
    println(s"Fit phase took ${(t1 - t0)/1000} seconds")

  }
}
