package arjology

import java.io.File

import com.typesafe.config.ConfigFactory

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SparkSession, DataFrame}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.GBTClassifier

//import org.apache.spark.ml.evaluation.
//import org.apache.spark.mllib.evaluation._

object credit_default {
  def main(args: Array[String]): Unit = {

    val config = ConfigFactory.parseFile(new File("gbt.conf"))
    val testIinput = config.getString("spark.test")
    val trainInput = config.getString("spark.train")
    val appName = config.getString("spark.appName")
    val sparkMaster = config.getString("spark.master")

    val t0 = System.nanoTime()


    val conf = new SparkConf().setAppName(appName)


    lazy val spark: SparkSession = {
      SparkSession
        .builder()
        .master(sparkMaster)
        .appName(appName)
        .getOrCreate()
    }
    spark.sparkContext.setLogLevel("WARN")

    val testDF: DataFrame = spark.read
      .option("header", "true")
      .csv(testIinput)

    val trainDF: DataFrame = spark.read
      .option("header", "true")
      .csv(trainInput)

    val feature_cols = trainDF.columns.diff(List("SK_ID_CURR", "TARGET"))
    val assembler = new VectorAssembler().setInputCols(feature_cols).setOutputCol("features")
    val full_model_input = assembler.transform(trainDF)
    val classifier = new GBTClassifier().setLabelCol("TARGET").setFeaturesCol("features")

  }
}
