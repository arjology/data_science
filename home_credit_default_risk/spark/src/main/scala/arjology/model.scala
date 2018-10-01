package arjology

import com.typesafe.config.ConfigFactory

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SparkSession, DataFrame}

import org.apache.spark.sql.functions.{sum, col}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, Imputer}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.StringIndexer

import arjology.schema.{trainSchema, testSchema}

import java.io.File

object model {
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

    val trainDFNoSpaces = spark.read.option("header", "true").schema(trainSchema).csv(trainInput)
    val testDFNoSpaces = spark.read.option("header", "true").schema(testSchema).csv(trainInput)

    trainIndexedDF =

  }
}
