# Predicting Taxi Fares in NYC

## Requirements

This project uses Spark 2.3.1 along with XGBoost. The following JARs are required
* [xgboost4j](https://mvnrepository.com/artifact/ml.dmlc/xgboost4j/0.72)
* [xgboost4j-spark](https://mvnrepository.com/artifact/ml.dmlc/xgboost4j-spark/0.72)
* [PySpark XGBoost](https://github.com/dmlc/xgboost/files/2161553/sparkxgb.zip)

## Assumptions for building the model

* Trip distance: increase in distance travelled should be reflected in an increased price.
* Time of travel: peak travel times will cost higher fares, along with difference in morning/afternoon fares.
* Day of travel: there may exist a difference in fares on weekends versus weekdays.
* Weather conditions: averse weather may lead to greater demand i.e. higher fares.
* Airport trips: these usually have fixed fares.
* Area: different neighborhoods might have different associated pick-up or drop-off costs.
* Availability: taxi availability will of course affect fares.

We can plot distributions of the various columns and see if they differentiate well between the fare amounts