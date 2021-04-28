package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * @author Antonio J. Nebro
 */
public class ReadLibSVMDataframe {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF) ;

        SparkSession sparkSession = SparkSession
                .builder()
                .appName("SparkML. Read LibSVM file")
                .master("local[8]")
                .getOrCreate();

        Dataset<Row> dataFrame = sparkSession
                .read()
                .format("libsvm")
                .load("data/classificationDataLibsvm.txt") ;

        dataFrame.printSchema();
        dataFrame.show(20);

        sparkSession.stop();
    }
}