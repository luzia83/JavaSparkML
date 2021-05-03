package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class SVMTrainingDataframe {
    static Logger log = Logger.getLogger(SVMTrainingDataframe.class.getName());

    public static void main(String[] args) throws IOException {
        Logger.getLogger("org").setLevel(Level.OFF) ;

        // Create a SparkSession.
        SparkSession sparkSession = SparkSession
                .builder()
                .appName("SVM with dataframe. Training")
                .master("local[4]")
                .getOrCreate();

        // Read data
        Dataset<Row> trainingDataFrame = sparkSession
                .read()
                .format("libsvm")
                .load("data/classificationDataLibsvm.txt");

        LinearSVC linearSVC = new LinearSVC()
                .setMaxIter(10)
                .setRegParam(0.1) ;

        LinearSVCModel model = linearSVC.fit(trainingDataFrame) ;

        System.out.println("Coefficients: " + model.coefficients() +
                " Intercept: " + model.intercept());

        model.save("SVMModel");

        sparkSession.stop();
    }
}