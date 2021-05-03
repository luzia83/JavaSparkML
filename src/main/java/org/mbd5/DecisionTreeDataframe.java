package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DecisionTreeDataframe {
    static Logger log = Logger.getLogger(DecisionTreeDataframe.class.getName());

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF);

        // Create a SparkSession.
        SparkSession sparkSession =
                SparkSession.builder()
                        .appName("Decision tree with dataframes example")
                        .master("local[2]")
                        .getOrCreate();

        Dataset<Row> data_frame = sparkSession
                .read()
                .format("libsvm")
                .load("data/wine.scale") ;

        System.out.println("Data read") ;
        data_frame.printSchema();
        data_frame.show() ;

        // Prepare training and test data.
        Dataset<Row>[] splits = data_frame.randomSplit(new double[] {0.8, 0.2}, 12345);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        System.out.println("Training data: " + training.count() + " elements") ;
        training.printSchema();
        training.show() ;

        System.out.println("Test data: " + test.count() + " elements") ;
        test.printSchema();
        test.show() ;

        DecisionTreeClassifier classifier = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features");

        DecisionTreeClassificationModel model = classifier.fit(training) ;

        Dataset<Row> prediction = model.transform(test) ;
        System.out.println("Prediction: " + prediction.count() + " elements") ;
        prediction.printSchema();
        prediction.show() ;

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy") ;

        double accuracy = evaluator.evaluate(prediction) ;
        System.out.println("Test error = " + (1.0 - accuracy)) ;


        sparkSession.stop();
    }
}