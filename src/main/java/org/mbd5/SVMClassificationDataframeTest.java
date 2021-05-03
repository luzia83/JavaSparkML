package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.types.DataTypes.DoubleType;

public class SVMClassificationDataframeTest {
    static Logger log = Logger.getLogger(SVMClassificationDataframeTest.class.getName());

    public static void main(String[] args)  {
        Logger.getLogger("org").setLevel(Level.OFF) ;

        if (args.length != 1) {
            log.fatal("Syntax Error: directory with the model missing")  ;
            throw new RuntimeException();
        }

        SparkSession sparkSession = SparkSession
                .builder()
                .appName("SVM Classification")
                .master("local[4]")
                .getOrCreate();

        LinearSVCModel model = LinearSVCModel.load(args[0]) ;
        System.out.println("Model loaded") ;

        Vector v1 = new DenseVector(new double[]{1.1, 3.2}) ;
        System.out.println("Vector 1: " + v1) ;
        Vector v2 = new DenseVector(new double[]{5.1, 1.4}) ;
        System.out.println("Vector 2: " + v2) ;
        Vector v3 = new DenseVector(new double[]{5.2, 2}) ;
        System.out.println("Vector 3: " + v3) ;
        Vector v4 = new DenseVector(new double[]{1, 4}) ;
        System.out.println("Vector 4: " + v4) ;

        // Prepare training data.
        List<Row> testData = Arrays.asList(
                RowFactory.create(0.0, v1),
                RowFactory.create(1.0, v2),
                RowFactory.create(1.0, v3),
                RowFactory.create(0.0, v4)) ;

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        Dataset<Row> testDataFrame = sparkSession.createDataFrame(testData, schema) ;

        Dataset<Row> results = model.transform(testDataFrame) ;

        System.out.println(v1 + ": " + model.predict(v1));
        System.out.println(v2 + ": " + model.predict(v2));
        System.out.println(v3 + ": " + model.predict(v3));
        System.out.println(v4 + ": " + model.predict(v4));

        results.printSchema();
        results.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator() ;
        evaluator.setMetricName("accuracy") ;
        evaluator.setLabelCol("label") ;
        evaluator.setPredictionCol("prediction") ;

        double accuracy = evaluator.evaluate(results) ;
        System.out.println("Test error: " + (1.0 - accuracy)) ;

        sparkSession.stop() ;
    }
}
