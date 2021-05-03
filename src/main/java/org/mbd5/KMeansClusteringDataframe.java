package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeansModel;
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

public class KMeansClusteringDataframe {
    static Logger log = Logger.getLogger(KMeansClusteringRDD.class.getName());

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF) ;

        if (args.length != 1) {
            log.fatal("Syntax Error: directory with the model missing")  ;
            throw new RuntimeException();
        }

        // Create a SparkSession.
        SparkSession sparkSession = SparkSession
                .builder()
                .appName("JavaKMeansExample")
                .master("local[4]")
                .getOrCreate();

        KMeansModel model = KMeansModel.load(args[0]) ;
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
                RowFactory.create(1.0, v1),
                RowFactory.create(2.0, v2),
                RowFactory.create(3.0, v3),
                RowFactory.create(4.0, v4)) ;

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        Dataset<Row> testDataFrame = sparkSession.createDataFrame(testData, schema) ;
        Dataset<Row> results = model.transform(testDataFrame) ;

        results.printSchema();
        results.show();

        Dataset<Row> rows = results.select("features", "label", "prediction");
        for (Row r: rows.collectAsList()) {
            System.out.println("Vector: " + r.get(0) + ", label: " + r.get(1) + ", prediction: " + r.get(2));
        }


        sparkSession.stop();
    }
}