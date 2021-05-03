package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class KMeansTrainingDataframe {
    static Logger log = Logger.getLogger(KMeansTrainingDataframe.class.getName());

    public static void main(String[] args) throws IOException {
        Logger.getLogger("org").setLevel(Level.OFF) ;

        // Create a SparkSession.
        SparkSession sparkSession = SparkSession
                .builder()
                .appName("JavaKMeansExample")
                .master("local[4]")
                .getOrCreate();

        // Read data
        Dataset<Row> dataset = sparkSession
                .read()
                .format("libsvm")
                .load("data/classificationdatalibsvm.txt");

        long startTime = System.currentTimeMillis() ;

        // Trains a k-means model.
        KMeans kmeans = new KMeans()
                .setK(2)
                .setMaxIter(100);

        KMeansModel model = kmeans.fit(dataset);
        long computingTime = System.currentTimeMillis() - startTime ;

        // Evaluate clustering by computing Within Set Sum of Squared Errors.
        //double WSSSE = model.computeCost(dataset);
        //System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

        // Shows the result.
        Vector[] centers = model.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }

        model.save("KNNModel");

        System.out.println("Computing time: " + computingTime) ;

        sparkSession.stop();
    }
}