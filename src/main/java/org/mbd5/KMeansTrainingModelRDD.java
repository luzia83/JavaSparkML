package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.Arrays;

/**
 * Created by ajnebro on 16/11/15.
 */
public class KMeansTrainingModelRDD {
    static Logger log = Logger.getLogger(KMeansTrainingModelRDD.class.getName());

    public static void main(String[] args) {

        Logger.getLogger("org").setLevel(Level.OFF) ;

        SparkConf conf;
        conf = new SparkConf().setAppName("K-means Example");
        JavaSparkContext sc = new JavaSparkContext(conf);

        if (args.length != 1) {
            log.fatal("Syntax Error: data file missing");
            throw new RuntimeException();
        }
        String path = args[0];

        // Load and parse data
        JavaRDD<String> data = sc.textFile(path);
        data.cache();

        JavaRDD<Vector> parsedData = data.map(
                s -> Vectors.dense(Arrays.stream(s.split(" "))
                        .mapToDouble(Double::parseDouble)
                        .toArray()));

        parsedData.cache();

        // Cluster the data into two classes using KMeans
        long startTime = System.currentTimeMillis() ;

        int numClusters = 2;
        int numIterations = 200;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

        long computingTime = System.currentTimeMillis() - startTime ;

        double WSSSE = clusters.computeCost(parsedData.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

        System.out.println("Computing time: " + computingTime) ;


        // Save and load model
        clusters.save(sc.sc(), "kmeansModelPath");
    }

}
