package org.mbd5;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;

import java.util.Arrays;


/**
 * Program illustrating basic statistical features of SparkML using RDDs
 *
 * @author Antonio J. Nebro
 */
public class StatisticalOperations {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF);

        SparkConf sparkConf = new SparkConf()
                .setAppName("Spark basic statistics")
                .setMaster("local[8]");

        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        JavaRDD<Vector> denseVectorRDD = sparkContext.parallelize(
                Arrays.asList(
                        Vectors.dense(1.0, 2.0, 3.0),
                        Vectors.dense(4.0, 5.0, 6.0),
                        Vectors.dense(8.0, 9.0, 0.0)
                )
        );

        JavaRDD<Vector> sparseVectorRDD = sparkContext.parallelize(
                Arrays.asList(
                        Vectors.sparse(3, new int[]{0, 1, 2}, new double[]{1.0, 2.0, 3.0}),
                        Vectors.sparse(3, new int[]{0, 1, 2}, new double[]{4.0, 5.0, 6.0}),
                        Vectors.sparse(3, new int[]{0, 1}, new double[]{8.0, 9.0})
                )
        );

        System.out.println("Dense vector:");
        denseVectorRDD.collect().forEach(System.out::println);

        System.out.println("Sparse vector:");
        sparseVectorRDD.collect().forEach(System.out::println);

        MultivariateStatisticalSummary summary = Statistics.colStats(denseVectorRDD.rdd());

        System.out.println("Dense vector statistics: ");
        System.out.println("Mean    : " + summary.mean());
        System.out.println("Variance: " + summary.variance());
        System.out.println("NonZeros: " + summary.numNonzeros());
        System.out.println("Count   : " + summary.count());
        System.out.println("Min     : " + summary.min());
        System.out.println("Max     : " + summary.max());

        summary = Statistics.colStats(sparseVectorRDD.rdd());
        System.out.println("Sparse vector statistics: ");
        System.out.println("Mean    : " + summary.mean());
        System.out.println("Variance: " + summary.variance());
        System.out.println("NonZeros: " + summary.numNonzeros());
        System.out.println("Count   : " + summary.count());
        System.out.println("Min     : " + summary.min());
        System.out.println("Max     : " + summary.max());
    }
}
