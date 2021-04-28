package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.mllib.regression.LabeledPoint;


/**
 * Hello world!
 */
public class SparkMLDataTypes {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF) ;

        SparkSession sparkSession = SparkSession
                .builder()
                .appName("Data types")
                .master("local[4]")
                .getOrCreate() ;

        Vector denseVector = new DenseVector(new double[]{1.0, 0.0, 3.2});
        Vector denseVector2 = Vectors.dense(1.0, 0.0, 3.2);

        Vector sparseVector = new SparseVector(3, new int[]{0, 2}, new double[]{1.0, 3.2});
        Vector sparseVector2 = Vectors.sparse(3, new int[]{0, 2}, new double[]{1.0, 3.2});

        System.out.println("Vector 1 (dense): " + denseVector) ;
        System.out.println("Vector 2 (dense): " + denseVector2) ;
        System.out.println("Vector 1 (sparse): " + sparseVector) ;
        System.out.println("Vector 2 (sparse): " + sparseVector2) ;

        LabeledPoint labeledPoint = new LabeledPoint(1.0, denseVector) ;
        LabeledPoint labeledPoint2 = new LabeledPoint(0.0, Vectors.sparse(5, new int[]{2, 4}, new double[]{5.2, 6.2})) ;

        System.out.println("Labeled point 1: " + labeledPoint) ;
        System.out.println("Labeled point 2: " + labeledPoint2) ;

        sparkSession.stop();
    }
}
