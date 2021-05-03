package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

/**
 * Created by ajnebro on 12/11/15.
 */
public class SVMClassificationRDD {
    static Logger log = Logger.getLogger(SVMClassificationRDD.class.getName());

    public static void main(String[] args) {
        if (args.length < 1) {
            log.fatal("Syntax Error: The directory with the SVM Model is required")  ;
            throw new RuntimeException();
        }

        Logger.getLogger("org").setLevel(Level.OFF) ;

        SparkConf conf = new SparkConf()
                .setAppName("SVM Classifier Example")
                .setMaster("local[4]");

        SparkContext sc = new SparkContext(conf);

        String path = args[0] ;

        // Load model
        SVMModel model = SVMModel.load(sc, path);
        System.out.println(model) ;

        Vector v = new DenseVector(new double[]{1.1, 3.2}) ;
        System.out.println("Vector 1: " + v) ;
        System.out.println("Data 1: " + model.predict(v)) ;
        v = new DenseVector(new double[]{5.1, 1.4}) ;
        System.out.println("Vector 2: " + v) ;
        System.out.println("Data 2: " + model.predict(v)) ;
        v = new DenseVector(new double[]{5.2, 2}) ;
        System.out.println("Vector 3: " + v) ;
        System.out.println("Data 3: " + model.predict(v)) ;
        v = new DenseVector(new double[]{1, 4}) ;
        System.out.println("Vector 4: " + v) ;
        System.out.println("Data 4: " + model.predict(v)) ;
        v = new DenseVector(new double[]{3, 1.5}) ;
        System.out.println("Vector 5: " + v) ;
        System.out.println("Data 5: " + model.predict(v)) ;
    }
}
