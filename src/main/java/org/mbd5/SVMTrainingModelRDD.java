package org.mbd5;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

/**
 * Created by ajnebro on 12/11/15.
 */
public class SVMTrainingModelRDD {
    static Logger log = Logger.getLogger(SVMTrainingModelRDD.class.getName());

    public static void main(String[] args) {
        if (args.length < 1) {
            log.fatal("Syntax Error: there must be one argument (a libsvm data file)");
            throw new RuntimeException();
        }

        Logger.getLogger("org").setLevel(Level.OFF) ;

        SparkConf sparkConf = new SparkConf()
                .setAppName("SVM Classifier Example")
                .setMaster("local[4]");

        SparkContext sparkContext = new SparkContext(sparkConf);

        String path = args[0];
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sparkContext, path).toJavaRDD();

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> training = data.sample(false, 0.6, 11L);
        training.cache();

        JavaRDD<LabeledPoint> test = data.subtract(training);
        test.cache() ;

        System.out.println("Number of tranining points: " + training.count()) ;
        System.out.println("Number of test points: " + test.count()) ;

        // Run training algorithm to build the model.
        int numIterations = 100;
        final SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        // Clear the prediction threshold so the model will return probabilities
        model.clearThreshold();

        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(
                point -> {
                    System.out.println(point.features() + " ") ;
                    Double score = model.predict(point.features());
                    System.out.println("Score: " + score + ". Label: " + point.label());
                    return new Tuple2<Object, Object>(score, point.label());
                }
        );

        System.out.println("Datos test");
        System.out.println(scoreAndLabels.collect());

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();

        System.out.println("Area under ROC = " + auROC);

        // Save and load model
        model.save(sparkContext, "modelPath");
        //SVMModel loadedModel = SVMModel.load(sparkContext, "modelPath");
        //System.out.println(loadedModel);
    }
}