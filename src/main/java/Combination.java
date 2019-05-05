import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;

import java.awt.geom.AffineTransform;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

import static java.awt.geom.AffineTransform.getTranslateInstance;

public class Combination {

    private static Instances trainingData;
    private static Instances testingData;


    public static void main(String[] args) throws Exception{
        trainingData = prepareTrainingInstance();
        testingData = prepareTestInstance();

        //definition of Decision tree
        J48  id3tree = new J48 ();
        String[] options = new String[1];
        options[0] = "-U";
        id3tree.setOptions(options);
        id3tree.buildClassifier(trainingData);

        //definiiton of Neural Network
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setOptions(Utils.splitOptions("-L 0.1 -M 0.2 -N 2 -V 0 -S 0 -E 20 -H 100"));

        //definition of Random Forest
        RandomForest rf = new RandomForest();
        rf.setNumIterations(10);

        Classifier[] classifiers = {
                id3tree,
                rf,
                mlp
        };

        //stacker to combine models
        /*Stacking stacker = new Stacking();
        stacker.setMetaClassifier(mlp);
        stacker.setClassifiers(classifiers);
        stacker.buildClassifier(trainingData);*/

        //voter to combine models
        Vote voter = new Vote();
        voter.setClassifiers(classifiers);
        voter.buildClassifier(trainingData);
        SerializationHelper.write(new FileOutputStream("voter"), voter);


//Ada Boost
//        AdaBoostM1 adaBoostM1 = new AdaBoostM1();
//        adaBoostM1.setClassifier(classifiers);
//        adaBoostM1.buildClassifier(trainingData);

        Evaluation evaluation = new Evaluation(trainingData);
        evaluation.evaluateModel(voter, testingData);
        System.out.println(evaluation.errorRate());
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());

    }


    private static Instances prepareTrainingInstance() {

        BufferedReader reader = null;
        try {
            // Read the training data
            reader = new BufferedReader(new FileReader("src/main/input/train.arff"));
            trainingData = new Instances(reader);

            // Setting class attribute
            trainingData.setClassIndex(0);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return trainingData;
    }

    private static Instances prepareTestInstance() {

        FileReader testreader = null;
        try {
            // Read the training data
            testreader = new FileReader("src/main/input/test.arff");
            testingData = new Instances(testreader);
            testingData.setClassIndex(0);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (testreader != null) {
                try {
                    testreader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return testingData;
    }


}
