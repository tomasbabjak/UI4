import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.trees.RandomForest;

public class RF {

    private static Instances trainingData;
    private static Instances testingData;


    public static void main(String[] args) throws Exception{

        RandomForest mlp = trainRF();
        mlp.buildClassifier(trainingData);
        //SerializationHelper.write(new FileOutputStream("rf"), mlp);
        testingData = prepareTestInstance();

        Evaluation evaluation = new Evaluation(trainingData);
        evaluation.evaluateModel(mlp, testingData);
        System.out.println(evaluation.errorRate());
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());

    }

    public static RandomForest trainRF() {

        FileReader trainreader = null;
        RandomForest mlp = null;

        try {
            trainreader = new FileReader("src/main/input/train.arff");
            trainingData = new Instances(trainreader);
            trainingData.setClassIndex(0);
            mlp = new RandomForest();
            mlp.setNumIterations(10);
        }
        catch (Exception e){
            e.printStackTrace();
        }finally {
            if (trainreader != null) {
                try {
                    trainreader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return mlp;
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
