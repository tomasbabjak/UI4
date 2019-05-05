import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;

import java.io.*;

public class NeuralNetwork {

    private static Instances trainingData;
    private static Instances testingData;

    public static void main(String[] args) throws Exception{

        Classifier mlp = trainNN();
        mlp.buildClassifier(trainingData);

        //SerializationHelper.write(new FileOutputStream("nnReduced"), mlp);
        testingData = prepareTestInstance();

        Evaluation evaluation = new Evaluation(trainingData);
        evaluation.evaluateModel(mlp, testingData);
        System.out.println(evaluation.errorRate());
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());

    }

    public static MultilayerPerceptron trainNN() {

        FileReader trainreader = null;
        MultilayerPerceptron mlp = null;

        try {
            trainreader = new FileReader("D:\\FIIT\\4. semester\\UI\\UI4\\mnist_train.arff");
            trainingData = new Instances(trainreader);
            trainingData.setClassIndex(0);
            trainingData.attribute(408).setWeight(10);
            trainingData.attribute(436).setWeight(10);
            trainingData.attribute(464).setWeight(10);

            mlp = new MultilayerPerceptron();
            mlp.setOptions(Utils.splitOptions("-L 0.1 -M 0.2 -N 2 -H 100"));
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
            testreader = new FileReader("D:\\FIIT\\4. semester\\UI\\UI4\\mnist_test.arff");
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
