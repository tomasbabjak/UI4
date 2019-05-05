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

        //Vytvor a natrenuj novu neuronovu siet
        Classifier mlp = trainNN();
        mlp.buildClassifier(trainingData);

        //serializacia teda ulozenie modelu:
        //SerializationHelper.write(new FileOutputStream("NeuralNetwork"), mlp);

        testingData = prepareTestInstance();

        //testovanie a vyhodnotenie modelu
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
            mlp = new MultilayerPerceptron();
            //Nastavenie parametrov neuronovej siete: -H je pocet skrytych vrstiev
            mlp.setOptions(Utils.splitOptions("-L 0.1 -M 0.2 -N 2 -H 100"));
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return mlp;
    }

    private static Instances prepareTestInstance() {

        FileReader testreader = null;
        try {
            testreader = new FileReader("D:\\FIIT\\4. semester\\UI\\UI4\\mnist_test.arff");
            testingData = new Instances(testreader);
                testingData.setClassIndex(0);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return testingData;
    }

}
