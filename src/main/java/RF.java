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

        //Vytvor a natrenuj novy random forest
        RandomForest mlp = trainRF();
        mlp.buildClassifier(trainingData);

        //serializacia teda ulozenie modelu:
        //SerializationHelper.write(new FileOutputStream("RandomForest"), mlp);

        testingData = prepareTestInstance();

        //testovanie a vyhodnotenie modelu
        Evaluation evaluation = new Evaluation(trainingData);
        evaluation.evaluateModel(mlp, testingData);
        System.out.println(evaluation.errorRate());
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());

    }

    public static RandomForest trainRF() {

        FileReader trainreader = null;
        RandomForest randomForest = null;

        try {
            trainreader = new FileReader("D:\\FIIT\\4. semester\\UI\\UI4\\mnist_train.arff");
            trainingData = new Instances(trainreader);
            trainingData.setClassIndex(0);
            randomForest = new RandomForest();
            //Tu sa urcuje pocet stromov v lese:
            randomForest.setNumIterations(10);
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return randomForest;
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
