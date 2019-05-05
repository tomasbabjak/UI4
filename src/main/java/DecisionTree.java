import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class DecisionTree {
    private static Instances trainingData;
    private static Instances testingData;

    public static void main(String[] args) throws Exception {

        J48 decisionTree = trainTree();
        decisionTree.buildClassifier(trainingData);

        // Vypisanie stromu a jeho serializacia
        //System.out.println(j48.toString());
        //SerializationHelper.write(new FileOutputStream("DecisionTree"), j48);

        Instances testingData = prepareTestInstance();

        // Testovanie stromu a evaluacia
        Evaluation evaluation = new Evaluation(trainingData);
        evaluation.evaluateModel(decisionTree, testingData);
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());

    }

    public static J48 trainTree() {

        FileReader trainreader = null;
        J48 decisionTree = null;

        try {
            trainreader = new FileReader("D:\\FIIT\\4. semester\\UI\\UI4\\mnist_train.arff");
            trainingData = new Instances(trainreader);
            trainingData.setClassIndex(0);

            //Nastavenie Unpruned tree teda neobmedzeneho stromu
            String[] options = new String[1];
            options[0] = "-U";

            decisionTree = new J48();
            decisionTree.setOptions(options);
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return decisionTree;
    }


    public static Instances prepareTestInstance() {

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader("D:\\FIIT\\4. semester\\UI\\UI4\\mnist_test.arff"));
            testingData = new Instances(reader);
            testingData.setClassIndex(0);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return testingData;
    }
}
