import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class DecisionTree {
    private static Instances trainingData;
    private Instances testingData;

    public static void main(String[] args) throws Exception {

        DecisionTree decisionTree = new DecisionTree("src/main/input/train.arff");
        J48 j48 = decisionTree.trainTheTree();

        // Print the resulted tree
        System.out.println(j48.toString());
        SerializationHelper.write(new FileOutputStream("tree"), j48);

        // Test the tree
         Instances testInstances = decisionTree.prepareTestInstance();
//        for(Instance i : testInstances){
//            int result = (int) id3tree.classifyInstance(i);
//            String readableResult = decisionTree.trainingData.attribute(0).value(result);
//            System.out.println(" ----------------------------------------- ");
//            System.out.println("Test data               : " + (i.toDoubleArray())[0]);
//            System.out.println("Test data classification: " + readableResult);
//
//        }

        Classifier cls = new J48();
        cls.buildClassifier(trainingData);

        Evaluation evaluation = new Evaluation(trainingData);
        evaluation.evaluateModel(cls, testInstances);
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());


    }

    public DecisionTree(String fileName) {
        BufferedReader reader = null;
        try {
            // Read the training data
            reader = new BufferedReader(new FileReader(fileName));
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
    }

    private J48  trainTheTree() {
        J48  j48 = new J48 ();

        String[] options = new String[1];
         //Use unpruned tree.
        options[0] = "-U";

        try {
            j48.setOptions(options);
            j48.buildClassifier(trainingData);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return j48;
    }

    private Instances prepareTestInstance() {

        BufferedReader reader = null;
        try {
            // Read the training data
            reader = new BufferedReader(new FileReader("src/main/input/test.arff"));
            testingData = new Instances(reader);

            // Setting class attribute
            testingData.setClassIndex(0);
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

        return testingData;
    }
}
