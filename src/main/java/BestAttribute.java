import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;


public class BestAttribute {
    public static void main(String[] args) throws Exception {

        int count = 0;

        FileReader trainreader = new FileReader("mnist_train.arff");
        Instances testingData = new Instances(trainreader);
        testingData.setClassIndex(0);

        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        eval.buildEvaluator(testingData);

        //evaluacia jednotlivych atributov
        Map<Attribute, Double> infogainscores = new HashMap<Attribute, Double>();
        for (int i = 0; i < testingData.numAttributes(); i++) {
            Attribute t_attr = testingData.attribute(i);
            double infogain  = eval.evaluateAttribute(i);
            infogainscores.put(t_attr, infogain);
        }

        //vypis obrazka s hodnotami atributov na jednotlivych poziciach pixelov
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                double infogain  = eval.evaluateAttribute(i*28+j) * 100;
                int inn = (int) infogain;
                if(inn < 10) System.out.print(inn + "   ");
                else System.out.print(inn + "  ");
            }
            System.out.print("\n");
        }

        //usporiadanie hash mapy podla hodnoty
        final Map<Attribute, Double> sortedByCount = infogainscores.entrySet()
                .stream()
                .sorted(Map.Entry.<Attribute, Double>comparingByValue().reversed())
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        System.out.println(sortedByCount);
        Object array[] = sortedByCount.keySet().toArray();
        System.out.println(array[0]);

//        for (int i = 5; i < 25; i++) {
//            for (int j = 5; j < 25; j++) {
//                FileReader trainreader = new FileReader("mnist_test_reduced.arff");
//                Instances testingData = new Instances(trainreader);
//                testingData.setClassIndex(0);
//                double arr[] = testingData.attributeToDoubleArray(1);
//                //for(double i : arr) i = 1;
//
//                ++count;
//                testingData.deleteAttributeAt(count);
//                Attribute att = new Attribute(i + "x" + j, count);
//                testingData.insertAttributeAt(att, count);
//
//
//
//                MultilayerPerceptron classifier = (MultilayerPerceptron) SerializationHelper.read("nnReduced");
//                Evaluation evalTrain = new Evaluation(testingData);
//                evalTrain.evaluateModel(classifier, testingData);
//                System.out.println(i + "x" + j + " : \n");
//                System.out.println(evalTrain.toSummaryString());
//            }
//        }

    }
}
