import com.opencsv.CSVReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Test1 {

    public static void main(String[] args) throws Exception{

        String csvFile = "mnist_test_reduced.arff";
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";
        int[] pole = new int[784];

        try {

            br = new BufferedReader(new FileReader(csvFile));
            FileWriter csvWriter = new FileWriter("mnist_test_reduced.arff");
            int input = 0;

            while ((line = br.readLine()) != null) {

                String[] country = line.split(cvsSplitBy);

                csvWriter.append(country[0]);
                csvWriter.append(",");
                for(int i = 0; i < 28; i++) {
                    if (i == 1 || i == 2 || i == 3 || i == 0 || i == 27 || i == 26 || i == 25 || i == 24)
                        continue;
                    for(int j = 0; j < 28; j++) {
                        if (j == 1 || j == 2 || j == 3 || j == 0 || j == 27 || j == 26 || j == 25 || j == 24)
                            continue;

                        csvWriter.append(country[i*28+j+1]);
                        if(i*28+j+1 != input) csvWriter.append(country[i*28+j+1]);
                        else csvWriter.append("0");

                        if (i*28+j+1 != 668) csvWriter.append(",");
                        //System.out.print(country[i]);
                        //pole[i] = Integer.parseInt(country[i]);
                        //if (pole[i] == 0) count++;
                    }
                }
                //csvWriter.append(Integer.toString(count));
                csvWriter.append("\n");

            }

            csvWriter.flush();
            csvWriter.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        FileReader trainreader = null;

        trainreader = new FileReader("D:\\FIIT\\4. semester\\UI\\UI4\\mnist_test_reduced1.arff");
        Instances instances = new Instances(trainreader);
        instances.setClassIndex(0);

        MultilayerPerceptron classifier = (MultilayerPerceptron) SerializationHelper.read("nnReduced");
        Evaluation evalTrain = new Evaluation(instances);
        evalTrain.evaluateModel(classifier, instances);
        //System.out.println(evalTrain.toMatrixString());
        System.out.println(evalTrain.toSummaryString());

    }
}
