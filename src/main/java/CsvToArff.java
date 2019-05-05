import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CsvToArff {
    public static void main(String[] args) throws Exception{
            String f1 = "mnist_test.csv";
            String f2 = "mnist_test_modified.arff";

            // load the CSV file (input file)
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(f1));
            Instances data = loader.getDataSet();

            for (Instance instances : data) {
                int count = 0;
                double[] list = instances.toDoubleArray();
                for (double l : list) {
                    if (l == 0) count++;
                }
                instances.attribute(data.numAttributes()).setStringValue(count + "");
            }

            Attribute attribute = new Attribute("new", data.numAttributes());
           //data.add(new DenseInstance(1.0, pole));

        // save as an  ARFF (output file)
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(f2));
            saver.writeBatch();

    }
}

