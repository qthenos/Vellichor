package Vellichor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

    static ArrayList<ArrayList<Double>> features;
    static ArrayList<String> labels;

    public static void main(String[] args) {
        process("./Vellichor/data/wdbc.data");
//        checkData();
        knn knn = new knn(features, labels, 7);
        int test = 0;
        System.out.print("Label : " + labels.get(test));
        System.out.println(" | Features: " + features.get(test));
        System.out.println("Prediction: " + knn.predict(features.get(test)));
    }

    public static void process(String filename) {
        features = new ArrayList<>();
        labels = new ArrayList<>();
        try (FileReader fileReader = new FileReader(filename)) {
            BufferedReader reader = new BufferedReader(fileReader);
            String line;
            String[] splitLine;
            while (reader.ready()) {
                line = reader.readLine();
                splitLine = line.split(",");
                labels.add(splitLine[1]);
                features.add(new ArrayList<>());
                for (int i=2; i<splitLine.length; i++) {
                    features.get(features.size()-1).add(Double.parseDouble(splitLine[i]));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void checkData() {
        for (int i=0; i<labels.size(); i++) {
            System.out.print("Label: " + labels.get(i));
            System.out.println(" | Features: " + features.get(i));
        }
    }
}
