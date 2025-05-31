package Vellichor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Main {

    static ArrayList<ArrayList<Double>> features;
    static ArrayList<String> labels;

    public static void main(String[] args) {
        loadData("./Vellichor/data/wdbc.data");
        
        Random random = new Random(17);
        shuffleData(random);
        
        runKNNExperiments(random);
    }
    
    public static void runKNNExperiments(Random random) {
        // using 0.2 of the data for testing. Will remove when we generate our own data to test on.
        DataSplit dataSplit = splitData(0.8);
        
        printDatasetInfo(dataSplit);
        
        testKNNPerformance(dataSplit);
        
        findOptimalK(dataSplit);
    }
    
    public static class DataSplit {
        public ArrayList<ArrayList<Double>> trainFeatures;
        public ArrayList<String> trainLabels;
        public ArrayList<ArrayList<Double>> testFeatures;
        public ArrayList<String> testLabels;
        
        public DataSplit(ArrayList<ArrayList<Double>> trainFeatures, ArrayList<String> trainLabels,
                        ArrayList<ArrayList<Double>> testFeatures, ArrayList<String> testLabels) {
            this.trainFeatures = trainFeatures;
            this.trainLabels = trainLabels;
            this.testFeatures = testFeatures;
            this.testLabels = testLabels;
        }
    }
    
    public static DataSplit splitData(double trainRatio) {
        int trainSize = (int) (features.size() * trainRatio);
        
        ArrayList<ArrayList<Double>> trainFeatures = new ArrayList<>(features.subList(0, trainSize));
        ArrayList<String> trainLabels = new ArrayList<>(labels.subList(0, trainSize));
        ArrayList<ArrayList<Double>> testFeatures = new ArrayList<>(features.subList(trainSize, features.size()));
        ArrayList<String> testLabels = new ArrayList<>(labels.subList(trainSize, labels.size()));
        
        return new DataSplit(trainFeatures, trainLabels, testFeatures, testLabels);
    }
    
    public static void printDatasetInfo(DataSplit dataSplit) {
        System.out.println("Dataset loaded: " + features.size() + " samples");
        System.out.println("Training set: " + dataSplit.trainFeatures.size() + " samples");
        System.out.println("Test set: " + dataSplit.testFeatures.size() + " samples");
        System.out.println();
    }
    
    public static void testKNNPerformance(DataSplit dataSplit) {
        System.out.println("=== KNN PERFORMANCE EVALUATION ===\n");
        
        int[] kValues = {1, 3, 5, 7, 9, 11, 15};
        
        for (int k : kValues) {
            System.out.println("=".repeat(50));
            System.out.println("Testing KNN with k = " + k);
            System.out.println("=".repeat(50));
            
            knn knnModel = new knn(dataSplit.trainFeatures, dataSplit.trainLabels, k);
            
            ArrayList<String> predictions = makePredictions(knnModel, dataSplit.testFeatures);
            
            PerformanceMetrics metrics = new PerformanceMetrics(dataSplit.testLabels, predictions);
            metrics.printClassificationReport();
        }
    }
    
    public static void findOptimalK(DataSplit dataSplit) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("FINDING OPTIMAL K VALUE");
        System.out.println("=".repeat(60));
        
        int[] kValues = {1, 3, 5, 7, 9, 11, 15, 21, 25};
        double bestF1 = 0.0;
        int bestK = 1;
        
        System.out.println("K\tAccuracy\tF1-Score\tPrecision\tRecall");
        System.out.println("-".repeat(60));
        
        for (int k : kValues) {
            try {
                knn knnModel = new knn(dataSplit.trainFeatures, dataSplit.trainLabels, k);
                
                ArrayList<String> predictions = makePredictions(knnModel, dataSplit.testFeatures);
                
                PerformanceMetrics metrics = new PerformanceMetrics(dataSplit.testLabels, predictions);
                double accuracy = metrics.getAccuracy();
                double f1 = metrics.getMacroF1Score();
                double precision = metrics.getMacroPrecision();
                double recall = metrics.getMacroRecall();
                
                System.out.printf("%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f%n", 
                    k, accuracy, f1, precision, recall);
                
                if (f1 > bestF1) {
                    bestF1 = f1;
                    bestK = k;
                }
            } catch (IllegalArgumentException e) {
                System.out.printf("%d\t%s%n", k, "Invalid k value");
            }
        }
        
        System.out.println("-".repeat(60));
        System.out.printf("🏆 Best k value: %d (F1-Score: %.4f)%n", bestK, bestF1);
        System.out.println("=".repeat(60));
    }
    
    public static void loadData(String filename) {
        System.out.println("Loading data from: " + filename);
        process(filename);
        System.out.println("Data loaded successfully!\n");
    }
    
    public static ArrayList<String> makePredictions(knn knnModel, ArrayList<ArrayList<Double>> testFeatures) {
        ArrayList<String> predictions = new ArrayList<>();
        for (ArrayList<Double> testSample : testFeatures) {
            String prediction = knnModel.predict(testSample);
            predictions.add(prediction);
        }
        return predictions;
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
    
    public static void shuffleData(Random random) {
        for (int i = features.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            
            Collections.swap(features, i, j);
            Collections.swap(labels, i, j);
        }
    }

    public static void checkData() {
        for (int i=0; i<Math.min(5, labels.size()); i++) {
            System.out.print("Label: " + labels.get(i));
            System.out.println(" | Features: " + features.get(i));
        }
    }
}
