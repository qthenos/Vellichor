package Vellichor;

import java.util.ArrayList;

public class DataSplit {
    public ArrayList<ArrayList<Double>> trainFeatures;
    public ArrayList<String> trainLabels;
    public ArrayList<ArrayList<Double>> testFeatures;
    public ArrayList<String> testLabels;
    public ArrayList<ArrayList<Double>> normalizedTrainFeatures;
    public ArrayList<ArrayList<Double>> normalizedTestFeatures;

    public DataSplit(ArrayList<ArrayList<Double>> trainFeatures, ArrayList<String> trainLabels,
                     ArrayList<ArrayList<Double>> testFeatures, ArrayList<String> testLabels) {
        this.trainFeatures = trainFeatures;
        this.trainLabels = trainLabels;
        this.testFeatures = testFeatures;
        this.testLabels = testLabels;
    }
}
