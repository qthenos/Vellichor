package Vellichor;

import java.util.*;

public class PerformanceMetrics {
    private ArrayList<String> actualLabels;
    private ArrayList<String> predictedLabels;
    private Set<String> uniqueLabels;
    
    public PerformanceMetrics(ArrayList<String> actual, ArrayList<String> predicted) {
        if (actual.size() != predicted.size()) {
            throw new IllegalArgumentException("Actual and predicted labels must have the same size");
        }
        this.actualLabels = actual;
        this.predictedLabels = predicted;
        this.uniqueLabels = new HashSet<>(actual);
    }
    
    private Map<String, Integer> getConfusionMatrix(String targetClass) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (int i = 0; i < actualLabels.size(); i++) {
            String actual = actualLabels.get(i);
            String predicted = predictedLabels.get(i);
            
            if (actual.equals(targetClass) && predicted.equals(targetClass)) {
                tp++;
            } else if (!actual.equals(targetClass) && predicted.equals(targetClass)) {
                fp++;
            } else if (!actual.equals(targetClass) && !predicted.equals(targetClass)) {
                tn++;
            } else if (actual.equals(targetClass) && !predicted.equals(targetClass)) {
                fn++;
            }
        }
        
        Map<String, Integer> confusionMatrix = new HashMap<>();
        confusionMatrix.put("tp", tp);
        confusionMatrix.put("fp", fp);
        confusionMatrix.put("tn", tn);
        confusionMatrix.put("fn", fn);
        return confusionMatrix;
    }
    
    public double getPrecision(String targetClass) {
        Map<String, Integer> cm = getConfusionMatrix(targetClass);
        int tp = cm.get("tp");
        int fp = cm.get("fp");
        return (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
    }
    
    public double getRecall(String targetClass) {
        Map<String, Integer> cm = getConfusionMatrix(targetClass);
        int tp = cm.get("tp");
        int fn = cm.get("fn");
        return (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
    }
    
    public double getF1Score(String targetClass) {
        double precision = getPrecision(targetClass);
        double recall = getRecall(targetClass);
        return (precision + recall) == 0 ? 0.0 : 2 * (precision * recall) / (precision + recall);
    }
    
    public double getAccuracy() {
        int correct = 0;
        for (int i = 0; i < actualLabels.size(); i++) {
            if (actualLabels.get(i).equals(predictedLabels.get(i))) {
                correct++;
            }
        }
        return (double) correct / actualLabels.size();
    }
    
    public double getMacroPrecision() {
        double sum = 0.0;
        for (String label : uniqueLabels) {
            sum += getPrecision(label);
        }
        return sum / uniqueLabels.size();
    }
    
    public double getMacroRecall() {
        double sum = 0.0;
        for (String label : uniqueLabels) {
            sum += getRecall(label);
        }
        return sum / uniqueLabels.size();
    }
    
    public double getMacroF1Score() {
        double sum = 0.0;
        for (String label : uniqueLabels) {
            sum += getF1Score(label);
        }
        return sum / uniqueLabels.size();
    }
    
    public double getWeightedPrecision() {
        double weightedSum = 0.0;
        int totalSamples = actualLabels.size();
        
        for (String label : uniqueLabels) {
            int support = Collections.frequency(actualLabels, label);
            double weight = (double) support / totalSamples;
            weightedSum += getPrecision(label) * weight;
        }
        return weightedSum;
    }
    
    public double getWeightedRecall() {
        double weightedSum = 0.0;
        int totalSamples = actualLabels.size();
        
        for (String label : uniqueLabels) {
            int support = Collections.frequency(actualLabels, label);
            double weight = (double) support / totalSamples;
            weightedSum += getRecall(label) * weight;
        }
        return weightedSum;
    }
    
    public double getWeightedF1Score() {
        double weightedSum = 0.0;
        int totalSamples = actualLabels.size();
        
        for (String label : uniqueLabels) {
            int support = Collections.frequency(actualLabels, label);
            double weight = (double) support / totalSamples;
            weightedSum += getF1Score(label) * weight;
        }
        return weightedSum;
    }
    
    public Map<String, Integer> getSupport() {
        Map<String, Integer> support = new HashMap<>();
        for (String label : uniqueLabels) {
            support.put(label, Collections.frequency(actualLabels, label));
        }
        return support;
    }
    
    public void printClassificationReport() {
        System.out.println("\n=== CLASSIFICATION REPORT ===");
        System.out.println("Class\t\tPrecision\tRecall\t\tF1-Score\tSupport");
        System.out.println("-------------------------------------------------------------");
        
        Map<String, Integer> support = getSupport();
        for (String label : uniqueLabels) {
            System.out.printf("%-12s\t%.4f\t\t%.4f\t\t%.4f\t\t%d%n", 
                label, getPrecision(label), getRecall(label), getF1Score(label), support.get(label));
        }
        
        System.out.println("-------------------------------------------------------------");
        System.out.printf("%-12s\t%.4f\t\t%.4f\t\t%.4f\t\t%d%n", 
            "Macro Avg", getMacroPrecision(), getMacroRecall(), getMacroF1Score(), actualLabels.size());
        System.out.printf("%-12s\t%.4f\t\t%.4f\t\t%.4f\t\t%d%n", 
            "Weighted Avg", getWeightedPrecision(), getWeightedRecall(), getWeightedF1Score(), actualLabels.size());
        System.out.println("-------------------------------------------------------------");
        System.out.printf("Accuracy: %.4f%n", getAccuracy());
        System.out.println("=== END REPORT ===\n");
    }
} 