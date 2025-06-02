package Vellichor;

import java.util.ArrayList;

public class DataNormalizer {
    private ArrayList<Double> means;
    private ArrayList<Double> standardDeviations;
    private boolean isFitted = false;
    
    public DataNormalizer() {
        means = new ArrayList<>();
        standardDeviations = new ArrayList<>();
    }

    public void fit(ArrayList<ArrayList<Double>> features) {
        if (features.isEmpty()) {
            throw new IllegalArgumentException("Features cannot be empty");
        }
        
        int numFeatures = features.get(0).size();
        means.clear();
        standardDeviations.clear();

        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            double sum = 0.0;
            for (ArrayList<Double> sample : features) {
                sum += sample.get(featureIndex);
            }
            double mean = sum / features.size();
            means.add(mean);
        }

        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            double sumSquaredDiffs = 0.0;
            double mean = means.get(featureIndex);
            
            for (ArrayList<Double> sample : features) {
                double diff = sample.get(featureIndex) - mean;
                sumSquaredDiffs += diff * diff;
            }
            
            double variance = sumSquaredDiffs / features.size();
            double standardDeviation = Math.sqrt(variance);

            if (standardDeviation == 0.0) {
                standardDeviation = 1.0;
            }
            
            standardDeviations.add(standardDeviation);
        }
        
        isFitted = true;
    }
    
    //  z score normalization: x = (x - mean(x)) / std(x)
    public ArrayList<ArrayList<Double>> transform(ArrayList<ArrayList<Double>> features) {
        if (!isFitted) {
            throw new IllegalStateException("Normalizer must be fitted.");
        }
        
        ArrayList<ArrayList<Double>> normalizedFeatures = new ArrayList<>();
        
        for (ArrayList<Double> sample : features) {
            ArrayList<Double> normalizedSample = new ArrayList<>();
            
            for (int featureIndex = 0; featureIndex < sample.size(); featureIndex++) {
                double originalValue = sample.get(featureIndex);
                double mean = means.get(featureIndex);
                double std = standardDeviations.get(featureIndex);

                double normalizedValue = (originalValue - mean) / std;
                normalizedSample.add(normalizedValue);
            }
            
            normalizedFeatures.add(normalizedSample);
        }
        
        return normalizedFeatures;
    }

    public ArrayList<ArrayList<Double>> fitTransform(ArrayList<ArrayList<Double>> features) {
        fit(features);
        return transform(features);
    }
} 