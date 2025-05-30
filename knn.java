package Vellichor;

import java.util.*;

public class knn {
    ArrayList<ArrayList<Double>> features;
    ArrayList<String> labels;
    int k;

    public knn(ArrayList<ArrayList<Double>> infeatures, ArrayList<String> inLabels, int in_k) {
        features = infeatures;
        labels = inLabels;
        k = in_k;
        if (in_k <= 0 || in_k > infeatures.size()) {
            throw new IllegalArgumentException("Invalid value of k: must be between 1 and the number of data points.");
        }
        if (in_k % 2 == 0) {
            throw new IllegalArgumentException("k must be odd to avoid voting ties.");
        }
    }

    public double euclideanDistance(ArrayList<Double> dp1, ArrayList<Double> dp2) {
        double sum = 0;
        for (int i=0; i<dp1.size(); i++) {
            sum += Math.pow(dp1.get(i) - dp2.get(i), 2);
        }
        return Math.sqrt(sum);
    }

    public HashMap<Integer, Double> getDistances(ArrayList<Double> attributes) {
        HashMap<Integer, Double> distances = new HashMap<>();
        for (int i=0; i<features.size(); i++) {
            distances.put(i, euclideanDistance(attributes, features.get(i)));
        }
        return distances;
    }

    public HashMap<Integer, String> getNearestNeighbors(HashMap<Integer, Double> distances) {
        HashMap<Integer, Double> distancesCopy = new HashMap<>(distances);
        HashMap<Integer, String> nearest = new HashMap<>();
        for (int n=0; n<this.k; n++) {
            Map.Entry<Integer, Double> lowest = null;
            for (Map.Entry<Integer, Double> entry : distancesCopy.entrySet()) {
                if (lowest == null || entry.getValue() < lowest.getValue()) {
                    lowest = entry;
                }
            }
            nearest.put(lowest.getKey(), labels.get(lowest.getKey()));
            distancesCopy.remove(lowest.getKey());
        }
        return nearest;
    }

    public String predict(ArrayList<Double> attributes) {
        if (attributes.size() != features.get(0).size()) {
            System.out.println("Invalid attributes");
            return null;
        }
        HashMap<Integer, Double> distances = getDistances(attributes);
        HashMap<Integer, String> nearestNeighbors = getNearestNeighbors(distances);
        int result = 0;
        for (String label : nearestNeighbors.values()) {
            if (Objects.equals(label, "M")) {
                result--;
            } else if (Objects.equals(label, "B")) {
                result++;
            } else {
                System.out.println("Invalid label (non M/B)");
                return null;
            }
        }
        return result > 0 ? "B" : "M";
    }
}
