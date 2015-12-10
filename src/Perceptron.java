import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.atomic.DoubleAccumulator;

/**
 * Created by dn1k on 05.12.2015.
 */
public class Perceptron {
    ArrayList<Layer> layers;
    double learningRate;
    ArrayList<DigitImage> train;

    public Perceptron(int[] layerSizes, double learningRate) {
        this.learningRate = learningRate;
        createLayers(layerSizes);
    }

    private void createLayers(int[] layerSizes) {
        layers = new ArrayList<>();
        for (int i = 0; i < layerSizes.length; i++) {
            layers.add(new Layer(layerSizes[i], learningRate));
        }
        for (int i = 0; i < layers.size(); i++) {
            if (i != 0) {
                layers.get(i).prevLayer = layers.get(i - 1);
            }
            if (i != layers.size() - 1) {
                layers.get(i).nextLayer = layers.get(i + 1);
                layers.get(i).initWeights();
            }
        }
    }

    public void fit(ArrayList<DigitImage> train, boolean fromSnapshot) throws FileNotFoundException {
        this.train = train;
        ArrayList<Double> result;
        ArrayList<DigitImage> testBatch = getRandSubset(1000);
        int epoch = 1;
        int epochCount = 5000;
        if (fromSnapshot) {
            epoch = loadShapshot();
        }
        for (epoch = epoch; epoch <= epochCount; epoch++) {
            ArrayList<DigitImage> batch = getRandSubset(1000);
            for (DigitImage sample : batch) {
                result = propagate(sample);
                backProp(result, sample.label);
            }
            if (epoch % 20 == 0) {
                printEpochRes(testBatch, epoch);
            }
            if (epoch % 100 == 0) {
                takeSnapshot(epoch);
            }
        }
    }

    private int loadShapshot() throws FileNotFoundException {
        Scanner in = new Scanner(new FileReader("snap.txt"));
        int epoch = in.nextInt();
        int ls = in.nextInt();
        if (ls != layers.size()) {
            System.out.println("Different structure");
            return 0;
        }
        for (int i = 0; i < ls; i++) {
            if (in.nextInt() != layers.get(i).size) {
                System.out.println("Different structure");
                return 0;
            }
        }

        for (int i = 0; i < layers.size() - 1; i++) {
            ArrayList<ArrayList<Double>> w = new ArrayList<>();
            for (int j = 0; j < layers.get(i).size + 1; j++) {
                w.add(new ArrayList<>());
                for (int q = 0; q < layers.get(i + 1).size; q++) {
                    w.get(j).add(Double.parseDouble(in.next()));
                }
            }
            layers.get(i).weights = w;
        }
        System.out.println("Snapshot loading completed");
        return epoch;
    }

    private void takeSnapshot(int epoch) throws FileNotFoundException {
        PrintWriter out;
        out = new PrintWriter("snap.txt");
        out.println(epoch);
        out.println(layers.size());
        for (int i = 0; i < layers.size(); i++) {
            out.println(layers.get(i).size);
        }
        for (int i = 0; i < layers.size() - 1; i++) {
            ArrayList<ArrayList<Double>> w = layers.get(i).weights;
            for (int j = 0; j < w.size(); j++) {
                for (int q = 0; q < w.get(j).size(); q++) {
                    out.print(w.get(j).get(q) + " ");
                }
                out.println();
            }
        }
        System.out.println("Snapshot saved");
        out.close();
    }

    private void printEpochRes(ArrayList<DigitImage> test, int epoch) {
        double pos = 0;
        for (int i = 0; i < test.size(); i++) {
            ArrayList<Double> resOut = propagate(test.get(i));
            double max = -1;
            int label = 0;
            for (int j = 0; j < resOut.size(); j++) {
                if (resOut.get(j) > max) {
                    max = resOut.get(j);
                    label = j;
                }
            }
//            if (i == 10) {
//                System.out.println(resOut);
//                System.out.println(label);
//                System.out.println(test.get(i).label);
//            }
            if (label == test.get(i).label) {
                pos += 1;
            }
        }
        System.out.println("Epoch = " + epoch + ", acc = " + pos/test.size());
    }

    private int predictLabel(DigitImage testImage) {
        ArrayList<Double> resOut = propagate(testImage);
        double max = -1;
        int label = 0;
        for (int i = 0; i < resOut.size(); i++) {
            if (resOut.get(i) > max) {
                max = resOut.get(i);
                label = i;
            }
        }
        return label;
    }

    private ArrayList<Double> propagate(DigitImage sample) {
        layers.get(0).activate(sample);
        layers.get(0).propagate();
        for (int i = 1; i < layers.size() - 1; i++) {
            layers.get(i).activate(null);
            layers.get(i).propagate();
        }
        layers.get(layers.size() - 1).activate(null);
        return layers.get(layers.size() - 1).getValues();
    }

    private ArrayList<DigitImage> getRandSubset(int size) {
        ArrayList<DigitImage> list = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis());
        for (int i = 0; i < size; i++) {
            list.add(train.get(rand.nextInt(train.size())));
        }
        return list;
    }

    private void backProp(ArrayList<Double> result, int label) {
        layers.get(layers.size() - 1).backPropOut(result, label);
        for (int i = layers.size() - 2; i >= 0; i--) {
            layers.get(i).backProp();
        }
    }
}
