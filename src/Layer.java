import java.util.ArrayList;
import java.util.Random;

/**
 * Created by dn1k on 05.12.2015.
 */
public class Layer {
    Layer prevLayer, nextLayer;
    int size;
    double learningRate;
    ArrayList<Neuron> neurons;
    ArrayList<ArrayList<Double>> weights;

    public Layer(int size, double learningRate) {
        this.size = size;
        this.learningRate = learningRate;
        createNeurons();
    }

    private void createNeurons() {
        neurons = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            neurons.add(new Neuron());
        }
    }

    public void initWeights() {
        weights = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis());
        for (int i = 0; i < size + 1; i++) {
            ArrayList<Double> w = new ArrayList<>();
            for (int j = 0; j < nextLayer.size; j++) {
                w.add(rand.nextDouble() - 0.5D);
            }
            weights.add(w);
        }
    }

    public void activate(DigitImage input) {
        if (input == null) {
            for (int i = 0; i < neurons.size(); i++) {
                neurons.get(i).activate();
            }
        } else {
            for (int i = 0; i < neurons.size(); i++) {
                neurons.get(i).soma = input.data[i]/255.0D;
                neurons.get(i).activate();
            }
        }
    }

    public void propagate() {
        for (int i = 0; i < neurons.size(); i++) {
            for (int j = 0; j < weights.get(i).size(); j++) {
                nextLayer.neurons.get(j).soma += neurons.get(i).pValue * weights.get(i).get(j);
            }
        }
        //bias
        for (int j = 0; j < weights.get(weights.size() - 1).size(); j++) {
            nextLayer.neurons.get(j).soma += weights.get(weights.size() - 1).get(j);
        }
    }

    public ArrayList<Double> getValues() {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < neurons.size(); i++) {
            values.add(neurons.get(i).pValue);
        }
        return values;
    }

    public void backPropOut(ArrayList<Double> result, int label) {
        int s;
        double value;
        for (int i = 0; i < size; i++) {
            if (i == label) {
                s = 1;
            } else {
                s = 0;
            }
            value = neurons.get(i).pValue;
            neurons.get(i).sigma = 2 * (s - value) * (value) * (1 - value);
        }
    }


    public void backProp() {
        double s_in, value;
        for (int i = 0; i < size; i++) {
            s_in = 0;
            value = neurons.get(i).pValue;
            for (int j = 0; j < nextLayer.size; j++) {
                s_in += nextLayer.neurons.get(j).sigma * weights.get(i).get(j);
                weights.get(i).set(j, weights.get(i).get(j) + (nextLayer.neurons.get(j).sigma * value * learningRate));
            }
            neurons.get(i).sigma = 2 * s_in * (value) * (1 - value);
        }
        //bias
        s_in = 0;
        for (int j = 0; j < nextLayer.size; j++) {
            s_in += nextLayer.neurons.get(j).sigma * weights.get(size - 1).get(j);
            weights.get(size - 1).set(j, weights.get(size - 1).get(j) + (nextLayer.neurons.get(j).sigma * learningRate));
        }
    }
}
