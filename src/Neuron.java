/**
 * Created by dn1k on 05.12.2015.
 */
public class Neuron {
    public double sigma;
    public double soma;
    public double pValue;

    public Neuron() {

    }

    public void activate() {
        pValue = Activation.sigmoid(soma);
        soma = 0;
    }
}
