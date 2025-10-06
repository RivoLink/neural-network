package mg.rivolink.ai;

public class Layer {

    public enum Activation {
        SIGMOID, RELU, LINEAR
    }

    public final int nCount;
    public final Neuron[] neurons;
    private final Activation activation;

    public Layer(int iLen, int nCount) {
        this(iLen, nCount, Activation.SIGMOID);
    }

    public Layer(int iLen, int nCount, Activation activation) {
        this.nCount = nCount;
        this.activation = activation;

        neurons = new Neuron[nCount];
        for (int i = 0; i < nCount; i++) {
            neurons[i] = new Neuron(iLen);
        }
    }

    public Layer setInputs(int bits) {
        for (int i = 0; i < nCount; i++) {
            neurons[i].setInput(bits);
        }
        return this;
    }

    public Layer setInputs(float... inputs) {
        for (int i = 0; i < nCount; i++) {
            neurons[i].setInput(inputs);
        }
        return this;
    }

    public float[] getOutputs() {
        float[] outputs = new float[nCount];
        for (int i = 0; i < nCount; i++) {
            switch (activation) {
                case RELU:
                    outputs[i] = neurons[i].getOutputReLU();
                    break;
                case LINEAR:
                    outputs[i] = neurons[i].getOutputLinear();
                    break;
                case SIGMOID:
                default:
                    outputs[i] = neurons[i].getOutput();
                    break;
            }
        }
        return outputs;
    }

    public Activation getActivation() {
        return activation;
    }

    // Copy layer weights
    public void copyWeightsFrom(Layer other) {
        if (this.nCount != other.nCount) {
            throw new IllegalArgumentException("Layer sizes don't match");
        }
        for (int i = 0; i < nCount; i++) {
            this.neurons[i].copyWeightsFrom(other.neurons[i]);
        }
    }
}
