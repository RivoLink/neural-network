package mg.rivolink.ai;

public class Layer {

    public final int inputSize;
    public final int neuronCount;

    public final Neuron[] neurons;
    private final Neuron.Activation activation;

    private float[] layerInputs;
    private float[] cachedOutputs;

    public Layer(int inputSize, int neuronCount) {
        this(inputSize, neuronCount, Neuron.Activation.SIGMOID);
    }

    public Layer(int inputSize, int neuronCount, Neuron.Activation activation) {
        this.inputSize = inputSize;
        this.neuronCount = neuronCount;
        this.activation = activation;

        this.neurons = new Neuron[neuronCount];
        this.cachedOutputs = new float[neuronCount];

        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputSize);
        }
    }

    public void setInputs(float[] inputs) {
        if (inputs.length != inputSize) {
            throw new IllegalArgumentException(
                "Input size mismatch: expected " + inputSize + ", got " + inputs.length
            );
        }
        this.layerInputs = inputs;
    }

    public float[] forward() {
        for (int i = 0; i < neuronCount; i++) {
            cachedOutputs[i] = neurons[i].computeOutput(layerInputs, activation);
        }
        return cachedOutputs;
    }

    public float[] getOutputs() {
        return cachedOutputs;
    }

    public Neuron.Activation getActivation() {
        return activation;
    }

    public float[] getInputs() {
        return layerInputs;
    }

    public float[] getLastZValues() {
        float[] zValues = new float[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            zValues[i] = neurons[i].getLastZ();
        }
        return zValues;
    }

    public void copyWeightsFrom(Layer other) {
        if (this.neuronCount != other.neuronCount) {
            throw new IllegalArgumentException("Layer sizes don't match");
        }
        for (int i = 0; i < neuronCount; i++) {
            this.neurons[i].copyWeightsFrom(other.neurons[i]);
        }
    }

    public void softUpdate(Layer other, float tau) {
        if (this.neuronCount != other.neuronCount) {
            throw new IllegalArgumentException("Layer sizes don't match");
        }
        for (int i = 0; i < neuronCount; i++) {
            this.neurons[i].softUpdate(other.neurons[i], tau);
        }
    }

}
