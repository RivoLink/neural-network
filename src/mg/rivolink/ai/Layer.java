package mg.rivolink.ai;

import java.io.Serializable;

public class Layer implements Serializable {

    private static final long serialVersionUID = 1L;

    public final int inputSize;
    public final int neuronCount;

    public final Neuron[] neurons;
    private final Neuron.Activation activation;

    private float[] layerInputs;
    private float[] cachedOutputs;
    private float[] cachedZValues;

    public Layer(int inputSize, int neuronCount) {
        this(inputSize, neuronCount, Neuron.Activation.SIGMOID);
    }

    public Layer(int inputSize, int neuronCount, Neuron.Activation activation) {
        this.inputSize = inputSize;
        this.neuronCount = neuronCount;
        this.activation = activation;

        this.neurons = new Neuron[neuronCount];
        this.cachedOutputs = new float[neuronCount];
        this.cachedZValues = new float[neuronCount];

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
            cachedZValues[i] = neurons[i].getLastZ();
        }
        return cachedOutputs;
    }

    public float[] getOutputs() {
        return cachedOutputs;
    }

    public float[] getLastZValues() {
        return cachedZValues;
    }

    public Neuron.Activation getActivation() {
        return activation;
    }

    public float[] getInputs() {
        return layerInputs;
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
