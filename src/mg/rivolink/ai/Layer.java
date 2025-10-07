package mg.rivolink.ai;

public class Layer {

    public final int neuronCount;
    public final Neuron[] neurons;

    public Layer(int inputLength, int neuronCount) {
        this.neuronCount = neuronCount;
        this.neurons = new Neuron[neuronCount];

        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputLength);
        }
    }

    public Layer setInputs(int bits) {
        for (int i = 0; i < neuronCount; i++) {
            neurons[i].setInput(bits);
        }
        return this;
    }

    public Layer setInputs(float... inputs) {
        for (int i = 0; i < neuronCount; i++) {
            neurons[i].setInput(inputs);
        }
        return this;
    }

    public float[] getOutputs() {
        float[] outputs = new float[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            outputs[i] = neurons[i].getOutput();
        }
        return outputs;
    }

}
