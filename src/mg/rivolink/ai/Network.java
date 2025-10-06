package mg.rivolink.ai;

public class Network {

    public float alpha = 0.001f; // Learning rate for DQN

    public final int inputSize;
    public final Layer hiddenLayer1;
    public final Layer hiddenLayer2;
    public final Layer outputLayer;

    // Constructor for 2 hidden layers network
    public Network(int inputSize, int hidden1Size, int hidden2Size, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayer1 = new Layer(inputSize, hidden1Size, Layer.Activation.RELU);
        this.hiddenLayer2 = new Layer(hidden1Size, hidden2Size, Layer.Activation.RELU);
        this.outputLayer = new Layer(hidden2Size, outputSize, Layer.Activation.LINEAR);
    }

    // Constructor for 1 hidden layer network (backward compatibility)
    public Network(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayer1 = new Layer(inputSize, hiddenSize, Layer.Activation.SIGMOID);
        this.hiddenLayer2 = null;
        this.outputLayer = new Layer(hiddenSize, outputSize, Layer.Activation.SIGMOID);
    }

    public float[] predict(int bits) {
        return setInput(bits).getOutputs();
    }

    public float[] predict(float... inputs) {
        return setInput(inputs).getOutputs();
    }

    // Train with integer target (classification)
    public void train(int bits, int[] target) {
        setInput(bits);
        if (hiddenLayer2 != null) {
            backpropagationMSE(target);
        } else {
            trainingClassification(target);
        }
    }

    public void train(float[] inputs, int[] target) {
        setInput(inputs);
        if (hiddenLayer2 != null) {
            backpropagationMSE(target);
        } else {
            trainingClassification(target);
        }
    }

    // Train with float target (regression - for DQN)
    public void train(float[] inputs, float[] target) {
        setInput(inputs);
        if (hiddenLayer2 != null) {
            backpropagationMSE(target);
        } else {
            backpropagationMSE(target);
        }
    }

    public void train(int[] bitsArray, int[][] ytrains, int epochs) {
        int size = Math.min(bitsArray.length, ytrains.length);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < size; i++) {
                this.train(bitsArray[i], ytrains[i]);
            }
        }
    }

    public void train(float[][] xtrains, int[][] ytrains, int epochs) {
        int size = Math.min(xtrains.length, ytrains.length);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < size; i++) {
                this.train(xtrains[i], ytrains[i]);
            }
        }
    }

    private Network setInput(int bits) {
        hiddenLayer1.setInputs(bits);
        return this;
    }

    private Network setInput(float... inputs) {
        hiddenLayer1.setInputs(inputs);
        return this;
    }

    private float[] getOutputs() {
        float[] h1 = hiddenLayer1.getOutputs();
        if (hiddenLayer2 != null) {
            float[] h2 = hiddenLayer2.setInputs(h1).getOutputs();
            return outputLayer.setInputs(h2).getOutputs();
        } else {
            return outputLayer.setInputs(h1).getOutputs();
        }
    }

    // Backpropagation with MSE loss (for DQN with float targets)
    private void backpropagationMSE(float[] target) {
        float[] yhat = getOutputs();
        
        if (hiddenLayer2 != null) {
            // 3-layer network
            float[] h1 = hiddenLayer1.getOutputs();
            float[] h2 = hiddenLayer2.getOutputs();

            Neuron[] h1N = hiddenLayer1.neurons;
            Neuron[] h2N = hiddenLayer2.neurons;
            Neuron[] oN = outputLayer.neurons;

            // Output layer gradients
            float[] deltaOut = new float[oN.length];
            for (int k = 0; k < oN.length; k++) {
                float error = target[k] - yhat[k];
                float derivative = getActivationDerivative(outputLayer, oN[k].getLastZ());
                deltaOut[k] = error * derivative;
            }

            // Hidden layer 2 gradients
            float[] deltaH2 = new float[h2N.length];
            for (int j = 0; j < h2N.length; j++) {
                float error = 0;
                for (int k = 0; k < oN.length; k++) {
                    error += deltaOut[k] * oN[k].weights[j];
                }
                float derivative = getActivationDerivative(hiddenLayer2, h2N[j].getLastZ());
                deltaH2[j] = error * derivative;
            }

            // Hidden layer 1 gradients
            float[] deltaH1 = new float[h1N.length];
            for (int i = 0; i < h1N.length; i++) {
                float error = 0;
                for (int j = 0; j < h2N.length; j++) {
                    error += deltaH2[j] * h2N[j].weights[i];
                }
                float derivative = getActivationDerivative(hiddenLayer1, h1N[i].getLastZ());
                deltaH1[i] = error * derivative;
            }

            // Update output layer
            for (int k = 0; k < oN.length; k++) {
                oN[k].biais += alpha * deltaOut[k];
                for (int j = 0; j < h2N.length; j++) {
                    oN[k].weights[j] += alpha * deltaOut[k] * h2[j];
                }
            }

            // Update hidden layer 2
            for (int j = 0; j < h2N.length; j++) {
                h2N[j].biais += alpha * deltaH2[j];
                for (int i = 0; i < h1N.length; i++) {
                    h2N[j].weights[i] += alpha * deltaH2[j] * h1[i];
                }
            }

            // Update hidden layer 1
            for (int i = 0; i < h1N.length; i++) {
                h1N[i].biais += alpha * deltaH1[i];
                for (int inp = 0; inp < inputSize; inp++) {
                    h1N[i].weights[inp] += alpha * deltaH1[i] * h1N[0].inputs[inp];
                }
            }
        }
    }

    // Backpropagation with MSE loss (for integer targets)
    private void backpropagationMSE(int[] target) {
        float[] targetFloat = new float[target.length];
        for (int i = 0; i < target.length; i++) {
            targetFloat[i] = target[i];
        }
        backpropagationMSE(targetFloat);
    }

    // Original training method for classification (backward compatibility)
    private void trainingClassification(int[] target) {
        float[] yhat = getOutputs();
        float[] x = hiddenLayer1.getOutputs();
        int[] y = target;

        Neuron[] hN = hiddenLayer1.neurons;
        Neuron[] oN = outputLayer.neurons;

        float gE_wij, gE_wjk;
        for (int j = 0; j < hN.length; j++) {
            for (int k = 0; k < oN.length; k++) {
                gE_wjk = alpha * (y[k] - yhat[k]) * yhat[k] * (1 - yhat[k]);
                oN[k].biais += gE_wjk;
                oN[k].weights[j] += gE_wjk * x[j];
            }

            for (int i = 0; i < this.inputSize; i++) {
                gE_wij = 0;
                float xi = hiddenLayer1.neurons[0].inputs[i];
                for (int k = 0; k < oN.length; k++) {
                    float wjk = outputLayer.neurons[k].weights[j];
                    gE_wij += (y[k] - yhat[k]) * yhat[k] * (1 - yhat[k]) * wjk * x[j] * (1 - x[j]);
                }
                gE_wij *= alpha;
                hN[j].biais += gE_wij;
                hN[j].weights[i] += gE_wij * xi;
            }
        }
    }

    private float getActivationDerivative(Layer layer, float z) {
        switch (layer.getActivation()) {
            case RELU:
                return Neuron.reluDerivative(z);
            case LINEAR:
                return Neuron.linearDerivative(z);
            case SIGMOID:
            default:
                return Neuron.sigmoidDerivative(z);
        }
    }

    // Copy weights from another network (for target network in DQN)
    public Network copy() {
        Network copy;
        if (hiddenLayer2 != null) {
            copy = new Network(inputSize, hiddenLayer1.nCount, hiddenLayer2.nCount, outputLayer.nCount);
            copy.hiddenLayer2.copyWeightsFrom(this.hiddenLayer2);
        } else {
            copy = new Network(inputSize, hiddenLayer1.nCount, outputLayer.nCount);
        }
        
        copy.alpha = this.alpha;
        copy.hiddenLayer1.copyWeightsFrom(this.hiddenLayer1);
        copy.outputLayer.copyWeightsFrom(this.outputLayer);
        
        return copy;
    }

    public void copyWeightsFrom(Network other) {
        this.alpha = other.alpha;
        this.hiddenLayer1.copyWeightsFrom(other.hiddenLayer1);
        if (this.hiddenLayer2 != null && other.hiddenLayer2 != null) {
            this.hiddenLayer2.copyWeightsFrom(other.hiddenLayer2);
        }
        this.outputLayer.copyWeightsFrom(other.outputLayer);
    }

    // Builder pattern for easier construction
    public static class Builder {
        private int inputSize;
        private int hidden1Size;
        private int hidden2Size = -1;
        private int outputSize;
        private float learningRate = 0.001f;

        public Builder() {}

        public Builder inputSize(int size) {
            this.inputSize = size;
            return this;
        }

        public Builder hiddenSize(int size) {
            this.hidden1Size = size;
            return this;
        }

        public Builder addHiddenLayer(int size) {
            if (this.hidden1Size == 0) {
                this.hidden1Size = size;
            } else {
                this.hidden2Size = size;
            }
            return this;
        }

        public Builder outputSize(int size) {
            this.outputSize = size;
            return this;
        }

        public Builder learningRate(float rate) {
            this.learningRate = rate;
            return this;
        }

        public Network build() {
            Network network;
            if (hidden2Size > 0) {
                network = new Network(inputSize, hidden1Size, hidden2Size, outputSize);
            } else {
                network = new Network(inputSize, hidden1Size, outputSize);
            }
            network.alpha = learningRate;
            return network;
        }
    }
}

