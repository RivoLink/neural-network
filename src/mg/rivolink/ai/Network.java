package mg.rivolink.ai;

import java.io.Serializable;

public class Network implements Serializable {

    private static final long serialVersionUID = 1L;

    public float tau = 0.01f;        // For soft updates in DQN
    public float alpha = 0.1f;       // Learning rate
    public float maxGradient = 1.0f; // For gradient clipping

    public final int inputSize;
    public final Layer hiddenLayer1;
    public final Layer hiddenLayer2;
    public final Layer outputLayer;

    public Network(int inputSize, int hidden1Size, int hidden2Size, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayer1 = new Layer(inputSize, hidden1Size, Neuron.Activation.RELU);
        this.hiddenLayer2 = new Layer(hidden1Size, hidden2Size, Neuron.Activation.RELU);
        this.outputLayer = new Layer(hidden2Size, outputSize, Neuron.Activation.LINEAR);
    }

    public Network(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayer1 = new Layer(inputSize, hiddenSize, Neuron.Activation.SIGMOID);
        this.hiddenLayer2 = null;
        this.outputLayer = new Layer(hiddenSize, outputSize, Neuron.Activation.SIGMOID);
    }

    public float[] predict(float[] inputs) {
        return forward(inputs);
    }

    private float[] forward(float[] inputs) {
        if (inputs.length != inputSize) {
            throw new IllegalArgumentException(
                "Input size mismatch: expected " + inputSize + ", got " + inputs.length
            );
        }

        hiddenLayer1.setInputs(inputs);
        float[] h1 = hiddenLayer1.forward();

        if (hiddenLayer2 != null) {
            hiddenLayer2.setInputs(h1);
            float[] h2 = hiddenLayer2.forward();
            outputLayer.setInputs(h2);
            return outputLayer.forward();
        } else {
            outputLayer.setInputs(h1);
            return outputLayer.forward();
        }
    }

    // Train with float target (classification with one-hot)
    public void train(float[] inputs, float[] target) {
        forward(inputs);
        backpropagation(target);
    }

    // Train with int target (classification)
    public void train(float[] inputs, int[] target) {
        float[] targetFloat = new float[target.length];
        for (int i = 0; i < target.length; i++) {
            targetFloat[i] = target[i];
        }
        this.train(inputs, targetFloat);
    }

    // Batch training for regression
    public void train(float[][] xtrains, float[][] ytrains, int epochs) {
        int size = Math.min(xtrains.length, ytrains.length);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < size; i++) {
                this.train(xtrains[i], ytrains[i]);
            }
        }
    }

    // Batch training for classification
    public void train(float[][] xtrains, int[][] ytrains, int epochs) {
        int size = Math.min(xtrains.length, ytrains.length);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < size; i++) {
                this.train(xtrains[i], ytrains[i]);
            }
        }
    }

    private void backpropagation(float[] target) {
        float[] yhat = outputLayer.getOutputs();
        boolean isSoftmax = outputLayer.getActivation() == Neuron.Activation.SOFTMAX;

        if (hiddenLayer2 != null) {
            // 3-layer network
            float[] h1 = hiddenLayer1.getOutputs();
            float[] h2 = hiddenLayer2.getOutputs();
            float[] h1Input = hiddenLayer1.getInputs();

            float[] z1 = hiddenLayer1.getLastZValues();
            float[] z2 = hiddenLayer2.getLastZValues();
            float[] zOut = outputLayer.getLastZValues();

            Neuron[] h1N = hiddenLayer1.neurons;
            Neuron[] h2N = hiddenLayer2.neurons;
            Neuron[] oN = outputLayer.neurons;

            // Output layer gradients
            float[] deltaOut = new float[oN.length];
            for (int k = 0; k < oN.length; k++) {
                if (isSoftmax) {
                    // Softmax + Cross-Entropy: delta = yhat - target
                    deltaOut[k] = yhat[k] - target[k];
                } else {
                    // MSE: delta = (yhat - target) * activation'(z)
                    float error = yhat[k] - target[k];
                    float derivative = Neuron.getActivationDerivative(zOut[k], outputLayer.getActivation());
                    deltaOut[k] = error * derivative;
                }
            }

            // Hidden layer 2 gradients
            float[] deltaH2 = new float[h2N.length];
            for (int j = 0; j < h2N.length; j++) {
                float error = 0;
                for (int k = 0; k < oN.length; k++) {
                    error += deltaOut[k] * oN[k].weights[j];
                }
                float derivative = Neuron.getActivationDerivative(z2[j], hiddenLayer2.getActivation());
                deltaH2[j] = error * derivative;
            }

            // Hidden layer 1 gradients
            float[] deltaH1 = new float[h1N.length];
            for (int i = 0; i < h1N.length; i++) {
                float error = 0;
                for (int j = 0; j < h2N.length; j++) {
                    error += deltaH2[j] * h2N[j].weights[i];
                }
                float derivative = Neuron.getActivationDerivative(z1[i], hiddenLayer1.getActivation());
                deltaH1[i] = error * derivative;
            }

            // Update layers with gradient descent
            updateLayerWeights(outputLayer, deltaOut, h2, alpha, maxGradient);
            updateLayerWeights(hiddenLayer2, deltaH2, h1, alpha, maxGradient);
            updateLayerWeights(hiddenLayer1, deltaH1, h1Input, alpha, maxGradient);

        } else {
            // 2-layer network
            float[] h1 = hiddenLayer1.getOutputs();
            float[] h1Input = hiddenLayer1.getInputs();
            float[] z1 = hiddenLayer1.getLastZValues();
            float[] zOut = outputLayer.getLastZValues();

            Neuron[] h1N = hiddenLayer1.neurons;
            Neuron[] oN = outputLayer.neurons;

            // Output layer gradients
            float[] deltaOut = new float[oN.length];
            for (int k = 0; k < oN.length; k++) {
                if (isSoftmax) {
                    // Softmax + Cross-Entropy: delta = yhat - target
                    deltaOut[k] = yhat[k] - target[k];
                } else {
                    // MSE: delta = (yhat - target) * activation'(z)
                    float error = yhat[k] - target[k];
                    float derivative = Neuron.getActivationDerivative(zOut[k], outputLayer.getActivation());
                    deltaOut[k] = error * derivative;
                }
            }

            // Hidden layer 1 gradients
            float[] deltaH1 = new float[h1N.length];
            for (int i = 0; i < h1N.length; i++) {
                float error = 0;
                for (int k = 0; k < oN.length; k++) {
                    error += deltaOut[k] * oN[k].weights[i];
                }
                float derivative = Neuron.getActivationDerivative(z1[i], hiddenLayer1.getActivation());
                deltaH1[i] = error * derivative;
            }

            // Update layers with gradient descent
            updateLayerWeights(outputLayer, deltaOut, h1, alpha, maxGradient);
            updateLayerWeights(hiddenLayer1, deltaH1, h1Input, alpha, maxGradient);
        }
    }

    // Gradient descent
    // b -= lr * delta, w -= lr * (delta * input)
    private void updateLayerWeights(Layer layer, float[] deltas, float[] inputs, float lr, float maxGrad) {
        for (int i = 0; i < layer.neuronCount; i++) {
            float biasGrad = Math.max(-maxGrad, Math.min(maxGrad, deltas[i]));
            layer.neurons[i].bias -= lr * biasGrad;

            for (int j = 0; j < layer.inputSize; j++) {
                float weightGrad = Math.max(-maxGrad, Math.min(maxGrad, deltas[i] * inputs[j]));
                layer.neurons[i].weights[j] -= lr * weightGrad;
            }
        }
    }

    // Copy network weights
    public Network copy() {
        Network copy;

        if (hiddenLayer2 != null) {
            copy = new Network(inputSize, hiddenLayer1.neuronCount, hiddenLayer2.neuronCount, outputLayer.neuronCount);
            copy.hiddenLayer2.copyWeightsFrom(this.hiddenLayer2);
        } else {
            copy = new Network(inputSize, hiddenLayer1.neuronCount, outputLayer.neuronCount);
        }

        copy.tau = this.tau;
        copy.alpha = this.alpha;
        copy.maxGradient = this.maxGradient;

        copy.hiddenLayer1.copyWeightsFrom(this.hiddenLayer1);
        copy.outputLayer.copyWeightsFrom(this.outputLayer);

        return copy;
    }

    // Set network weights from another one
    public void copyWeightsFrom(Network other) {
        this.tau = other.tau;
        this.alpha = other.alpha;
        this.maxGradient = other.maxGradient;

        this.hiddenLayer1.copyWeightsFrom(other.hiddenLayer1);

        if ((this.hiddenLayer2 != null) && (other.hiddenLayer2 != null)) {
            this.hiddenLayer2.copyWeightsFrom(other.hiddenLayer2);
        }

        this.outputLayer.copyWeightsFrom(other.outputLayer);
    }

    // Soft update for target networks (DQN)
    public void softUpdate(Network other) {
        hiddenLayer1.softUpdate(other.hiddenLayer1, tau);

        if (hiddenLayer2 != null && other.hiddenLayer2 != null) {
            hiddenLayer2.softUpdate(other.hiddenLayer2, tau);
        }

        outputLayer.softUpdate(other.outputLayer, tau);
    }

    // Builder pattern
    public static class Builder {
        private int inputSize;
        private int hidden1Size;
        private int hidden2Size = -1;
        private int outputSize;

        private float tau = 0.01f;
        private float learningRate = 0.1f;
        private float maxGradient = 1.0f;

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

        public Builder maxGradient(float max) {
            this.maxGradient = max;
            return this;
        }

        public Builder tau(float t) {
            this.tau = t;
            return this;
        }

        public Network build() {
            Network network;

            if (hidden2Size > 0) {
                network = new Network(inputSize, hidden1Size, hidden2Size, outputSize);
            } else {
                network = new Network(inputSize, hidden1Size, outputSize);
            }

            network.tau = tau;
            network.alpha = learningRate;
            network.maxGradient = maxGradient;

            return network;
        }
    }

}
