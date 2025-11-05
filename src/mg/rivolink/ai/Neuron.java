package mg.rivolink.ai;

import java.io.Serializable;
import java.util.Random;

public class Neuron implements Serializable {

    private static final long serialVersionUID = 1L;

    static final Random r = new Random();

    public float bias;

    public final int size;
    public final float weights[];

    private float lastZ;
    private float lastOutput;

    public enum Activation {
        SIGMOID, RELU, LEAKY_RELU, TANH, LINEAR, SOFTMAX
    }

    public Neuron(int size) {
        this.size = size;
        this.weights = new float[size];
        initializeWeights();
    }

    // He initialization, better for ReLU
    // value: [-1, 1] * sqrt(2/size)
    private void initializeWeights() {
        float scale = (float)Math.sqrt(2.0 / size);
        bias = (r.nextFloat() * 2 - 1) * scale;
        for (int i = 0; i < size; i++) {
            weights[i] = (r.nextFloat() * 2 - 1) * scale;
        }
    }

    // Xavier initialization, better for sigmoid/tanh/softmax
    // value: [-1, 1] * sqrt(1/size)
    public void initializeXavier() {
        float scale = (float)Math.sqrt(1.0 / size);
        bias = (r.nextFloat() * 2 - 1) * scale;
        for (int i = 0; i < size; i++) {
            weights[i] = (r.nextFloat() * 2 - 1) * scale;
        }
    }

    public float getLastZ() {
        return lastZ;
    }

    public float getLastOutput() {
        return lastOutput;
    }

    public float computeOutput(float[] inputs, Activation activation) {
        lastZ = dot(inputs, weights) + bias;
        lastOutput = applyActivation(lastZ, activation);
        return lastOutput;
    }

    private float applyActivation(float z, Activation activation) {
        switch (activation) {
            case RELU:
                return relu(z);
            case LEAKY_RELU:
                return leakyRelu(z);
            case TANH:
                return tanh(z);
            case LINEAR:
                return z;
            case SOFTMAX:
                return z; // Handled at layer level
            case SIGMOID:
            default:
                return sigmoid(z);
        }
    }

    public static float sigmoid(float z) {
        z = Math.max(-88f, Math.min(88f, z));
        return (float)(1.0 / (1.0 + Math.exp(-z)));
    }

    public static float sigmoidDerivative(float z) {
        float s = sigmoid(z);
        return s * (1 - s);
    }

    public static float relu(float z) {
        return Math.max(0, z);
    }

    public static float reluDerivative(float z) {
        return z > 0 ? 1 : 0;
    }

    public static float leakyRelu(float z) {
        return z > 0 ? z : 0.01f * z;
    }

    public static float leakyReluDerivative(float z) {
        return z > 0 ? 1 : 0.01f;
    }

    public static float tanh(float z) {
        return (float)Math.tanh(z);
    }

    public static float tanhDerivative(float z) {
        float t = tanh(z);
        return 1 - t * t;
    }

    public static float linearDerivative(float z) {
        return 1;
    }

    public static float getActivationDerivative(float z, Activation activation) {
        switch (activation) {
            case RELU:
                return reluDerivative(z);
            case LEAKY_RELU:
                return leakyReluDerivative(z);
            case TANH:
                return tanhDerivative(z);
            case LINEAR:
                return linearDerivative(z);
            case SOFTMAX:
                return 1; // Handled at layer level
            case SIGMOID:
            default:
                return sigmoidDerivative(z);
        }
    }

    public static float dot(float[] x, float[] w) {
        if (x.length != w.length) {
            throw new IllegalArgumentException("Array lengths must match");
        }

        float dot = 0;
        for (int i = 0; i < x.length; i++) {
            dot += x[i] * w[i];
        }
        return dot;
    }

    public void copyWeightsFrom(Neuron other) {
        if (this.size != other.size) {
            throw new IllegalArgumentException("Neuron sizes don't match");
        }

        this.bias = other.bias;
        System.arraycopy(other.weights, 0, this.weights, 0, size);
    }

    public void softUpdate(Neuron other, float tau) {
        this.bias = tau * other.bias + (1 - tau) * this.bias;
        for (int i = 0; i < size; i++) {
            this.weights[i] = tau * other.weights[i] + (1 - tau) * this.weights[i];
        }
    }

    // Gradient descent
    // b -= lr * delta, w -= lr * (delta * input)
    public void updateWeights(float[] gradients, float[] inputs, float learningRate, float maxGradient) {
        float biasGrad = Math.max(-maxGradient, Math.min(maxGradient, gradients[0]));
        bias -= learningRate * biasGrad;

        for (int i = 0; i < size; i++) {
            float weightGrad = Math.max(-maxGradient, Math.min(maxGradient, gradients[i + 1] * inputs[i]));
            weights[i] -= learningRate * weightGrad;
        }
    }

}
