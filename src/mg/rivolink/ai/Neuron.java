package mg.rivolink.ai;

import java.util.Random;

public class Neuron {
    static final Random r = new Random();

    public float biais;
    public final float inputs[];
    public final float weights[];
    public final int size;

    // Store last output for backpropagation
    private float lastOutput;
    private float lastZ; // Before activation

    public Neuron(int size) {
        this.size = size;
        this.inputs = new float[size];
        this.weights = new float[size];

        // He initialization for better convergence with ReLU
        float scale = (float)Math.sqrt(2.0 / size);
        biais = (r.nextFloat() * 2 - 1) * scale;
        for (int i = 0; i < size; i++) {
            weights[i] = (r.nextFloat() * 2 - 1) * scale;
        }
    }

    public Neuron setInput(int bits) {
        for (int i = 0; i < size; i++) {
            inputs[size - (i + 1)] = (bits & (1 << i)) >> i;
        }
        return this;
    }

    public Neuron setInput(float... inputs) {
        for (int i = 0; i < size; i++) {
            this.inputs[i] = inputs[i];
        }
        return this;
    }

    public float getOutput() {
        lastZ = dot(inputs, weights) + biais;
        lastOutput = sigmoid(lastZ);
        return lastOutput;
    }

    public float getOutputReLU() {
        lastZ = dot(inputs, weights) + biais;
        lastOutput = relu(lastZ);
        return lastOutput;
    }

    public float getOutputLinear() {
        lastZ = dot(inputs, weights) + biais;
        lastOutput = lastZ;
        return lastOutput;
    }

    public float getLastOutput() {
        return lastOutput;
    }

    public float getLastZ() {
        return lastZ;
    }

    public static float sigmoid(float z) {
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

    public static float linearDerivative(float z) {
        return 1;
    }

    public static int[] toBits(int n) {
        int size = Integer.toBinaryString(n).length();
        int[] bits = new int[size];
        for (int i = 0; i < size; i++) {
            bits[size - (i + 1)] = (n & (1 << i)) >> i;
        }
        return bits;
    }

    public static float dot(float[] x, float[] w) {
        if (x.length != w.length)
            return 0;

        float dot = 0;
        for (int i = 0; i < x.length; i++) {
            dot += x[i] * w[i];
        }
        return dot;
    }

    // Copy neuron weights and bias
    public void copyWeightsFrom(Neuron other) {
        this.biais = other.biais;
        for (int i = 0; i < size; i++) {
            this.weights[i] = other.weights[i];
        }
    }
}
