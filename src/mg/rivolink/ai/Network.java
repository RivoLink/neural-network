package mg.rivolink.ai;

public class Network {

    public static final float LEARNING_RATE = 0.1f;

    public final int inputSize;
    public final Layer hiddenLayer;
    public final Layer outputLayer;

    public Network(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayer = new Layer(inputSize, hiddenSize);
        this.outputLayer = new Layer(hiddenSize, outputSize);
    }

    public float[] predict(int bits) {
        return setInput(bits).getOutputs();
    }

    public float[] predict(float... inputs) {
        return setInput(inputs).getOutputs();
    }

    public void train(int bits, int[] target) {
        setInput(bits);
        training(target);
    }

    public void train(float[] inputs, int[] target) {
        setInput(inputs);
        training(target);
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
        hiddenLayer.setInputs(bits);
        return this;
    }

    private Network setInput(float... inputs){
		hiddenLayer.setInputs(inputs);
		return this;
	}

    private float[] getOutputs() {
        float[] hiddenOutputs = hiddenLayer.getOutputs();
        return outputLayer.setInputs(hiddenOutputs).getOutputs();
    }

    private void training(int[] target) {
        float[] yhat = getOutputs();
        float[] x = hiddenLayer.getOutputs();
        int[] y = target;

        float alpha = LEARNING_RATE;

        Neuron[] hN = hiddenLayer.neurons;
        Neuron[] oN = outputLayer.neurons;

        // Initial inputs are always the same
        float[] inputs = hN[0].inputs;

        float gE_wij;
        float gE_wjk;

        for (int j = 0 ; j < hN.length ; j++) {
            for(int k = 0 ; k < oN.length ; k++){
                gE_wjk = alpha * (y[k] - yhat[k]) * yhat[k] * (1 - yhat[k]);
                oN[k].bias += gE_wjk;
                oN[k].weights[j] += gE_wjk * x[j];
            }

            for (int i = 0 ; i < inputSize ; i++) {
                gE_wij = 0;
                float xi = inputs[i];

                for (int k = 0 ; k < oN.length ; k++) {
                    float wjk = oN[k].weights[j];
                    gE_wij += (y[k] - yhat[k]) * yhat[k] * (1 - yhat[k]) * wjk * x[j] * (1 - x[j]);
                }

                gE_wij *= alpha;
                hN[j].bias += gE_wij;
                hN[j].weights[i] += gE_wij * xi;
            }
        }
    }

}
