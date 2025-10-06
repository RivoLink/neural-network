package mg.rivolink.ai;

import java.util.List;
import java.util.Random;

/**
 * Deep Q-Network Agent
 * Implements DQN with experience replay and target network
 */
public class DQNAgent {
    
    private final Network qNetwork;
    private final Network targetNetwork;
    private final ExperienceReplay replayBuffer;
    
    private float epsilon;
    private final float epsilonMin;
    private final float epsilonDecay;
    private final float gamma; // Discount factor
    private final int batchSize;
    private final int targetUpdateFrequency;
    private final boolean useSoftUpdate;
    private final float tau; // Soft update parameter
    
    private int stepCount = 0;
    private final Random random;

    public static class Builder {
        private int stateSize;
        private int actionSize;
        private int[] hiddenLayers = {128, 128};
        
        private float learningRate = 0.001f;
        private float gamma = 0.99f;
        private float epsilon = 1.0f;
        private float epsilonMin = 0.01f;
        private float epsilonDecay = 0.995f;
        
        private int bufferSize = 10000;
        private int batchSize = 64;
        private int targetUpdateFrequency = 100;
        private boolean useSoftUpdate = false;
        private float tau = 0.001f;

        public Builder(int stateSize, int actionSize) {
            this.stateSize = stateSize;
            this.actionSize = actionSize;
        }

        public Builder hiddenLayers(int... layers) {
            this.hiddenLayers = layers;
            return this;
        }

        public Builder learningRate(float rate) {
            this.learningRate = rate;
            return this;
        }

        public Builder gamma(float gamma) {
            this.gamma = gamma;
            return this;
        }

        public Builder epsilon(float epsilon, float min, float decay) {
            this.epsilon = epsilon;
            this.epsilonMin = min;
            this.epsilonDecay = decay;
            return this;
        }

        public Builder bufferSize(int size) {
            this.bufferSize = size;
            return this;
        }

        public Builder batchSize(int size) {
            this.batchSize = size;
            return this;
        }

        public Builder targetUpdateFrequency(int frequency) {
            this.targetUpdateFrequency = frequency;
            return this;
        }

        public Builder useSoftUpdate(boolean use, float tau) {
            this.useSoftUpdate = use;
            this.tau = tau;
            return this;
        }

        public DQNAgent build() {
            // Build network architecture
            int[] layerSizes = new int[hiddenLayers.length + 1];
            System.arraycopy(hiddenLayers, 0, layerSizes, 0, hiddenLayers.length);
            layerSizes[layerSizes.length - 1] = actionSize;

            Network qNet = new Network(stateSize, Neuron.ActivationType.RELU, layerSizes);
            qNet.learningRate = learningRate;
            qNet.gradientClipValue = 1.0f;

            Network targetNet = qNet.copy();

            ExperienceReplay buffer = new ExperienceReplay(bufferSize);

            return new DQNAgent(qNet, targetNet, buffer, gamma, epsilon, 
                               epsilonMin, epsilonDecay, batchSize, 
                               targetUpdateFrequency, useSoftUpdate, tau);
        }
    }

    private DQNAgent(Network qNetwork, Network targetNetwork, ExperienceReplay replayBuffer,
                     float gamma, float epsilon, float epsilonMin, float epsilonDecay,
                     int batchSize, int targetUpdateFrequency, boolean useSoftUpdate, float tau) {
        this.qNetwork = qNetwork;
        this.targetNetwork = targetNetwork;
        this.replayBuffer = replayBuffer;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        this.batchSize = batchSize;
        this.targetUpdateFrequency = targetUpdateFrequency;
        this.useSoftUpdate = useSoftUpdate;
        this.tau = tau;
        this.random = new Random();
    }

    /**
     * Select action using epsilon-greedy policy
     */
    public int selectAction(float[] state) {
        // Exploration
        if (random.nextFloat() < epsilon) {
            return random.nextInt(qNetwork.getLayer(qNetwork.getLayerCount() - 1).nCount);
        }
        
        // Exploitation
        return getBestAction(state);
    }

    /**
     * Get best action (greedy)
     */
    public int getBestAction(float[] state) {
        float[] qValues = qNetwork.predict(state);
        return argmax(qValues);
    }

    /**
     * Store experience in replay buffer
     */
    public void remember(float[] state, int action, float reward, float[] nextState, boolean done) {
        replayBuffer.add(state, action, reward, nextState, done);
    }

    /**
     * Train the network on a batch of experiences
     */
    public float train() {
        if (!replayBuffer.canSample(batchSize)) {
            return 0;
        }

        List<ExperienceReplay.Experience> batch = replayBuffer.sample(batchSize);
        float totalLoss = 0;

        for (ExperienceReplay.Experience exp : batch) {
            // Get current Q-values
            float[] qValues = qNetwork.predict(exp.state);
            
            // Calculate target Q-value
            float target;
            if (exp.done) {
                target = exp.reward;
            } else {
                // Use target network for stability
                float[] nextQValues = targetNetwork.predict(exp.nextState);
                float maxNextQ = max(nextQValues);
                target = exp.reward + gamma * maxNextQ;
            }

            // Update only the Q-value for the taken action
            float[] targets = qValues.clone();
            targets[exp.action] = target;

            // Train
            totalLoss += qNetwork.train(exp.state, targets);
        }

        stepCount++;

        // Update target network
        if (useSoftUpdate) {
            targetNetwork.softUpdate(qNetwork, tau);
        } else if (stepCount % targetUpdateFrequency == 0) {
            targetNetwork.copyWeightsFrom(qNetwork);
        }

        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }

        return totalLoss / batchSize;
    }

    /**
     * Train for multiple steps
     */
    public float trainSteps(int steps) {
        float totalLoss = 0;
        int trainedSteps = 0;

        for (int i = 0; i < steps; i++) {
            if (replayBuffer.canSample(batchSize)) {
                totalLoss += train();
                trainedSteps++;
            }
        }

        return trainedSteps > 0 ? totalLoss / trainedSteps : 0;
    }

    // Utility methods
    private int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private float max(float[] array) {
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
            }
        }
        return maxValue;
    }

    // Getters
    public float getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = Math.max(epsilonMin, epsilon);
    }

    public int getStepCount() {
        return stepCount;
    }

    public int getBufferSize() {
        return replayBuffer.size();
    }

    public Network getQNetwork() {
        return qNetwork;
    }

    public Network getTargetNetwork() {
        return targetNetwork;
    }

    /**
     * Save current policy (for inference)
     */
    public Network getPolicy() {
        return qNetwork.copy();
    }

    /**
     * Load policy
     */
    public void loadPolicy(Network policy) {
        qNetwork.copyWeightsFrom(policy);
        targetNetwork.copyWeightsFrom(policy);
    }
}
