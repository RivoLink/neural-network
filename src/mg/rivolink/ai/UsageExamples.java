package mg.rivolink.ai;

/**
 * Examples demonstrating how to use the improved neural network library
 */
public class UsageExamples {

    /**
     * Example 1: Simple Classification Network
     */
    public static void classificationExample() {
        System.out.println("=== Classification Example ===");
        
        // Create network: 4 inputs -> 8 hidden -> 3 outputs
        Network network = new Network.Builder()
            .inputSize(4)
            .addHiddenLayer(8)
            .outputSize(3)
            .learningRate(0.01f)
            .hiddenActivation(Neuron.ActivationType.RELU)
            .build();

        // Training data (XOR-like problem)
        float[][] xTrain = {
            {0, 0, 0, 0},
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {1, 1, 0, 0}
        };
        
        float[][] yTrain = {
            {1, 0, 0},  // Class 0
            {0, 1, 0},  // Class 1
            {0, 1, 0},  // Class 1
            {0, 0, 1}   // Class 2
        };

        // Train
        network.train(xTrain, yTrain, 1000, true);

        // Predict
        float[] prediction = network.predict(1, 1, 0, 0);
        System.out.println("Prediction: " + argmax(prediction));
    }

    /**
     * Example 2: Deep Network with Regularization
     */
    public static void deepNetworkExample() {
        System.out.println("\n=== Deep Network with Regularization ===");
        
        Network network = new Network.Builder()
            .inputSize(10)
            .addHiddenLayer(64)
            .addHiddenLayer(32)
            .addHiddenLayer(16)
            .outputSize(2)
            .learningRate(0.001f)
            .l2Regularization(0.0001f)
            .gradientClipping(5.0f)
            .dropout(0, 0.2f)  // 20% dropout on first hidden layer
            .dropout(1, 0.2f)  // 20% dropout on second hidden layer
            .build();

        System.out.println("Network created with " + network.getLayerCount() + " layers");
        System.out.println("L2 regularization: " + network.l2Lambda);
        System.out.println("Gradient clipping: " + network.gradientClipValue);
    }

    /**
     * Example 3: DQN Agent for Reinforcement Learning
     */
    public static void dqnExample() {
        System.out.println("\n=== DQN Agent Example ===");
        
        // Create DQN agent
        // State: 4 dimensions (e.g., CartPole: position, velocity, angle, angular velocity)
        // Actions: 2 (left, right)
        DQNAgent agent = new DQNAgent.Builder(4, 2)
            .hiddenLayers(128, 64)
            .learningRate(0.001f)
            .gamma(0.99f)
            .epsilon(1.0f, 0.01f, 0.995f)
            .bufferSize(100000)
            .batchSize(64)
            .targetUpdateFrequency(100)
            .build();

        // Simulate training loop
        int episodes = 10;
        for (int episode = 0; episode < episodes; episode++) {
            float[] state = getInitialState();
            float totalReward = 0;
            boolean done = false;
            int steps = 0;

            while (!done && steps < 500) {
                // Select action
                int action = agent.selectAction(state);
                
                // Take action in environment (simulated)
                float[] nextState = simulateStep(state, action);
                float reward = calculateReward(state, action);
                done = isTerminal(state);

                // Store experience
                agent.remember(state, action, reward, nextState, done);

                // Train
                if (agent.getBufferSize() >= 64) {
                    agent.train();
                }

                state = nextState;
                totalReward += reward;
                steps++;
            }

            System.out.printf("Episode %d: Steps=%d, Reward=%.2f, Epsilon=%.4f, Buffer=%d%n",
                episode + 1, steps, totalReward, agent.getEpsilon(), agent.getBufferSize());
        }
    }

    /**
     * Example 4: Using Soft Updates for Target Network
     */
    public static void softUpdateExample() {
        System.out.println("\n=== Soft Update Example ===");
        
        DQNAgent agent = new DQNAgent.Builder(8, 4)
            .hiddenLayers(64, 32)
            .useSoftUpdate(true, 0.001f)  // Soft update with tau=0.001
            .build();

        System.out.println("Using soft updates instead of hard updates");
        System.out.println("This provides more stable learning");
    }

    /**
     * Example 5: Transfer Learning (copying weights)
     */
    public static void transferLearningExample() {
        System.out.println("\n=== Transfer Learning Example ===");
        
        // Train a network on task A
        Network networkA = new Network.Builder()
            .inputSize(10)
            .addHiddenLayer(32)
            .addHiddenLayer(16)
            .outputSize(5)
            .build();

        // ... train networkA ...

        // Create new network for task B with same architecture
        Network networkB = new Network.Builder()
            .inputSize(10)
            .addHiddenLayer(32)
            .addHiddenLayer(16)
            .outputSize(3)  // Different output size
            .build();

        // Copy hidden layer weights from A to B
        networkB.getLayer(0).copyWeightsFrom(networkA.getLayer(0));
        networkB.getLayer(1).copyWeightsFrom(networkA.getLayer(1));
        // Output layer has different size, so we don't copy it

        System.out.println("Transferred learning from network A to B");
    }

    /**
     * Example 6: Different Activation Functions
     */
    public static void activationFunctionsExample() {
        System.out.println("\n=== Activation Functions Example ===");
        
        // Test different activations
        float z = 0.5f;
        
        System.out.println("z = " + z);
        System.out.println("Sigmoid: " + Neuron.sigmoid(z));
        System.out.println("ReLU: " + Neuron.relu(z));
        System.out.println("Leaky ReLU: " + Neuron.leakyRelu(z));
        System.out.println("Tanh: " + Neuron.tanh(z));

        // Create network with tanh activation
        Network network = new Network(4, Neuron.ActivationType.TANH, 8, 8, 2);
        System.out.println("\nCreated network with Tanh activation");
    }

    // Helper methods for simulation
    private static float[] getInitialState() {
        return new float[]{0, 0, 0, 0};
    }

    private static float[] simulateStep(float[] state, int action) {
        // Simulate environment step (dummy implementation)
        float[] nextState = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            nextState[i] = state[i] + (action == 0 ? -0.1f : 0.1f);
        }
        return nextState;
    }

    private static float calculateReward(float[] state, int action) {
        // Simple reward function (dummy implementation)
        return -Math.abs(state[0]);
    }

    private static boolean isTerminal(float[] state) {
        // Check if episode should end (dummy implementation)
        return Math.abs(state[0]) > 2.0f;
    }

    private static int argmax(float[] array) {
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

    public static void main(String[] args) {
        classificationExample();
        deepNetworkExample();
        dqnExample();
        softUpdateExample();
        transferLearningExample();
        activationFunctionsExample();
    }
}
