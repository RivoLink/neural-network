package mg.rivolink.test;

import mg.rivolink.ai.Network;

public class Network2HiddenTest {

    public static void main(String[] args) {
        System.out.println("=== Neural Network Test (2 Hidden Layers) ===");
        System.out.println();

        testXORProblem();
        System.out.println("\n----------\n");

        testANDProblem();
        System.out.println("\n----------\n");

        testORProblem();
        System.out.println();

        System.out.println("=== All Tests Completed ===");
    }

    private static void testXORProblem() {
        System.out.println("Test 1: XOR Problem (2-8-4-1 network)");
        System.out.println("Training a 2-8-4-1 network to learn XOR...");
        System.out.println();

        // 2 inputs, 8 neurons in hidden layer 1, 4 neurons in hidden layer 2, 1 output
        Network xorNetwork = new Network(2, 8, 4, 1);

        // Configure learning parameters
        xorNetwork.alpha = 0.1f;
        xorNetwork.tau = 0.01f;
        xorNetwork.maxGradient = 1.0f;

        float[][] xorInputs = {
            {0f, 0f},
            {0f, 1f},
            {1f, 0f},
            {1f, 1f},
        };

        float[][] xorTargets = {
            {0f},
            {1f},
            {1f},
            {0f},
        };

        // Train the network
        long startTime = System.currentTimeMillis();
        xorNetwork.train(xorInputs, xorTargets, 10000);
        long trainingTime = System.currentTimeMillis() - startTime;

        System.out.println("After training (Time: " + trainingTime + "ms):");
        System.out.println();

        float totalError = 0;
        for (int i = 0; i < xorInputs.length; i++) {
            float[] output = xorNetwork.predict(xorInputs[i]);
            float expected = xorTargets[i][0];
            float error = Math.abs(output[0] - expected);
            totalError += error;

            System.out.printf("Input: [%.0f, %.0f] => Output: %.4f (Expected: %.0f) | Error: %.4f\n",
                xorInputs[i][0],
                xorInputs[i][1],
                output[0],
                expected,
                error
            );
        }

        System.out.printf("Average Error: %.4f", totalError / xorInputs.length);
        System.out.println();
    }

    private static void testANDProblem() {
        System.out.println("Test 2: AND Problem (2-6-3-1 network)");
        System.out.println("Training a 2-6-3-1 network to learn AND...");
        System.out.println();

        Network andNetwork = new Network(2, 6, 3, 1);

        andNetwork.alpha = 0.1f;
        andNetwork.tau = 0.01f;
        andNetwork.maxGradient = 1.0f;

        float[][] andInputs = {
            {0f, 0f},
            {0f, 1f},
            {1f, 0f},
            {1f, 1f},
        };

        float[][] andTargets = {
            {0f},
            {0f},
            {0f},
            {1f},
        };

        long startTime = System.currentTimeMillis();
        andNetwork.train(andInputs, andTargets, 5000);
        long trainingTime = System.currentTimeMillis() - startTime;

        System.out.println("After training (Time: " + trainingTime + "ms):");
        System.out.println();

        float totalError = 0;
        for (int i = 0; i < andInputs.length; i++) {
            float[] output = andNetwork.predict(andInputs[i]);
            float expected = andTargets[i][0];
            float error = Math.abs(output[0] - expected);
            totalError += error;

            System.out.printf("Input: [%.0f, %.0f] => Output: %.4f (Expected: %.0f) | Error: %.4f\n",
                andInputs[i][0],
                andInputs[i][1],
                output[0],
                expected,
                error
            );
        }

        System.out.printf("Average Error: %.4f", totalError / andInputs.length);
        System.out.println();
    }

    private static void testORProblem() {
        System.out.println("Test 3: OR Problem (2-5-3-1 network)");
        System.out.println("Training a 2-5-3-1 network to learn OR...");
        System.out.println();

        Network orNetwork = new Network(2, 5, 3, 1);

        orNetwork.alpha = 0.1f;
        orNetwork.tau = 0.01f;
        orNetwork.maxGradient = 1.0f;

        float[][] orInputs = {
            {0f, 0f},
            {0f, 1f},
            {1f, 0f},
            {1f, 1f},
        };

        float[][] orTargets = {
            {0f},
            {1f},
            {1f},
            {1f},
        };

        long startTime = System.currentTimeMillis();
        orNetwork.train(orInputs, orTargets, 5000);
        long trainingTime = System.currentTimeMillis() - startTime;

        System.out.println("After training (Time: " + trainingTime + "ms):");
        System.out.println();

        float totalError = 0;
        for (int i = 0; i < orInputs.length; i++) {
            float[] output = orNetwork.predict(orInputs[i]);
            float expected = orTargets[i][0];
            float error = Math.abs(output[0] - expected);
            totalError += error;

            System.out.printf("Input: [%.0f, %.0f] => Output: %.4f (Expected: %.0f) | Error: %.4f\n",
                orInputs[i][0],
                orInputs[i][1],
                output[0],
                expected,
                error
            );
        }

        System.out.printf("Average Error: %.4f", totalError / orInputs.length);
        System.out.println();
    }

}
