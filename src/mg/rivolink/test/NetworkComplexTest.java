package mg.rivolink.test;

import mg.rivolink.ai.Network;

public class NetworkComplexTest {

    public static void main(String[] args) {
        System.out.println("=== Neural Network Test (Complex Problems - 2 Hidden Layers) ===");
        System.out.println();

        testMajorityFunction();
        System.out.println("\n----------\n");

        testParityFunction();
        System.out.println("\n----------\n");

        testIrisClassification();
        System.out.println("\n----------\n");

        testFunctionApproximation();
        System.out.println();

        System.out.println("=== All Tests Completed ===");
    }

    private static void testMajorityFunction() {
        System.out.println("Test 1: Majority Function (3-8-4-1 network)");
        System.out.println("Training network to learn majority voting (2+ inputs = 1)...");
        System.out.println();

        Network majorityNetwork = new Network(3, 8, 4, 1);

        majorityNetwork.alpha = 0.1f;
        majorityNetwork.tau = 0.01f;
        majorityNetwork.maxGradient = 1.0f;

        // All 8 combinations of 3 binary inputs
        float[][] majorityInputs = {
            {0f, 0f, 0f},  // 0 ones -> 0
            {0f, 0f, 1f},  // 1 one  -> 0
            {0f, 1f, 0f},  // 1 one  -> 0
            {0f, 1f, 1f},  // 2 ones -> 1
            {1f, 0f, 0f},  // 1 one  -> 0
            {1f, 0f, 1f},  // 2 ones -> 1
            {1f, 1f, 0f},  // 2 ones -> 1
            {1f, 1f, 1f},  // 3 ones -> 1
        };

        float[][] majorityTargets = {
            {0f}, {0f}, {0f}, {1f}, {0f}, {1f}, {1f}, {1f}
        };

        long startTime = System.currentTimeMillis();
        majorityNetwork.train(majorityInputs, majorityTargets, 15000);
        long trainingTime = System.currentTimeMillis() - startTime;

        System.out.println("After training (Time: " + trainingTime + "ms):");
        System.out.println();

        float totalError = 0;
        for (int i = 0; i < majorityInputs.length; i++) {
            float[] output = majorityNetwork.predict(majorityInputs[i]);
            float expected = majorityTargets[i][0];
            float cleanedOutput = cleanOutput(output[0]);
            float error = Math.abs(cleanedOutput - expected);
            totalError += error;

            int onesCount = (int)(majorityInputs[i][0] + majorityInputs[i][1] + majorityInputs[i][2]);
            System.out.printf("Input: [%.0f, %.0f, %.0f] (%d ones) => Output: %.4f (Expected: %.0f) | Error: %.4f\n",
                majorityInputs[i][0], majorityInputs[i][1], majorityInputs[i][2],
                onesCount, cleanedOutput, expected, error
            );
        }

        System.out.printf("Average Error: %.4f\n", totalError / majorityInputs.length);
    }

    private static void testParityFunction() {
        System.out.println("Test 2: Parity Function (4-12-6-1 network)");
        System.out.println("Training network to learn parity (odd number of 1s = 1)...");
        System.out.println();

        Network parityNetwork = new Network(4, 12, 6, 1);

        parityNetwork.alpha = 0.1f;
        parityNetwork.tau = 0.01f;
        parityNetwork.maxGradient = 1.0f;

        // All 16 combinations of 4 binary inputs
        float[][] parityInputs = new float[16][4];
        float[][] parityTargets = new float[16][1];

        for (int i = 0; i < 16; i++) {
            // Convert i to binary representation
            int onesCount = 0;
            for (int j = 0; j < 4; j++) {
                int bit = (i >> j) & 1;
                parityInputs[i][j] = bit;
                onesCount += bit;
            }
            // Parity: 1 if odd number of ones
            parityTargets[i][0] = (onesCount % 2 == 1) ? 1f : 0f;
        }

        long startTime = System.currentTimeMillis();
        parityNetwork.train(parityInputs, parityTargets, 20000);
        long trainingTime = System.currentTimeMillis() - startTime;

        System.out.println("After training (Time: " + trainingTime + "ms):");
        System.out.println();

        float totalError = 0;
        int correctCount = 0;
        for (int i = 0; i < parityInputs.length; i++) {
            float[] output = parityNetwork.predict(parityInputs[i]);
            float expected = parityTargets[i][0];
            float cleanedOutput = cleanOutput(output[0]);
            float error = Math.abs(cleanedOutput - expected);
            totalError += error;

            if (Math.round(cleanedOutput) == expected) {
                correctCount++;
            }

            int onesCount = (int)(parityInputs[i][0] + parityInputs[i][1] + parityInputs[i][2] + parityInputs[i][3]);
            System.out.printf("Input: [%.0f, %.0f, %.0f, %.0f] (%d ones) => Output: %.4f (Expected: %.0f) | Error: %.4f\n",
                parityInputs[i][0], parityInputs[i][1], parityInputs[i][2], parityInputs[i][3],
                onesCount, cleanedOutput, expected, error
            );
        }

        System.out.printf("Average Error: %.4f | Accuracy: %d/%d (%.1f%%)\n",
            totalError / parityInputs.length, correctCount, parityInputs.length,
            (correctCount * 100f) / parityInputs.length);
    }

    private static void testIrisClassification() {
        System.out.println("Test 3: Iris-like Classification (4-10-5-3 network)");
        System.out.println("Training network to classify 3 flower species...");
        System.out.println();

        Network irisNetwork = new Network(4, 10, 5, 3);

        irisNetwork.alpha = 0.1f;
        irisNetwork.tau = 0.01f;
        irisNetwork.maxGradient = 1.0f;

        // Simplified iris-like data (normalized to 0-1)
        float[][] irisInputs = {
            {0.2f, 0.3f, 0.1f, 0.0f}, // Setosa
            {0.3f, 0.4f, 0.15f, 0.05f}, // Setosa
            {0.25f, 0.35f, 0.12f, 0.02f}, // Setosa

            {0.7f, 0.5f, 0.6f, 0.4f}, // Versicolor
            {0.65f, 0.55f, 0.58f, 0.38f}, // Versicolor
            {0.72f, 0.48f, 0.62f, 0.42f}, // Versicolor

            {0.85f, 0.6f, 0.8f, 0.7f}, // Virginica
            {0.82f, 0.62f, 0.78f, 0.68f}, // Virginica
            {0.88f, 0.58f, 0.82f, 0.72f}, // Virginica
        };

        float[][] irisTargets = {
            {1f, 0f, 0f}, // Setosa
            {1f, 0f, 0f},
            {1f, 0f, 0f},

            {0f, 1f, 0f}, // Versicolor
            {0f, 1f, 0f},
            {0f, 1f, 0f},

            {0f, 0f, 1f}, // Virginica
            {0f, 0f, 1f},
            {0f, 0f, 1f},
        };

        long startTime = System.currentTimeMillis();
        irisNetwork.train(irisInputs, irisTargets, 20000);
        long trainingTime = System.currentTimeMillis() - startTime;

        System.out.println("After training (Time: " + trainingTime + "ms):");
        System.out.println();

        String[] speciesNames = {"Setosa", "Versicolor", "Virginica"};
        float totalError = 0;
        int correctCount = 0;

        for (int i = 0; i < irisInputs.length; i++) {
            float[] output = irisNetwork.predict(irisInputs[i]);

            // Clean all outputs
            for (int j = 0; j < output.length; j++) {
                output[j] = cleanOutput(output[j]);
            }

            // Find predicted class
            int predictedClass = 0;
            float maxOutput = output[0];
            for (int j = 1; j < output.length; j++) {
                if (output[j] > maxOutput) {
                    maxOutput = output[j];
                    predictedClass = j;
                }
            }

            // Find actual class
            int actualClass = 0;
            for (int j = 0; j < irisTargets[i].length; j++) {
                if (irisTargets[i][j] == 1f) {
                    actualClass = j;
                    break;
                }
            }

            if (predictedClass == actualClass) {
                correctCount++;
            }

            float error = 0;
            for (int j = 0; j < output.length; j++) {
                error += Math.abs(output[j] - irisTargets[i][j]);
            }
            totalError += error;

            System.out.printf("Input sample %d => Predicted: %s | Actual: %s | Outputs: [%.4f, %.4f, %.4f]\n",
                i + 1, speciesNames[predictedClass], speciesNames[actualClass],
                output[0], output[1], output[2]
            );
        }

        System.out.printf("Average Error: %.4f | Accuracy: %d/%d (%.1f%%)\n",
            totalError / irisInputs.length, correctCount, irisInputs.length,
            (correctCount * 100f) / irisInputs.length);
    }

    private static void testFunctionApproximation() {
        System.out.println("Test 4: Function Approximation (1-12-6-1 network)");
        System.out.println("Training network to approximate f(x) = sin(x) + cos(2x)...");
        System.out.println();

        Network funcNetwork = new Network(1, 12, 6, 1);

        funcNetwork.alpha = 0.1f;
        funcNetwork.tau = 0.01f;
        funcNetwork.maxGradient = 1.0f;

        // Generate training data for complex function
        int samples = 20;
        float[][] funcInputs = new float[samples][1];
        float[][] funcTargets = new float[samples][1];

        for (int i = 0; i < samples; i++) {
            float x = (i / (float)samples) * 2 * (float)Math.PI;
            funcInputs[i][0] = x / (2 * (float)Math.PI); // Normalize to 0-1

            float y = (float)(Math.sin(x) + Math.cos(2 * x));
            funcTargets[i][0] = (y + 2) / 4; // Normalize to 0-1
        }

        long startTime = System.currentTimeMillis();
        funcNetwork.train(funcInputs, funcTargets, 25000);
        long trainingTime = System.currentTimeMillis() - startTime;

        System.out.println("After training (Time: " + trainingTime + "ms):");
        System.out.println();

        float totalError = 0;
        for (int i = 0; i < funcInputs.length; i++) {
            float[] output = funcNetwork.predict(funcInputs[i]);
            float cleanedOutput = cleanOutput(output[0]);
            float expected = funcTargets[i][0];
            float error = Math.abs(cleanedOutput - expected);
            totalError += error;

            float x = funcInputs[i][0] * 2 * (float)Math.PI;
            float actualY = (float)(Math.sin(x) + Math.cos(2 * x));

            System.out.printf("x: %.4f => Output: %.4f (Expected: %.4f) | Actual f(x): %.4f | Error: %.4f\n",
                x, cleanedOutput, expected, actualY, error
            );
        }

        System.out.printf("Average Error: %.4f\n", totalError / funcInputs.length);
    }

    // Clean up negative zero and round very small values
    private static float cleanOutput(float value) {
        if (Math.abs(value) < 1e-6f) {
            return 0f;
        }
        return value;
    }

}
