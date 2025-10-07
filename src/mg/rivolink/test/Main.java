package mg.rivolink.test;

import mg.rivolink.ai.Network;

public class Main {

    public static void main(String[] args) {
        System.out.println("=== Neural Network Test ===");
        System.out.println();

        System.out.println("Test: XOR Problem");
        System.out.println("Training a 2-4-1 network to learn XOR...");
        System.out.println();

        Network xorNetwork = new Network(2, 4, 1);

        float[][] xorInputs = {
            {0f, 0f},
            {0f, 1f},
            {1f, 0f},
            {1f, 1f},
        };

        int[][] xorTargets = {
            {0},
            {1},
            {1},
            {0},
        };

        // Train the network
        xorNetwork.train(xorInputs, xorTargets, 10000);

        System.out.println("After training:");

        for (int i = 0; i < xorInputs.length; i++) {
            float[] output = xorNetwork.predict(xorInputs[i]);

            System.out.printf("Input: [%d, %d] => Output: %.4f (Expected: %d)\n",
                (int)xorInputs[i][0],
                (int)xorInputs[i][1],
                (float)output[0],
                (int)xorTargets[i][0]
            );
        }

        System.out.println();
        System.out.println("=== All Tests Completed ===");
    }

}
