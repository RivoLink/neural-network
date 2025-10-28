package mg.rivolink.test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import mg.rivolink.ai.Layer;
import mg.rivolink.ai.Network;
import mg.rivolink.ai.Neuron;
import mg.rivolink.io.NetworkIO;

public class NetworkIOTest {

    private static final Path MODEL_DIR = Paths.get("models");
    private static final float EPSILON = 1e-5f;

    public static void main(String[] args) {
        System.out.println("=== NetworkIO Smoke Tests ===");
        System.out.println();

        testBinaryRoundTrip();
        System.out.println("\n----------\n");

        testJsonRoundTrip();
        System.out.println("\n----------\n");

        testXmlRoundTrip();
        System.out.println("\n----------\n");

        testCopyAndMetadata();
        System.out.println();

        System.out.println("=== All Tests Completed ===");
    }

    private static void testBinaryRoundTrip() {
        System.out.println("Test 1: Binary save/load round-trip");
        try {
            Files.createDirectories(MODEL_DIR);
            Path modelPath = MODEL_DIR.resolve("network-io-roundtrip.bin");
            Files.deleteIfExists(modelPath);

            Network original = new Network(2, 3, 1);
            seedDeterministicWeights(original);

            float[] sampleInput = new float[] {0.25f, -0.75f};
            float[] originalOutput = original.predict(sampleInput);

            NetworkIO.save(original, modelPath.toString());
            System.out.println(" - Saved model to " + modelPath);

            Network restored = NetworkIO.load(modelPath.toString());
            float[] restoredOutput = restored.predict(sampleInput);

            if (!approxEquals(originalOutput, restoredOutput)) {
                throw new IllegalStateException("Round-trip outputs differ: "
                    + Arrays.toString(originalOutput) + " vs "
                    + Arrays.toString(restoredOutput));
            }

            System.out.println(" - Restored network matched original output "
                + Arrays.toString(restoredOutput));
            System.out.println(" - Metadata after save: "
                + NetworkIO.getMetadata(modelPath.toString()));
        } catch (Exception e) {
            throw new RuntimeException("Test 1 failed", e);
        }
    }

    private static void testJsonRoundTrip() {
        System.out.println("Test 2: JSON save/load round-trip");
        try {
            Files.createDirectories(MODEL_DIR);
            Path modelPath = MODEL_DIR.resolve("network-io-roundtrip.json");
            Files.deleteIfExists(modelPath);

            Network original = new Network(3, 5, 2, 1);
            seedDeterministicWeights(original);

            float[] sampleInput = new float[] {0.5f, -0.25f, 0.75f};
            float[] originalOutput = original.predict(sampleInput);

            NetworkIO.save(original, modelPath.toString(), NetworkIO.Format.JSON);
            System.out.println(" - Saved JSON model to " + modelPath);

            Network restored = NetworkIO.load(modelPath.toString());
            float[] restoredOutput = restored.predict(sampleInput);

            if (!approxEquals(originalOutput, restoredOutput)) {
                throw new IllegalStateException("JSON round-trip mismatch: "
                    + Arrays.toString(originalOutput) + " vs "
                    + Arrays.toString(restoredOutput));
            }

            System.out.println(" - Restored network matched original output "
                + Arrays.toString(restoredOutput));
        } catch (Exception e) {
            throw new RuntimeException("Test 2 failed", e);
        }
    }

    private static void testXmlRoundTrip() {
        System.out.println("Test 3: XML save/load round-trip");
        try {
            Files.createDirectories(MODEL_DIR);
            Path modelPath = MODEL_DIR.resolve("network-io-roundtrip.xml");
            Files.deleteIfExists(modelPath);

            Network original = new Network(4, 6, 3, 2);
            seedDeterministicWeights(original);

            float[] sampleInput = new float[] {0.1f, 0.2f, 0.3f, 0.4f};
            float[] originalOutput = original.predict(sampleInput);

            NetworkIO.save(original, modelPath.toString(), NetworkIO.Format.XML);
            System.out.println(" - Saved XML model to " + modelPath);

            Network restored = NetworkIO.load(modelPath.toString());
            float[] restoredOutput = restored.predict(sampleInput);

            if (!approxEquals(originalOutput, restoredOutput)) {
                throw new IllegalStateException("XML round-trip mismatch: "
                    + Arrays.toString(originalOutput) + " vs "
                    + Arrays.toString(restoredOutput));
            }

            System.out.println(" - Restored network matched original output "
                + Arrays.toString(restoredOutput));
        } catch (Exception e) {
            throw new RuntimeException("Test 3 failed", e);
        }
    }

    private static void testCopyAndMetadata() {
        System.out.println("Test 4: Copy and metadata");
        try {
            Files.createDirectories(MODEL_DIR);
            Path source = MODEL_DIR.resolve("network-io-source.bin");
            Path destination = MODEL_DIR.resolve("network-io-copy.bin");
            Files.deleteIfExists(source);
            Files.deleteIfExists(destination);

            Network network = new Network(3, 4, 2);
            seedDeterministicWeights(network);
            NetworkIO.save(network, source.toString());

            NetworkIO.copy(source.toString(), destination.toString());
            long sizeSource = Files.size(source);
            long sizeDestination = Files.size(destination);

            if (sizeSource != sizeDestination) {
                throw new IllegalStateException("Copied file size mismatch");
            }

            NetworkIO.ModelMetadata metadata = NetworkIO.getMetadata(destination.toString());
            System.out.println(" - Copied model bytes: " + sizeDestination);
            System.out.println(" - Metadata: " + metadata);
        } catch (Exception e) {
            throw new RuntimeException("Test 4 failed", e);
        }
    }

    private static void seedDeterministicWeights(Network network) {
        Layer[] layers = new Layer[] {
            network.hiddenLayer1,
            network.hiddenLayer2,
            network.outputLayer
        };

        float value = 0.05f;
        for (Layer layer : layers) {
            if (layer == null) {
                continue;
            }
            for (Neuron neuron : layer.neurons) {
                neuron.bias = value;
                value += 0.05f;
                for (int i = 0; i < neuron.weights.length; i++) {
                    neuron.weights[i] = value;
                    value += 0.05f;
                }
            }
        }
    }

    private static boolean approxEquals(float[] a, float[] b) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > EPSILON) {
                return false;
            }
        }
        return true;
    }

}
