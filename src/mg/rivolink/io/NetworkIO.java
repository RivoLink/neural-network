package mg.rivolink.io;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Base64;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import mg.rivolink.ai.Network;

public class NetworkIO {

    public enum Format {
        BINARY,
        JSON,
        XML
    }

    private static final Pattern JSON_DATA_PATTERN = Pattern.compile("\"data\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern XML_DATA_PATTERN = Pattern.compile("<data>(.*?)</data>", Pattern.DOTALL);

    public static void save(Network network, String filepath) throws IOException {
        save(network, filepath, Format.BINARY);
    }

    public static void save(Network network, String filepath, Format format) throws IOException {
        switch (format) {
            case BINARY:
                saveBinary(network, filepath);
                break;
            case JSON:
                saveJson(network, filepath);
                break;
            case XML:
                saveXml(network, filepath);
                break;
            default:
                throw new IllegalArgumentException("Unknown format: " + format);
        }
    }

    public static Network load(String filepath) throws IOException, ClassNotFoundException {
        String lower = filepath.toLowerCase();
        if (lower.endsWith(".json")) {
            return load(filepath, Format.JSON);
        }
        if (lower.endsWith(".xml")) {
            return load(filepath, Format.XML);
        }
        return load(filepath, Format.BINARY);
    }

    public static Network load(String filepath, Format format) throws IOException, ClassNotFoundException {
        switch (format) {
            case BINARY:
                return loadBinary(filepath);
            case JSON:
                return loadJson(filepath);
            case XML:
                return loadXml(filepath);
            default:
                throw new IllegalArgumentException("Unknown format: " + format);
        }
    }

    private static void saveBinary(Network network, String filepath) throws IOException {
        Path path = Paths.get(filepath);
        ensureParentDirectory(path);

        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(Files.newOutputStream(path)))) {
            oos.writeObject(network);
        }
    }

    private static Network loadBinary(String filepath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(Paths.get(filepath))))) {
            return (Network) ois.readObject();
        }
    }

    private static void saveJson(Network network, String filepath) throws IOException {
        Path path = Paths.get(filepath);
        ensureParentDirectory(path);

        byte[] serialized = serializeNetwork(network);
        String base64 = Base64.getEncoder().encodeToString(serialized);
        int hiddenLayers = network.hiddenLayer2 != null ? 2 : 1;

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"format\": \"java-serialized\",\n");
        json.append("  \"version\": 1,\n");
        json.append("  \"alpha\": ").append(network.alpha).append(",\n");
        json.append("  \"tau\": ").append(network.tau).append(",\n");
        json.append("  \"maxGradient\": ").append(network.maxGradient).append(",\n");
        json.append("  \"hiddenLayers\": ").append(hiddenLayers).append(",\n");
        json.append("  \"data\": \"").append(base64).append("\"\n");
        json.append("}\n");

        Files.write(path, json.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static Network loadJson(String filepath) throws IOException, ClassNotFoundException {
        Path path = Paths.get(filepath);
        String content = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);

        Matcher matcher = JSON_DATA_PATTERN.matcher(content);
        if (!matcher.find()) {
            throw new IOException("Invalid JSON model: missing data field");
        }

        String base64 = matcher.group(1).trim();
        try {
            byte[] data = Base64.getDecoder().decode(base64);
            return deserializeNetwork(data);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid base64 payload in JSON model", e);
        }
    }

    private static void saveXml(Network network, String filepath) throws IOException {
        Path path = Paths.get(filepath);
        ensureParentDirectory(path);

        byte[] serialized = serializeNetwork(network);
        String base64 = Base64.getEncoder().encodeToString(serialized);
        int hiddenLayers = network.hiddenLayer2 != null ? 2 : 1;

        StringBuilder xml = new StringBuilder();
        xml.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.append("<network>\n");
        xml.append("  <format>java-serialized</format>\n");
        xml.append("  <version>1</version>\n");
        xml.append("  <alpha>").append(network.alpha).append("</alpha>\n");
        xml.append("  <tau>").append(network.tau).append("</tau>\n");
        xml.append("  <maxGradient>").append(network.maxGradient).append("</maxGradient>\n");
        xml.append("  <hiddenLayers>").append(hiddenLayers).append("</hiddenLayers>\n");
        xml.append("  <data>").append(base64).append("</data>\n");
        xml.append("</network>\n");

        Files.write(path, xml.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static Network loadXml(String filepath) throws IOException, ClassNotFoundException {
        Path path = Paths.get(filepath);
        String content = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);

        Matcher matcher = XML_DATA_PATTERN.matcher(content);
        if (!matcher.find()) {
            throw new IOException("Invalid XML model: missing <data> element");
        }

        String base64 = matcher.group(1).trim();
        try {
            byte[] data = Base64.getDecoder().decode(base64);
            return deserializeNetwork(data);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid base64 payload in XML model", e);
        }
    }

    private static byte[] serializeNetwork(Network network) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(baos))) {
            oos.writeObject(network);
        }
        return baos.toByteArray();
    }

    private static Network deserializeNetwork(byte[] data) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new ByteArrayInputStream(data)))) {
            return (Network) ois.readObject();
        }
    }

    private static void ensureParentDirectory(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    public static void copy(String source, String destination) throws IOException {
        Files.copy(Paths.get(source), Paths.get(destination), StandardCopyOption.REPLACE_EXISTING);
    }

    public static ModelMetadata getMetadata(String filepath) throws IOException {
        Path path = Paths.get(filepath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("Model not found: " + filepath);
        }

        return new ModelMetadata(
            filepath,
            Files.size(path),
            Files.getLastModifiedTime(path).toMillis()
        );
    }

    public static class ModelMetadata {

        public final String filepath;
        public final long sizeBytes;
        public final long lastModified;

        public ModelMetadata(String filepath, long sizeBytes, long lastModified) {
            this.filepath = filepath;
            this.sizeBytes = sizeBytes;
            this.lastModified = lastModified;
        }

        public String getSizeFormatted() {
            if (sizeBytes < 1024) return sizeBytes + "B";
            if (sizeBytes < 1024 * 1024) return (sizeBytes / 1024) + "KB";
            return (sizeBytes / (1024 * 1024)) + "MB";
        }

        @Override
        public String toString() {
            return String.format(
                "Model: %s, Size: %s, Modified: %tc", filepath, getSizeFormatted(), lastModified
            );
        }
    }

}
