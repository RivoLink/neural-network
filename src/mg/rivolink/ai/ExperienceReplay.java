package mg.rivolink.ai;

import java.util.*;

/**
 * Experience Replay Buffer for Deep Q-Learning
 * Stores transitions and allows sampling for training
 */
public class ExperienceReplay {
    
    public static class Experience {
        public final float[] state;
        public final int action;
        public final float reward;
        public final float[] nextState;
        public final boolean done;

        public Experience(float[] state, int action, float reward, float[] nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }

    private final int capacity;
    private final List<Experience> buffer;
    private final Random random;

    public ExperienceReplay(int capacity) {
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
        this.random = new Random();
    }

    /**
     * Add experience to the buffer
     * If buffer is full, remove oldest experience
     */
    public void add(float[] state, int action, float reward, float[] nextState, boolean done) {
        if (buffer.size() >= capacity) {
            buffer.remove(0);
        }
        buffer.add(new Experience(state, action, reward, nextState, done));
    }

    /**
     * Sample a batch of experiences randomly
     */
    public List<Experience> sample(int batchSize) {
        if (batchSize > buffer.size()) {
            batchSize = buffer.size();
        }

        List<Experience> batch = new ArrayList<>(batchSize);
        Set<Integer> selectedIndices = new HashSet<>();

        while (selectedIndices.size() < batchSize) {
            int idx = random.nextInt(buffer.size());
            if (selectedIndices.add(idx)) {
                batch.add(buffer.get(idx));
            }
        }

        return batch;
    }

    /**
     * Get buffer size
     */
    public int size() {
        return buffer.size();
    }

    /**
     * Check if buffer has enough experiences for sampling
     */
    public boolean canSample(int batchSize) {
        return buffer.size() >= batchSize;
    }

    /**
     * Clear the buffer
     */
    public void clear() {
        buffer.clear();
    }

    /**
     * Get the most recent experience
     */
    public Experience getLastExperience() {
        if (buffer.isEmpty()) {
            return null;
        }
        return buffer.get(buffer.size() - 1);
    }
}
