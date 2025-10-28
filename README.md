# Neural Network (Java)

Lightweight multi-layer perceptron implementation written in Java, focused on
teaching-grade experiments and small reinforcement-learning prototypes.

## Requirements
- JDK 8 or newer (project targets bytecode level 8 by default)
- Bash-compatible shell for the helper scripts

## Project Layout
- `src/mg/rivolink/ai` – core network, layers, and neuron primitives
- `src/mg/rivolink/io` – model persistence utilities (binary, JSON, XML wrappers)
- `src/mg/rivolink/test` – runnable smoke tests that cover XOR/AND/OR training scenarios
- `scripts/` – build tooling (`build.sh`)
- `dist/` – packaged jars after a build
- `models/` – persisted models produced by IO tests or manual experiments

## Build & Artifacts
```bash
make build
```
Invokes `scripts/build.sh --clean`, compiles the library, and emits
`dist/neural-network.jar` plus a matching sources jar.  
Use `./scripts/build.sh --target=11` if you need a different Java release.

## Running Tests & Demos
Compile then execute any of the `main`-based tests:
```bash
javac -cp src -d bin $(find src -name '*.java')
java -cp bin mg.rivolink.test.Network1HiddenTest
java -cp bin mg.rivolink.test.Network2HiddenTest
java -cp bin mg.rivolink.test.NetworkIOTest
```
Each test prints convergence diagnostics or IO verification results to stdout.

## Saving & Loading Models
Use `NetworkIO.save(network, path)` for binary models, or pass an explicit
format:
```java
NetworkIO.save(network, "models/xor.json", NetworkIO.Format.JSON);
Network restored = NetworkIO.load("models/xor.json");
```
Binary (`.bin`), JSON (`.json`), and XML (`.xml`) files are interchangeable and
round-trip-compatible via the built-in serialization wrappers.
