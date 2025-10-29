build:
	bash scripts/build.sh --clean
.PHONY: build

build-java21:
	bash scripts/build.sh --clean --target=21
.PHONY: build-java21
