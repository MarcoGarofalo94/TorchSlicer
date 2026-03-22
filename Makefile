DOCKER_REGISTRY ?= marcog94
PROJECT_NAME    = torchslicer

CPU_TAG = $(DOCKER_REGISTRY)/$(PROJECT_NAME):cpu
GPU_TAG = $(DOCKER_REGISTRY)/$(PROJECT_NAME):gpu

.PHONY: help
help: ## Show available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── build ──────────────────────────────────────────────────────────────────────
.PHONY: build-cpu
build-cpu: ## Build CPU image
	docker build -f docker-images/Dockerfile.cpu -t $(CPU_TAG) .

.PHONY: build-gpu
build-gpu: ## Build GPU image (requires NVIDIA Container Toolkit)
	docker build -f docker-images/Dockerfile.gpu -t $(GPU_TAG) .

.PHONY: build
build: build-cpu build-gpu ## Build all images

# ── push ───────────────────────────────────────────────────────────────────────
.PHONY: push-cpu
push-cpu: ## Push CPU image
	docker push $(CPU_TAG)

.PHONY: push-gpu
push-gpu: ## Push GPU image
	docker push $(GPU_TAG)

.PHONY: push
push: push-cpu push-gpu ## Push all images

# ── utility ────────────────────────────────────────────────────────────────────
.PHONY: list
list: ## List project images
	@docker images | grep $(PROJECT_NAME) || echo "No $(PROJECT_NAME) images found"

.PHONY: clean
clean: ## Remove project images
	-docker rmi $(CPU_TAG) $(GPU_TAG)

.PHONY: clean-containers
clean-containers: ## Stop and remove running project containers
	-docker ps -a -q --filter "ancestor=$(CPU_TAG)" | xargs docker rm -f
	-docker ps -a -q --filter "ancestor=$(GPU_TAG)" | xargs docker rm -f
