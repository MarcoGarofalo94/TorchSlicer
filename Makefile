DOCKER_REGISTRY ?= marcog94
PROJECT_NAME    = torchslicer

CPU_IMAGE = $(DOCKER_REGISTRY)/$(PROJECT_NAME):cpu
GPU_IMAGE = $(DOCKER_REGISTRY)/$(PROJECT_NAME):gpu
PLATFORM ?=
PLATFORMS ?= linux/amd64,linux/arm64

# Write UID/GID into .env so docker compose picks them up for `user:` fields.
# This ensures run artifacts in ./runs/ are owned by the host user, not root.
.PHONY: _env
_env:
	@printf 'UID=%s\nGID=%s\n' "$$(id -u)" "$$(id -g)" > .env

.PHONY: help
help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "} /^[[:alpha:]][-[:alnum:]_]*:.*## / {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── build ──────────────────────────────────────────────────────────────────────

.PHONY: build-cpu
build-cpu: ## Build CPU image
	docker build $(if $(PLATFORM),--platform $(PLATFORM),) -f docker-images/Dockerfile.cpu -t $(CPU_IMAGE) .

.PHONY: build-gpu
build-gpu: ## Build GPU image (requires NVIDIA Container Toolkit)
	docker build $(if $(PLATFORM),--platform $(PLATFORM),) -f docker-images/Dockerfile.gpu -t $(GPU_IMAGE) .

.PHONY: build
build: build-cpu build-gpu ## Build both CPU and GPU images

.PHONY: buildx-cpu
buildx-cpu: ## Build multi-arch CPU image (defaults: linux/amd64,linux/arm64) and load locally
	docker buildx build --platform $(PLATFORMS) -f docker-images/Dockerfile.cpu -t $(CPU_IMAGE) --load .

.PHONY: push-cpu-multiarch
push-cpu-multiarch: ## Build and push multi-arch CPU image manifest list
	docker buildx build --platform $(PLATFORMS) -f docker-images/Dockerfile.cpu -t $(CPU_IMAGE) --push .

# ── run (centralized topology) ─────────────────────────────────────────────────
#
# DEVICE=cpu (default) — CPU image, no GPU reservations
# DEVICE=gpu           — GPU image + NVIDIA device reservations
# CONFIG=path          — YAML experiment config (optional)
#
# Examples:
#   make run-centralized CONFIG=experiments/resnet18_4gpu.yaml
#   make run-centralized DEVICE=gpu CONFIG=experiments/resnet18_4gpu.yaml

_GPU_OVERRIDE_CENTRALIZED = $(if $(filter gpu,$(DEVICE)),-f docker-compose.gpu.yml)
_IMAGE_TAG                 = $(if $(filter gpu,$(DEVICE)),gpu,cpu)

# COORDINATOR_CMD overrides the coordinator script, e.g.:
#   make run-centralized DEVICE=gpu CONFIG=experiments/hf_gpt2_4gpu_baseline.yaml \
#        COORDINATOR_CMD="python3 examples/train/hf/coordinator/main.py"
# WORKER_CMD overrides the worker script, e.g.:
#   make run-centralized DEVICE=gpu CONFIG=experiments/resnet18_3worker_ft.yaml \
#        WORKER_CMD="python3 examples/train/centralized/GRPC/worker/main_ft_test.py"
.PHONY: run-centralized
run-centralized: _env ## Run centralized stack. DEVICE=cpu|gpu  CONFIG=path  [COORDINATOR_CMD=...]
	IMAGE_TAG=$(_IMAGE_TAG) EXPERIMENT_CONFIG=$(CONFIG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		COORDINATOR_CMD='$(COORDINATOR_CMD)' \
		WORKER_CMD='$(WORKER_CMD)' \
		docker compose -f docker-compose.yml $(_GPU_OVERRIDE_CENTRALIZED) up

.PHONY: down-centralized
down-centralized: _env ## Stop and remove centralized stack
	docker compose -f docker-compose.yml down

# ── run (P2P topology) ─────────────────────────────────────────────────────────
#
# DEVICE=cpu (default) — CPU image, no GPU reservations
# DEVICE=gpu           — GPU image + NVIDIA device reservations
# CONFIG=path          — YAML experiment config (optional)
#
# Examples:
#   make run-p2p CONFIG=experiments/resnet18_2gpu_p2p.yaml
#   make run-p2p DEVICE=gpu CONFIG=experiments/resnet18_2gpu_p2p.yaml

_GPU_OVERRIDE_P2P = $(if $(filter gpu,$(DEVICE)),-f docker-compose.p2p.gpu.yml)

.PHONY: run-p2p
run-p2p: _env ## Run P2P stack. DEVICE=cpu|gpu  CONFIG=path/to/experiment.yaml
	IMAGE_TAG=$(_IMAGE_TAG) EXPERIMENT_CONFIG=$(CONFIG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		docker compose -f docker-compose.p2p.yml $(_GPU_OVERRIDE_P2P) up --abort-on-container-exit

.PHONY: down-p2p
down-p2p: _env ## Stop and remove P2P stack
	docker compose -f docker-compose.p2p.yml down

# ── push ───────────────────────────────────────────────────────────────────────

.PHONY: push-cpu
push-cpu: ## Push CPU image
	docker push $(CPU_IMAGE)

.PHONY: push-gpu
push-gpu: ## Push GPU image
	docker push $(GPU_IMAGE)

.PHONY: push
push: push-cpu push-gpu ## Push both images

# ── utility ────────────────────────────────────────────────────────────────────

.PHONY: list
list: ## List project images
	@docker images | grep $(PROJECT_NAME) || echo "No $(PROJECT_NAME) images found"

.PHONY: clean
clean: ## Remove project images
	-docker rmi $(CPU_IMAGE) $(GPU_IMAGE)

.PHONY: clean-containers
clean-containers: ## Stop and remove all running project containers
	-docker ps -a -q --filter "ancestor=$(CPU_IMAGE)" | xargs docker rm -f
	-docker ps -a -q --filter "ancestor=$(GPU_IMAGE)" | xargs docker rm -f
