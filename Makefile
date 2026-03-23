DOCKER_REGISTRY ?= marcog94
PROJECT_NAME    = torchslicer

CPU_TAG = $(DOCKER_REGISTRY)/$(PROJECT_NAME):cpu
GPU_TAG = $(DOCKER_REGISTRY)/$(PROJECT_NAME):gpu

# Write UID/GID into .env so docker compose picks them up for `user:` fields.
# This ensures run artifacts in ./runs/ are owned by the host user, not root.
.PHONY: _env
_env:
	@printf 'UID=%s\nGID=%s\n' "$$(id -u)" "$$(id -g)" > .env

.PHONY: help
help: ## Show available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── build ──────────────────────────────────────────────────────────────────────
.PHONY: build-cpu
build-cpu: ## Build CPU image
	docker build -f docker-images/Dockerfile.cpu -t $(CPU_TAG) .

.PHONY: build-gpu
build-gpu: ## Build GPU image (requires NVIDIA Container Toolkit)
	docker build -f docker-images/Dockerfile.gpu -t $(GPU_TAG) .

.PHONY: build
build: build-cpu build-gpu ## Build all images

# ── run ────────────────────────────────────────────────────────────────────────
# Coordinator blocks after training until `docker compose down` (SIGTERM).
# Workers stay up and accept the next run without restart.
# Pass CONFIG= to use a YAML experiment config file:
#   make run-cpu CONFIG=experiments/resnet18_4gpu.yaml
.PHONY: run-cpu
run-cpu: _env ## Run full stack (CPU). CONFIG=path/to/experiment.yaml optional
	EXPERIMENT_CONFIG=$(CONFIG) docker compose up

.PHONY: run-gpu
run-gpu: _env ## Run full stack (GPU). CONFIG=path/to/experiment.yaml optional
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.yml \
		-f docker-compose.gpu.yml \
		up

.PHONY: run-monitor
run-monitor: _env ## Run CPU stack + Jaeger + dashboard. CONFIG= optional
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.yml \
		-f docker-compose.monitor.yml \
		up

.PHONY: run-gpu-monitor
run-gpu-monitor: _env ## Run GPU stack + Jaeger + dashboard. CONFIG= optional
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.yml \
		-f docker-compose.gpu.yml \
		-f docker-compose.monitor.yml \
		up

.PHONY: run-p2p-gpu
run-p2p-gpu: _env ## Run P2P GPU stack (2 workers, no coordinator). CONFIG=path/to/experiment.yaml optional
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.p2p.yml \
		-f docker-compose.p2p.gpu.yml \
		up --abort-on-container-exit

.PHONY: run-p2p-cpu
run-p2p-cpu: _env ## Run P2P CPU stack (2 workers, no coordinator). CONFIG=path/to/experiment.yaml optional
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.p2p.yml \
		up --abort-on-container-exit

.PHONY: down
down: _env ## Stop and remove centralized stack containers
	docker compose down

.PHONY: down-p2p
down-p2p: _env ## Stop and remove P2P stack containers
	docker compose -f docker-compose.p2p.yml down

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
