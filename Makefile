GHCR_REGISTRY = ghcr.io/marcogarofalo94
PROJECT_NAME  = torchslicer
PLATFORM      ?=
PLATFORMS     ?= linux/amd64,linux/arm64

# Local build tags (used by dev compose files)
LOCAL_CPU = torchslicer:cpu
LOCAL_GPU = torchslicer:gpu

# Registry image names (ghcr.io — attached to the GitHub repo)
GHCR_CPU = $(GHCR_REGISTRY)/$(PROJECT_NAME):cpu
GHCR_GPU = $(GHCR_REGISTRY)/$(PROJECT_NAME):gpu

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
build-cpu: ## Build CPU image and tag for ghcr.io
	docker build $(if $(PLATFORM),--platform $(PLATFORM),) -f docker-images/Dockerfile.cpu -t $(LOCAL_CPU) .
	docker tag $(LOCAL_CPU) $(GHCR_CPU)

.PHONY: build-gpu
build-gpu: ## Build GPU image and tag for ghcr.io (requires NVIDIA Container Toolkit)
	docker build $(if $(PLATFORM),--platform $(PLATFORM),) -f docker-images/Dockerfile.gpu -t $(LOCAL_GPU) .
	docker tag $(LOCAL_GPU) $(GHCR_GPU)

.PHONY: build
build: build-cpu build-gpu ## Build both CPU and GPU images

.PHONY: buildx-cpu
buildx-cpu: ## Build and load multi-arch CPU image (linux/amd64,linux/arm64) locally
	docker buildx build --platform $(PLATFORMS) -f docker-images/Dockerfile.cpu -t $(LOCAL_CPU) --load .

# ── push (GitHub Container Registry) ──────────────────────────────────────────

.PHONY: ghcr-login
ghcr-login: ## Authenticate with ghcr.io (requires GITHUB_TOKEN env var or interactive prompt)
	@echo "$(GITHUB_TOKEN)" | docker login ghcr.io -u marcogarofalo94 --password-stdin 2>/dev/null || docker login ghcr.io

.PHONY: push-cpu
push-cpu: ## Push CPU image to ghcr.io
	docker push $(GHCR_CPU)

.PHONY: push-gpu
push-gpu: ## Push GPU image to ghcr.io
	docker push $(GHCR_GPU)

.PHONY: push
push: push-cpu push-gpu ## Push both images to ghcr.io

.PHONY: push-cpu-multiarch
push-cpu-multiarch: ## Build and push multi-arch CPU image manifest (linux/amd64,linux/arm64)
	docker buildx build --platform $(PLATFORMS) -f docker-images/Dockerfile.cpu -t $(GHCR_CPU) --push .

# ── run (centralized topology — all services on one host) ─────────────────────
#
# DEVICE=cpu (default) — CPU image, no GPU reservations
# DEVICE=gpu           — GPU image + NVIDIA device reservations
# CONFIG=path          — YAML experiment config (optional)
#
# Examples:
#   make run-centralized CONFIG=experiments/resnet18_4gpu.yaml
#   make run-centralized DEVICE=gpu CONFIG=experiments/resnet18_4gpu.yaml

_GPU_OVERRIDE_CENTRALIZED = $(if $(filter gpu,$(DEVICE)),-f docker-compose.gpu.yml)
_GPU_OVERRIDE_WORKER      = $(if $(filter gpu,$(DEVICE)),-f docker-compose.worker.gpu.yml)
_IMAGE_TAG                = $(if $(filter gpu,$(DEVICE)),gpu,cpu)

# COORDINATOR_CMD overrides the coordinator entry point, e.g.:
#   make run-centralized COORDINATOR_CMD="python3 examples/train/hf/coordinator/main.py"
# WORKER_CMD overrides the worker entry point.
.PHONY: run-centralized
run-centralized: _env ## Run centralized stack (coordinator + workers). DEVICE=cpu|gpu  CONFIG=path
	IMAGE_TAG=$(_IMAGE_TAG) EXPERIMENT_CONFIG=$(CONFIG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		COORDINATOR_CMD='$(COORDINATOR_CMD)' \
		WORKER_CMD='$(WORKER_CMD)' \
		docker compose -f docker-compose.yml $(_GPU_OVERRIDE_CENTRALIZED) up

.PHONY: run-centralized-auto
run-centralized-auto: _env ## Run centralized stack; tear down automatically on exit
	IMAGE_TAG=$(_IMAGE_TAG) EXPERIMENT_CONFIG=$(CONFIG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		KEEP_ALIVE=0 \
		COORDINATOR_CMD='$(COORDINATOR_CMD)' \
		WORKER_CMD='$(WORKER_CMD)' \
		bash -lc 'status=0; docker compose -f docker-compose.yml $(_GPU_OVERRIDE_CENTRALIZED) up || status=$$?; docker compose -f docker-compose.yml $(_GPU_OVERRIDE_CENTRALIZED) down --remove-orphans; exit $$status'

.PHONY: down-centralized
down-centralized: _env ## Stop and remove centralized stack
	docker compose -f docker-compose.yml down

# ── run (workers only — coordinator runs on host or remote machine) ─────────────
#
# Use this when the coordinator runs directly on the host (python3 .../coordinator/main.py)
# or on a different machine entirely.  Each worker machine runs this target independently.
#
# Required:
#   COORDINATOR_ADDRESS — host:port of the coordinator, e.g. 192.168.1.10:50054
#                         or host.docker.internal:50054 (Docker Desktop, coordinator on host)
# Optional:
#   DEVICE=cpu|gpu      — device selection (default: cpu)
#   CONFIG=path         — experiment YAML to mount into the worker containers
#
# Examples:
#   make run-workers COORDINATOR_ADDRESS=192.168.1.10:50054 DEVICE=gpu

.PHONY: run-workers
run-workers: ## Run workers only (coordinator on host/remote). COORDINATOR_ADDRESS=host:port  DEVICE=cpu|gpu
	@test -n "$(COORDINATOR_ADDRESS)" || \
		(echo "Error: COORDINATOR_ADDRESS is required.  Example: make run-workers COORDINATOR_ADDRESS=192.168.1.10:50054"; exit 1)
	COORDINATOR_ADDRESS=$(COORDINATOR_ADDRESS) \
		IMAGE_TAG=$(_IMAGE_TAG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		EXPERIMENT_CONFIG=$(CONFIG) \
		docker compose -f docker-compose.worker.yml $(_GPU_OVERRIDE_WORKER) up

.PHONY: down-workers
down-workers: ## Stop worker-only stack
	docker compose -f docker-compose.worker.yml down

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
run-p2p: _env ## Run P2P stack. DEVICE=cpu|gpu  CONFIG=path
	IMAGE_TAG=$(_IMAGE_TAG) EXPERIMENT_CONFIG=$(CONFIG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		docker compose -f docker-compose.p2p.yml $(_GPU_OVERRIDE_P2P) up --abort-on-container-exit

.PHONY: run-p2p-auto
run-p2p-auto: _env ## Run P2P stack; tear down automatically on exit
	IMAGE_TAG=$(_IMAGE_TAG) EXPERIMENT_CONFIG=$(CONFIG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		bash -lc 'status=0; docker compose -f docker-compose.p2p.yml $(_GPU_OVERRIDE_P2P) up --abort-on-container-exit || status=$$?; docker compose -f docker-compose.p2p.yml $(_GPU_OVERRIDE_P2P) down --remove-orphans; exit $$status'

.PHONY: down-p2p
down-p2p: _env ## Stop and remove P2P stack
	docker compose -f docker-compose.p2p.yml down

# ── monitoring (Phoenix tracing) ───────────────────────────────────────────────
#
# Layered on top of the centralized stack.  Phoenix UI at http://localhost:6006.
#
# Examples:
#   make run-phoenix CONFIG=experiments/resnet18_4gpu.yaml
#   make run-phoenix DEVICE=gpu CONFIG=experiments/resnet18_4gpu.yaml

_PHOENIX_OVERLAY = -f examples/monitoring/docker-compose.phoenix.yml

.PHONY: run-phoenix
run-phoenix: _env ## Run centralized stack + Phoenix tracing UI (http://localhost:6006). CONFIG=path  DEVICE=cpu|gpu
	IMAGE_TAG=$(_IMAGE_TAG) EXPERIMENT_CONFIG=$(CONFIG) DEVICE=$(if $(filter gpu,$(DEVICE)),cuda,auto) \
		KEEP_ALIVE=0 \
		COORDINATOR_CMD='$(COORDINATOR_CMD)' \
		WORKER_CMD='$(WORKER_CMD)' \
		bash -lc 'status=0; docker compose -f docker-compose.yml $(_GPU_OVERRIDE_CENTRALIZED) $(_PHOENIX_OVERLAY) up --abort-on-container-exit || status=$$?; docker compose -f docker-compose.yml $(_GPU_OVERRIDE_CENTRALIZED) $(_PHOENIX_OVERLAY) down --remove-orphans; exit $$status'

.PHONY: down-phoenix
down-phoenix: _env ## Stop Phoenix stack
	docker compose -f docker-compose.yml $(_PHOENIX_OVERLAY) down

# ── utility ────────────────────────────────────────────────────────────────────

.PHONY: list
list: ## List project images
	@docker images | grep $(PROJECT_NAME) || echo "No $(PROJECT_NAME) images found"

.PHONY: clean
clean: ## Remove local project images
	-docker rmi $(LOCAL_CPU) $(LOCAL_GPU) $(GHCR_CPU) $(GHCR_GPU)

.PHONY: clean-containers
clean-containers: ## Stop and remove all running project containers
	-docker ps -a -q --filter "ancestor=$(LOCAL_CPU)" | xargs docker rm -f
	-docker ps -a -q --filter "ancestor=$(LOCAL_GPU)" | xargs docker rm -f
