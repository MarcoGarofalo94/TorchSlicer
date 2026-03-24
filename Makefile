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

.PHONY: run-arch-ext-gpu
run-arch-ext-gpu: _env ## Run architecture extension smoke test (GPU, 2 workers: Mistral + DeepSeek MoE)
	docker compose \
		-f docker-compose.arch-ext.yml \
		-f docker-compose.arch-ext.gpu.yml \
		up --abort-on-container-exit

.PHONY: run-arch-ext-cpu
run-arch-ext-cpu: _env ## Run architecture extension smoke test (CPU, 2 workers: Mistral + DeepSeek MoE)
	docker compose \
		-f docker-compose.arch-ext.yml \
		up --abort-on-container-exit

.PHONY: down-arch-ext
down-arch-ext: ## Stop and remove arch extension test stack
	docker compose -f docker-compose.arch-ext.yml down

.PHONY: run-hf-dist-gpu
run-hf-dist-gpu: _env ## Run HuggingFace GPT-2 distributed training (GPU, 4 workers). CONFIG=path/to/experiment.yaml
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.hf-dist.yml \
		-f docker-compose.hf-dist.gpu.yml \
		up --abort-on-container-exit

.PHONY: run-hf-dist-cpu
run-hf-dist-cpu: _env ## Run HuggingFace GPT-2 distributed training (CPU, 4 workers). CONFIG=path/to/experiment.yaml
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.hf-dist.yml \
		up --abort-on-container-exit

.PHONY: down-hf-dist
down-hf-dist: ## Stop and remove HF distributed stack
	docker compose -f docker-compose.hf-dist.yml down

.PHONY: run-hf-gpu
run-hf-gpu: _env ## Run HuggingFace GPT-2 fine-tuning (GPU, LocalExecutor)
	docker compose \
		-f docker-compose.hf.yml \
		-f docker-compose.hf.gpu.yml \
		up --abort-on-container-exit

.PHONY: run-hf-cpu
run-hf-cpu: _env ## Run HuggingFace GPT-2 fine-tuning (CPU, LocalExecutor)
	docker compose \
		-f docker-compose.hf.yml \
		up --abort-on-container-exit

.PHONY: down-hf
down-hf: ## Stop and remove HF fine-tuning container
	docker compose -f docker-compose.hf.yml down

.PHONY: run-lm-gpu
run-lm-gpu: _env ## Run TinyGPT LM experiment (GPU, P2P). CONFIG=path/to/experiment.yaml
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.lm.yml \
		-f docker-compose.lm.gpu.yml \
		up --abort-on-container-exit

.PHONY: run-lm-cpu
run-lm-cpu: _env ## Run TinyGPT LM experiment (CPU, P2P). CONFIG=path/to/experiment.yaml
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.lm.yml \
		up --abort-on-container-exit

.PHONY: run-lora-gpu
run-lora-gpu: _env ## Run TinyGPT+LoRA experiment (GPU, P2P). CONFIG=path/to/experiment.yaml
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.lora.yml \
		-f docker-compose.lora.gpu.yml \
		up --abort-on-container-exit

.PHONY: run-lora-cpu
run-lora-cpu: _env ## Run TinyGPT+LoRA experiment (CPU, P2P). CONFIG=path/to/experiment.yaml
	EXPERIMENT_CONFIG=$(CONFIG) docker compose \
		-f docker-compose.lora.yml \
		up --abort-on-container-exit

.PHONY: down-lora
down-lora: _env ## Stop and remove TinyGPT+LoRA stack containers
	docker compose -f docker-compose.lora.yml down

.PHONY: down-lm
down-lm: _env ## Stop and remove TinyGPT LM stack containers
	docker compose -f docker-compose.lm.yml down

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
