.PHONY: help build deploy test clean

ENVIRONMENT ?= dev
AWS_REGION ?= ap-southeast-1
DOCKER_REGISTRY ?= 
VERSION ?= $(shell git rev-parse --short HEAD)

help:
	@echo "Available commands:"
	@echo "  make local-start    - Start local environment with docker-compose"
	@echo "  make local-test     - Test local Lambda endpoints"
	@echo "  make sam-build      - Build SAM application"
	@echo "  make sam-deploy     - Deploy to AWS"
	@echo "  make sam-local      - Start SAM local API"
	@echo "  make clean          - Clean all artifacts"

# Docker Compose Commands
local-start:
	docker-compose up -d

local-start-with-gateway:
	docker-compose --profile api up -d

local-test:
	@./scripts/test-local.sh

local-logs:
	docker-compose logs -f

local-stop:
	docker-compose down

# SAM Commands
sam-validate:
	sam validate --lint

sam-build: sam-validate
	sam build \
		--use-container \
		--parallel \
		--cached \
		--parameter-overrides "Environment=$(ENVIRONMENT)"

sam-deploy: sam-build
	sam deploy \
		--stack-name retrieval-agent-$(ENVIRONMENT) \
		--region $(AWS_REGION) \
		--parameter-overrides "Environment=$(ENVIRONMENT)" \
		--no-confirm-changeset \
		--no-fail-on-empty-changeset

sam-local:
	sam local start-api \
		--port 3000 \
		--env-vars env.json \
		--parameter-overrides "Environment=local"

sam-logs:
	sam logs \
		--stack-name retrieval-agent-$(ENVIRONMENT) \
		--tail

# Docker Build & Push (for ECR)
docker-build:
	docker-compose build

docker-push: docker-build
	@$(eval ECR_REPO := $(shell aws ecr describe-repositories --repository-names retrieval-agent --query 'repositories[0].repositoryUri' --output text))
	@aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REPO)
	@docker tag research-agent:latest $(ECR_REPO):research-$(VERSION)
	@docker tag transport-agent:latest $(ECR_REPO):transport-$(VERSION)
	@docker push $(ECR_REPO):research-$(VERSION)
	@docker push $(ECR_REPO):transport-$(VERSION)

# Testing
test-integration:
	python -m pytest tests/integration -v

test-unit:
	python -m pytest tests/unit -v

# Cleanup
clean:
	docker-compose down -v
	rm -rf .aws-sam/
	docker system prune -f