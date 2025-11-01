.PHONY: help build deploy test clean

ENVIRONMENT ?= dev
AWS_REGION ?= ap-southeast-1
DOCKER_REGISTRY ?= 
VERSION ?= $(shell git rev-parse --short HEAD)

help:
	@echo "Available commands:"
	@echo "  make local-start    - Start local environment with docker-compose"
	@echo "  make local-test     - Test local Lambda endpoints"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-push    - Push Docker images to ECR with git commit tag"
	@echo "  make sam-build      - Build SAM application"
	@echo "  make sam-deploy     - Deploy SAM stack to AWS"
	@echo "  make deploy         - Complete workflow: build, push, and deploy (uses current git commit)"
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
	@echo "Deploying with git commit: $(VERSION)"
	sam deploy \
		--stack-name retrieval-agent-stack \
		--region $(AWS_REGION) \
		--parameter-overrides "Environment=$(ENVIRONMENT) ImageTag=$(VERSION)" \
		--no-confirm-changeset \
		--no-fail-on-empty-changeset \
		--force-upload

sam-local:
	sam local start-api \
		--port 3000 \
		--env-vars env.json \
		--parameter-overrides "Environment=local"

sam-logs:
	sam logs \
		--stack-name retrieval-agent-stack \
		--tail

# Docker Build & Push (for ECR)
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-push: docker-build
	@echo "Pushing images to ECR with tag: $(VERSION)"
	@$(eval ECR_REPO := $(shell aws ecr describe-repositories --repository-names retrieval-agent --query 'repositories[0].repositoryUri' --output text --region $(AWS_REGION)))
	@aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REPO)
	docker tag research-agent:latest $(ECR_REPO):research-$(VERSION)
	docker tag transport-agent:latest $(ECR_REPO):transport-$(VERSION)
	docker push $(ECR_REPO):research-$(VERSION)
	docker push $(ECR_REPO):transport-$(VERSION)
	@echo "Successfully pushed research-$(VERSION) and transport-$(VERSION)"

# Complete deployment workflow
deploy: docker-push sam-deploy
	@echo "Deployment complete! Deployed git commit: $(VERSION)"

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