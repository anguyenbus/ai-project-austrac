# Lambda Container Build and Deployment Makefile
#
# Usage:
#   make help              Show available targets
#   make build             Build Lambda container image
#   make push              Push image to ECR
#   make deploy            Deploy Lambda function
#   make all               Build, push, and deploy
#   make test-local        Test container locally
#   make clean             Remove local images

.PHONY: help build push deploy all test-local load-env clean ecr-login ecr-create publish config check-deps
.PHONY: cfn-validate cfn-create cfn-update cfn-delete cfn-status cfn-outputs cfn-events infra-deploy

# Configuration variables
IMAGE_NAME ?= instantinsighttest
IMAGE_TAG ?= latest
AWS_REGION ?= ap-southeast-2
FUNCTION_NAME ?= instantinsight-query-processor
DOCKERFILE ?= lambda/Dockerfile
ENVIRONMENT ?= production

# Derived variables
AWS_ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME := $(IMAGE_NAME)
ECR_URI := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO_NAME)
FULL_IMAGE_URI := $(ECR_URI):$(IMAGE_TAG)

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m

# Help target
help:
	@echo "$(COLOR_BOLD)Lambda Container Deployment Makefile$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)Available targets:$(COLOR_RESET)"
	@echo "  $(COLOR_BOLD)build$(COLOR_RESET)          Build Lambda container image"
	@echo "  $(COLOR_BOLD)push$(COLOR_RESET)           Push image to ECR"
	@echo "  $(COLOR_BOLD)deploy$(COLOR_RESET)         Deploy Lambda function"
	@echo "  $(COLOR_BOLD)all$(COLOR_RESET)            Build, push, and deploy (full workflow)"
	@echo "  $(COLOR_BOLD)test-local$(COLOR_RESET)     Test container locally"
	@echo "  $(COLOR_BOLD)clean$(COLOR_RESET)          Remove local images"
	@echo ""
	@echo "$(COLOR_GREEN)CloudFormation targets:$(COLOR_RESET)"
	@echo "  $(COLOR_BOLD)cfn-validate$(COLOR_RESET)   Validate CloudFormation template"
	@echo "  $(COLOR_BOLD)cfn-create$(COLOR_RESET)     Create CloudFormation stack"
	@echo "  $(COLOR_BOLD)cfn-update$(COLOR_RESET)     Update CloudFormation stack"
	@echo "  $(COLOR_BOLD)cfn-delete$(COLOR_RESET)     Delete CloudFormation stack"
	@echo "  $(COLOR_BOLD)cfn-status$(COLOR_RESET)     Show stack status"
	@echo "  $(COLOR_BOLD)cfn-outputs$(COLOR_RESET)    Show stack outputs"
	@echo "  $(COLOR_BOLD)cfn-events$(COLOR_RESET)     Show stack events"
	@echo "  $(COLOR_BOLD)infra-deploy$(COLOR_RESET)   Create or update infrastructure"
	@echo ""
	@echo "$(COLOR_GREEN)Environment variables:$(COLOR_RESET)"
	@echo "  IMAGE_NAME=$(IMAGE_NAME)"
	@echo "  IMAGE_TAG=$(IMAGE_TAG)"
	@echo "  AWS_REGION=$(AWS_REGION)"
	@echo "  AWS_ACCOUNT_ID=$(AWS_ACCOUNT_ID)"
	@echo "  ECR_URI=$(ECR_URI)"
	@echo ""
	@echo "$(COLOR_YELLOW)Examples:$(COLOR_RESET)"
	@echo "  make build IMAGE_TAG=v1.0.0"
	@echo "  make push IMAGE_TAG=v1.0.0"
	@echo "  make all IMAGE_TAG=v1.0.0"

# Build Lambda container image
build:
	@echo "$(COLOR_BOLD)🔨 Building Lambda container image...$(COLOR_RESET)"
	@echo "Image: $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "Platform: linux/amd64"
	@cd .. && docker build \
		-t "$(IMAGE_NAME):$(IMAGE_TAG)" \
		-f "$(DOCKERFILE)" \
		--platform linux/amd64 \
		--build-arg IMAGE_TAG=$(IMAGE_TAG) \
		.
	@echo "$(COLOR_GREEN)✅ Container image built successfully$(COLOR_RESET)"

# ECR authentication
ecr-login:
	@echo "$(COLOR_BOLD)🔐 Authenticating to ECR...$(COLOR_RESET)"
	@aws ecr get-login-password --region "$(AWS_REGION)" | \
		docker login --username AWS --password-stdin "$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com"
	@echo "$(COLOR_GREEN)✅ ECR authentication successful$(COLOR_RESET)"

# Create ECR repository if it doesn't exist
ecr-create:
	@echo "$(COLOR_BOLD)📝 Checking ECR repository...$(COLOR_RESET)"
	@if ! aws ecr describe-repositories --repository-names "$(ECR_REPO_NAME)" --region "$(AWS_REGION)" >/dev/null 2>&1; then \
		echo "Creating ECR repository: $(ECR_REPO_NAME)"; \
		aws ecr create-repository \
			--repository-name "$(ECR_REPO_NAME)" \
			--region "$(AWS_REGION)" \
			--image-scanning-configuration scanOnPush=true \
			--encryption-configuration encryptionType=AES256 \
			--tags Key=Project,Value=instantinsight Key=Component,Value=QueryProcessor; \
		echo "$(COLOR_GREEN)✅ ECR repository created$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_GREEN)✅ ECR repository already exists$(COLOR_RESET)"; \
	fi

# Push image to ECR
push: ecr-login ecr-create
	@echo "$(COLOR_BOLD)📦 Pushing container image to ECR...$(COLOR_RESET)"
	@echo "Repository: $(ECR_URI)"
	@echo "Tag: $(IMAGE_TAG)"
	@docker tag "$(IMAGE_NAME):$(IMAGE_TAG)" "$(FULL_IMAGE_URI)"
	@docker push "$(FULL_IMAGE_URI)"
	@echo "$(COLOR_GREEN)✅ Container image pushed successfully$(COLOR_RESET)"
	@echo "Image URI: $(FULL_IMAGE_URI)"

# Deploy Lambda function
deploy:
	@echo "$(COLOR_BOLD)🚀 Deploying Lambda function...$(COLOR_RESET)"
	@echo "Function: $(FUNCTION_NAME)"
	@echo "Image: $(FULL_IMAGE_URI)"
	@if aws lambda get-function --function-name "$(FUNCTION_NAME)" --region "$(AWS_REGION)" >/dev/null 2>&1; then \
		echo "♻️  Updating existing Lambda function..."; \
		aws lambda update-function-code \
			--function-name "$(FUNCTION_NAME)" \
			--image-uri "$(FULL_IMAGE_URI)" \
			--region "$(AWS_REGION)" \
			--output text; \
		echo "⏳ Waiting for function update..."; \
		aws lambda wait function-updated \
			--function-name "$(FUNCTION_NAME)" \
			--region "$(AWS_REGION)"; \
		echo "$(COLOR_GREEN)✅ Lambda function updated successfully$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)⚠️  Function does not exist. Please create it using CloudFormation first.$(COLOR_RESET)"; \
		echo "Run: make cfn-create"; \
		exit 1; \
	fi

# Publish Lambda version and update alias
publish:
	@echo "$(COLOR_BOLD)📝 Publishing new Lambda version...$(COLOR_RESET)"
	@VERSION=$$(aws lambda publish-version \
		--function-name "$(FUNCTION_NAME)" \
		--region "$(AWS_REGION)" \
		--query 'Version' \
		--output text); \
	echo "$(COLOR_GREEN)✅ Deployed version: $$VERSION$(COLOR_RESET)"; \
	if aws lambda get-alias \
		--function-name "$(FUNCTION_NAME)" \
		--name "production" \
		--region "$(AWS_REGION)" >/dev/null 2>&1; then \
		echo "🔄 Updating 'production' alias to version $$VERSION..."; \
		aws lambda update-alias \
			--function-name "$(FUNCTION_NAME)" \
			--name "production" \
			--function-version "$$VERSION" \
			--region "$(AWS_REGION)" \
			--output text; \
		echo "$(COLOR_GREEN)✅ Production alias updated$(COLOR_RESET)"; \
	fi

# Full deployment workflow
all: build push deploy publish
	@echo "$(COLOR_GREEN)$(COLOR_BOLD)✅ Full deployment completed successfully!$(COLOR_RESET)"

# Load environment variables from .env file
load-env:
	@bash lambda/scripts/load-env.sh

# Test container locally (with environment loading)
test-local: build
	@echo "$(COLOR_BOLD)🧪 Testing container locally...$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)Starting Lambda runtime interface emulator on port 9000$(COLOR_RESET)"
	@echo "Note: Using host.docker.internal to connect to local services"
	@echo "Use Ctrl+C to stop"
	@echo ""
	@cd .. && bash lambda/scripts/run-local-test.sh "$(IMAGE_NAME):$(IMAGE_TAG)"

# Test local invocation
test-invoke:
	@echo "$(COLOR_BOLD)🧪 Invoking local Lambda function...$(COLOR_RESET)"
	@curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
		-H "Content-Type: application/json" \
		-d '{"query": "How many products are discontinued for each category?"}' | jq .

# Clean up local images
clean:
	@echo "$(COLOR_BOLD)🧹 Cleaning up local images...$(COLOR_RESET)"
	@docker rmi "$(IMAGE_NAME):$(IMAGE_TAG)" 2>/dev/null || true
	@docker rmi "$(FULL_IMAGE_URI)" 2>/dev/null || true
	@echo "$(COLOR_GREEN)✅ Cleanup complete$(COLOR_RESET)"

# CloudFormation targets
cfn-validate:
	@echo "$(COLOR_BOLD)✓ Validating CloudFormation template...$(COLOR_RESET)"
	@aws cloudformation validate-template \
		--template-body file://lambda/infrastructure/cloudformation/template.yaml \
		--region $(AWS_REGION)
	@echo "$(COLOR_GREEN)✅ Template is valid$(COLOR_RESET)"

cfn-create:
	@echo "$(COLOR_BOLD)🚀 Creating CloudFormation stack...$(COLOR_RESET)"
	@aws cloudformation create-stack \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--template-body file://lambda/infrastructure/cloudformation/template.yaml \
		--parameters file://lambda/infrastructure/cloudformation/parameters.json \
		--capabilities CAPABILITY_NAMED_IAM \
		--region $(AWS_REGION) \
		--tags \
			Key=Project,Value=instantinsight \
			Key=Environment,Value=$(ENVIRONMENT) \
			Key=Component,Value=QueryProcessor
	@echo "$(COLOR_YELLOW)⏳ Stack creation in progress...$(COLOR_RESET)"
	@aws cloudformation wait stack-create-complete \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--region $(AWS_REGION)
	@echo "$(COLOR_GREEN)✅ Stack created successfully$(COLOR_RESET)"

cfn-update:
	@echo "$(COLOR_BOLD)♻️  Updating CloudFormation stack...$(COLOR_RESET)"
	@aws cloudformation update-stack \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--template-body file://lambda/infrastructure/cloudformation/template.yaml \
		--parameters file://lambda/infrastructure/cloudformation/parameters.json \
		--capabilities CAPABILITY_NAMED_IAM \
		--region $(AWS_REGION) || echo "$(COLOR_YELLOW)No updates to perform$(COLOR_RESET)"
	@if aws cloudformation describe-stacks \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].StackStatus' \
		--output text | grep -q "UPDATE_IN_PROGRESS"; then \
		echo "$(COLOR_YELLOW)⏳ Stack update in progress...$(COLOR_RESET)"; \
		aws cloudformation wait stack-update-complete \
			--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
			--region $(AWS_REGION); \
		echo "$(COLOR_GREEN)✅ Stack updated successfully$(COLOR_RESET)"; \
	fi

cfn-delete:
	@echo "$(COLOR_BOLD)💥 Deleting CloudFormation stack...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)⚠️  WARNING: This will destroy all Lambda infrastructure!$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		aws cloudformation delete-stack \
			--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
			--region $(AWS_REGION); \
		echo "$(COLOR_YELLOW)⏳ Stack deletion in progress...$(COLOR_RESET)"; \
		aws cloudformation wait stack-delete-complete \
			--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
			--region $(AWS_REGION); \
		echo "$(COLOR_GREEN)✅ Stack deleted successfully$(COLOR_RESET)"; \
	fi

cfn-status:
	@echo "$(COLOR_BOLD)📊 CloudFormation stack status:$(COLOR_RESET)"
	@aws cloudformation describe-stacks \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].[StackName,StackStatus,CreationTime]' \
		--output table

cfn-outputs:
	@echo "$(COLOR_BOLD)📋 CloudFormation stack outputs:$(COLOR_RESET)"
	@aws cloudformation describe-stacks \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].Outputs' \
		--output table

cfn-events:
	@echo "$(COLOR_BOLD)📜 CloudFormation stack events:$(COLOR_RESET)"
	@aws cloudformation describe-stack-events \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--region $(AWS_REGION) \
		--max-items 20 \
		--query 'StackEvents[].[Timestamp,ResourceStatus,ResourceType,LogicalResourceId]' \
		--output table

# Infrastructure deployment (create or update)
infra-deploy: cfn-validate
	@if aws cloudformation describe-stacks \
		--stack-name instantinsight-query-processor-$(ENVIRONMENT) \
		--region $(AWS_REGION) >/dev/null 2>&1; then \
		$(MAKE) cfn-update; \
	else \
		$(MAKE) cfn-create; \
	fi

# Show current configuration
config:
	@echo "$(COLOR_BOLD)Current Configuration:$(COLOR_RESET)"
	@echo "  IMAGE_NAME:      $(IMAGE_NAME)"
	@echo "  IMAGE_TAG:       $(IMAGE_TAG)"
	@echo "  AWS_REGION:      $(AWS_REGION)"
	@echo "  AWS_ACCOUNT_ID:  $(AWS_ACCOUNT_ID)"
	@echo "  FUNCTION_NAME:   $(FUNCTION_NAME)"
	@echo "  ECR_URI:         $(ECR_URI)"
	@echo "  FULL_IMAGE_URI:  $(FULL_IMAGE_URI)"

# Validate Docker and AWS CLI are available
check-deps:
	@echo "$(COLOR_BOLD)Checking dependencies...$(COLOR_RESET)"
	@command -v docker >/dev/null 2>&1 || { echo "$(COLOR_YELLOW)❌ docker not found$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_GREEN)✅ docker found$(COLOR_RESET)"
	@command -v aws >/dev/null 2>&1 || { echo "$(COLOR_YELLOW)❌ aws CLI not found$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_GREEN)✅ aws CLI found$(COLOR_RESET)"
	@command -v jq >/dev/null 2>&1 || { echo "$(COLOR_YELLOW)⚠️  jq not found (optional)$(COLOR_RESET)"; }