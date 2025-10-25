# ResearchAgent Dockerfile for AWS Lambda
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements first for optimal caching
# If requirements change, only this layer and below are rebuilt
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install all Python dependencies in a single layer
# boto3 is included in Lambda runtime, only add if specific version needed
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt && \
    # Uncomment if specific boto3 version required:
    # pip install --no-cache-dir boto3==1.34.0 && \
    # Clean up pip cache to reduce layer size
    pip cache purge 2>/dev/null || true

# Copy shared modules in a single operation
COPY tools.py monitoring.py ${LAMBDA_TASK_ROOT}/

# Copy all ResearchAgent files in a single layer
# main.py is REQUIRED because lambda_handler.py imports from it
COPY ResearchAgent/main.py \
    ResearchAgent/config.py \
    ResearchAgent/tools.py \
    ResearchAgent/tool_clustering.py \
    ResearchAgent/lambda_handler.py \
    ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_handler.lambda_handler"]
