# TransportAgent Dockerfile for AWS Lambda
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements first for optimal layer caching
# If requirements change, only this layer and below are rebuilt
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install all Python dependencies in a single RUN to minimize layers
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt && \
    pip cache purge 2>/dev/null || true

# Copy shared modules from root
COPY tools.py monitoring.py ${LAMBDA_TASK_ROOT}/

# Copy all TransportAgent files in a single layer
# main.py is REQUIRED because lambda_handler.py imports from it
COPY TransportAgent/main.py \
    TransportAgent/config.py \
    TransportAgent/carbon_calculator.py \
    TransportAgent/lambda_handler.py \
    ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_handler.lambda_handler"]
