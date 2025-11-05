"""
Shared utilities for both ResearchAgent and TransportAgent.

Contains:
- log_structured(): Structured CloudWatch logging with session context
- write_status(): Write processing status to S3 for async tracking
- get_status(): Retrieve processing status from S3
- status_handler(): Lambda handler for /status endpoint

Status File Locations:
- ResearchAgent: s3://{bucket}/retrieval_agent/status/{filename}.json
- TransportAgent: s3://{bucket}/transport_agent/status/{filename}.json

Note: Status filename uses the full input filename (e.g., 20251031T003447_f312ea72.json) not just the session ID (e.g., f312ea72)
"""

import json
import os
import logging
import tempfile
import boto3
import time
from datetime import datetime
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Helper function for structured logging
def log_structured(level, message, session_id=None, stage=None, **kwargs):
    """
    Log structured messages with session context for better filtering/analytics.

    CloudWatch Insights query example:
    fields @timestamp, message, session_id, stage, duration_ms
    | filter session_id = "A52321B"
    | sort @timestamp asc
    """
    log_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'session_id': session_id,
        'agent': 'ResearchAgent',
        'stage': stage,
        'message': message,
        **kwargs
    }

    # Create human-readable message with structured data
    log_message = f"[Session: {session_id}] [Stage: {stage}] {message}"
    if kwargs:
        log_message += f" | {json.dumps(kwargs)}"

    if level == 'INFO':
        logger.info(log_message)
    elif level == 'ERROR':
        logger.error(log_message)
    elif level == 'WARNING':
        logger.warning(log_message)

    return log_data

# Initialize S3 client
s3_client = boto3.client('s3')

# Output and status prefixes for both agents
# These are defaults, actual prefix is determined by agent_type parameter
AGENT_PREFIXES = {
    'ResearchAgent': {
        'output': 'retrieval_agent/processed/',
        'status': 'retrieval_agent/status/'
    },
    'TransportAgent': {
        'output': 'transport_agent/processed/',
        'status': 'transport_agent/status/'
    }
}


def write_status(bucket_name: str, session_id: str, status: str, **kwargs):
    """
    Write processing status to S3 for async tracking.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID (short ID like 'f312ea72')
        status: Status string ('queued', 'processing', 'completed', 'failed')
        **kwargs: Additional status data (must include 'input_key' for filename extraction)
        Optional 'agent_type': 'ResearchAgent' or 'TransportAgent' (default: 'ResearchAgent')
    """
    # Determine agent type and get correct prefix
    agent_type = kwargs.get('agent_type', 'ResearchAgent')
    status_prefix = AGENT_PREFIXES.get(agent_type, AGENT_PREFIXES['ResearchAgent'])['status']

    # Extract full filename from input_key if provided
    # Example: retrieval_agent/active/20251031T003447_f312ea72.json -> 20251031T003447_f312ea72.json
    input_key = kwargs.get('input_key', '')
    if input_key:
        filename = os.path.basename(input_key)  # Get just the filename
        status_filename = filename
    else:
        # Fallback to session_id if input_key not provided
        status_filename = f"{session_id}.json"

    status_key = f"{status_prefix}{status_filename}"
    status_data = {
        'session_id': session_id,
        'status': status,
        'timestamp': datetime.utcnow().isoformat(),
        'agent': kwargs.get('agent_type', 'ResearchAgent'),  # Allow override from caller
        **kwargs
    }

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=status_key,
            Body=json.dumps(status_data),
            ContentType='application/json'
        )
        logger.info(f"Status updated: {status} for session {session_id} at {status_key}")
    except Exception as e:
        logger.error(f"Failed to write status for {session_id}: {e}")


def get_status(bucket_name: str, session_id: str, input_key: str = None, agent_type: str = None):
    """
    Get processing status from S3.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID (e.g., 'f312ea72')
        input_key: Optional input key to extract full filename
        agent_type: Optional agent type ('ResearchAgent' or 'TransportAgent')

    Returns:
        Status dict or None if not found
    """
    # If input_key and agent_type provided, extract filename for direct lookup
    if input_key and agent_type:
        filename = os.path.basename(input_key)
        status_prefix = AGENT_PREFIXES.get(agent_type, AGENT_PREFIXES['ResearchAgent'])['status']
        status_key = f"{status_prefix}{filename}"

        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=status_key)
            status_data = json.loads(response['Body'].read())
            return status_data
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    # Fallback: search across both agent status prefixes
    try:
        for prefix_config in AGENT_PREFIXES.values():
            list_response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix_config['status'],
                MaxKeys=100
            )

            if 'Contents' in list_response:
                for obj in list_response['Contents']:
                    if session_id in obj['Key']:
                        response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                        status_data = json.loads(response['Body'].read())
                        return status_data
        return None
    except Exception as e:
        logger.error(f"Failed to find status file for session {session_id}: {e}")
        return None

def status_handler(event, context):
    """
    Lambda handler to check status of async processing.

    Expected payload:
    {
        "bucket_name": "iss-travel-planner",
        "session": "A52321B"
    }

    Returns:
        Status information or 404 if not found
    """
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        bucket_name = body.get('bucket_name', '').strip()
        session_id = body.get('session', '').strip()

        # Validate parameters
        if not bucket_name or not session_id:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameters: bucket_name and session'
                })
            }

        # Get status from S3
        status_data = get_status(bucket_name, session_id)

        if not status_data:
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'session_id': session_id,
                    'status': 'not_found',
                    'message': 'No status found for this session'
                })
            }

        # If completed, include output location
        if status_data.get('status') == 'completed':
            output_key = status_data.get('output_key')
            status_data['output_location'] = f"s3://{bucket_name}/{output_key}"

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(status_data)
        }

    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Failed to retrieve status: {str(e)}"
            })
        }
