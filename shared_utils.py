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
        **kwargs: Additional status data
        Optional 'agent_type': 'ResearchAgent' or 'TransportAgent' (default: 'ResearchAgent')
    """
    # Determine agent type and get correct prefix
    agent_type = kwargs.get('agent_type', 'ResearchAgent')
    status_prefix = AGENT_PREFIXES.get(agent_type, AGENT_PREFIXES['ResearchAgent'])['status']

    # Always use session_id as filename (e.g., f312ea72.json)
    status_key = f"{status_prefix}{session_id}.json"
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


def get_status(bucket_name: str, session_id: str, agent_type: str):
    """
    Get processing status from S3.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID (e.g., 'f312ea72')
        agent_type: Agent type ('ResearchAgent' or 'TransportAgent') - REQUIRED

    Returns:
        Status dict or None if not found
    """
    if not agent_type or agent_type not in AGENT_PREFIXES:
        logger.error(f"Invalid agent_type: {agent_type}. Must be 'ResearchAgent' or 'TransportAgent'")
        return None

    status_prefix = AGENT_PREFIXES[agent_type]['status']
    status_key = f"{status_prefix}{session_id}.json"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=status_key)
        status_data = json.loads(response['Body'].read())
        return status_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        logger.error(f"Failed to get status for {session_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to get status for session {session_id}: {e}")
        return None


def delete_status(bucket_name: str, session_id: str, agent_type: str):
    """
    Delete status file from S3 after job completion.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID (e.g., 'f312ea72')
        agent_type: Agent type ('ResearchAgent' or 'TransportAgent') - REQUIRED

    Returns:
        True if deleted successfully, False otherwise
    """
    if not agent_type or agent_type not in AGENT_PREFIXES:
        logger.error(f"Invalid agent_type: {agent_type}. Must be 'ResearchAgent' or 'TransportAgent'")
        return False

    status_prefix = AGENT_PREFIXES[agent_type]['status']
    status_key = f"{status_prefix}{session_id}.json"

    try:
        s3_client.delete_object(Bucket=bucket_name, Key=status_key)
        logger.info(f"Deleted status file: {status_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete status file {status_key}: {e}")
        return False


def check_processed_result(bucket_name: str, session_id: str, agent_type: str):
    """
    Check if processed result exists for the session.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID (e.g., 'f312ea72')
        agent_type: Agent type ('ResearchAgent' or 'TransportAgent') - REQUIRED

    Returns:
        dict with result data if found, None otherwise
    """
    if not agent_type or agent_type not in AGENT_PREFIXES:
        logger.error(f"Invalid agent_type: {agent_type}. Must be 'ResearchAgent' or 'TransportAgent'")
        return None

    output_prefix = AGENT_PREFIXES[agent_type]['output']

    # List all files in the processed folder with session_id in name
    try:
        list_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=output_prefix,
            MaxKeys=100
        )

        if 'Contents' not in list_response:
            return None

        # Find file with session_id in the name
        for obj in list_response['Contents']:
            filename = os.path.basename(obj['Key'])
            if session_id in filename:
                # Found the processed file, read and return it
                response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                result_data = json.loads(response['Body'].read())
                return {
                    'status': 'completed',
                    'result': result_data,
                    'output_key': obj['Key'],
                    'output_location': f"s3://{bucket_name}/{obj['Key']}",
                    'completed_at': obj['LastModified'].isoformat()
                }

        return None
    except Exception as e:
        logger.error(f"Failed to check processed result for {session_id}: {e}")
        return None

