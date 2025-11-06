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


def write_status(bucket_name: str, filename: str, status: str, **kwargs):
    """
    Write processing status to S3 for async tracking.

    Preserves certain fields from existing status file (like started_at) when updating.

    Args:
        bucket_name: S3 bucket name
        filename: Full filename (e.g., '20251031T003447_f312ea72.json')
        status: Status string ('queued', 'processing', 'completed', 'failed')
        **kwargs: Additional status data
        Optional 'agent_type': 'ResearchAgent' or 'TransportAgent' (default: 'ResearchAgent')
    """
    # Determine agent type and get correct prefix
    agent_type = kwargs.get('agent_type', 'ResearchAgent')
    status_prefix = AGENT_PREFIXES.get(agent_type, AGENT_PREFIXES['ResearchAgent'])['status']

    # Use full filename for status file (e.g., 20251031T003447_f312ea72.json)
    status_key = f"{status_prefix}{filename}"

    # Extract session_id from filename for backwards compatibility
    session_id = kwargs.get('session_id', filename.replace('.json', '').split('_')[-1])

    # Try to read existing status file to preserve certain fields
    existing_data = {}
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=status_key)
        existing_data = json.loads(response['Body'].read())
        logger.info(f"Found existing status file for {filename}, preserving started_at")
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchKey':
            logger.warning(f"Error reading existing status for {filename}: {e}")
    except Exception as e:
        logger.warning(f"Error reading existing status for {filename}: {e}")

    # Build new status data
    status_data = {
        'session_id': session_id,
        'filename': filename,
        'status': status,
        'timestamp': datetime.utcnow().isoformat(),
        'agent': kwargs.get('agent_type', 'ResearchAgent'),  # Allow override from caller
        **kwargs
    }

    # Preserve started_at from existing file if not provided in new data
    if 'started_at' not in status_data and 'started_at' in existing_data:
        status_data['started_at'] = existing_data['started_at']

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=status_key,
            Body=json.dumps(status_data),
            ContentType='application/json'
        )
        logger.info(f"Status updated: {status} for {filename} at {status_key}")
    except Exception as e:
        logger.error(f"Failed to write status for {filename}: {e}")


def get_status(bucket_name: str, session_id: str, agent_type: str):
    """
    Get processing status from S3 by searching for files with session_id in name.

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

    try:
        # List all files in status folder
        list_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=status_prefix,
            MaxKeys=100
        )

        if 'Contents' not in list_response:
            return None

        # Find file with session_id in the name
        for obj in list_response['Contents']:
            filename = os.path.basename(obj['Key'])
            if session_id in filename:
                # Found the status file, read and return it
                response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                status_data = json.loads(response['Body'].read())
                return status_data

        return None
    except Exception as e:
        logger.error(f"Failed to get status for session {session_id}: {e}")
        return None


def delete_status(bucket_name: str, session_id: str, agent_type: str):
    """
    Delete status file from S3.

    NOTE: This function is kept for backward compatibility but is no longer used.
    Status files are now kept permanently as audit records.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID (e.g., 'f312ea72')
        agent_type: Agent type ('ResearchAgent' or 'TransportAgent') - REQUIRED

    Returns:
        True (no-op for backward compatibility)
    """
    logger.info(f"delete_status called for {session_id} but status files are now kept permanently")
    return True


def check_processed_result(bucket_name: str, session_id: str, agent_type: str):
    """
    Check if processed result exists for the session.

    New logic:
    1. Check status file first (in /status folder)
    2. If status == "completed" with output_key, fetch result from /processed
    3. Return combined response with status metadata + result data

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID (e.g., 'f312ea72')
        agent_type: Agent type ('ResearchAgent' or 'TransportAgent') - REQUIRED

    Returns:
        dict with status + result data if completed, None otherwise
    """
    if not agent_type or agent_type not in AGENT_PREFIXES:
        logger.error(f"Invalid agent_type: {agent_type}. Must be 'ResearchAgent' or 'TransportAgent'")
        return None

    status_prefix = AGENT_PREFIXES[agent_type]['status']
    output_prefix = AGENT_PREFIXES[agent_type]['output']

    try:
        # Step 1: Check status file first
        list_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=status_prefix,
            MaxKeys=100
        )

        if 'Contents' not in list_response:
            return None

        # Find status file with session_id in the name
        status_file_key = None
        for obj in list_response['Contents']:
            filename = os.path.basename(obj['Key'])
            if session_id in filename:
                status_file_key = obj['Key']
                break

        if not status_file_key:
            return None

        # Step 2: Read status file
        response = s3_client.get_object(Bucket=bucket_name, Key=status_file_key)
        status_data = json.loads(response['Body'].read())

        # Step 3: Check if status is "completed"
        if status_data.get('status') != 'completed':
            return None

        # Step 4: Get output_key from status
        output_key = status_data.get('output_key')
        if not output_key:
            logger.warning(f"Status is completed but no output_key found for session {session_id}")
            return None

        # Step 5: Fetch result from /processed folder
        try:
            result_response = s3_client.get_object(Bucket=bucket_name, Key=output_key)
            result_data = json.loads(result_response['Body'].read())

            # Step 6: Return combined response
            return {
                'status': 'completed',
                'session_id': session_id,
                'output_key': output_key,
                'output_location': status_data.get('output_location', f"s3://{bucket_name}/{output_key}"),
                'completed_at': status_data.get('completed_at'),
                'result': result_data
            }
        except Exception as e:
            logger.error(f"Failed to fetch result from {output_key}: {e}")
            return None

    except Exception as e:
        logger.error(f"Failed to check processed result for {session_id}: {e}")
        return None

