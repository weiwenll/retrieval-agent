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

# Helper function to normalize filenames
def normalize_filename(input_key: str) -> str:
    """
    Normalize filename from S3 key to ensure correct format.

    Handles cases like:
    - Double extensions: '20251103T164210_a91be637.json.json' -> '20251103T164210_a91be637.json'
    - Missing extensions: '20251103T164210_a91be637' -> '20251103T164210_a91be637.json'

    Args:
        input_key: S3 key or filename (e.g., 'retrieval_agent/active/20251103T164210_a91be637.json.json')

    Returns:
        Normalized filename (e.g., '20251103T164210_a91be637.json')
    """
    import os
    import re

    # Extract basename from path
    filename = os.path.basename(input_key)

    # Remove duplicate .json extensions (e.g., .json.json -> .json)
    while filename.endswith('.json.json'):
        filename = filename[:-5]  # Remove one '.json'
        logger.info(f"[normalize_filename] Removed duplicate .json extension: {filename}")

    # Ensure filename ends with .json
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
        logger.info(f"[normalize_filename] Added missing .json extension: {filename}")

    # Validate format: should be like '20251103T164210_a91be637.json'
    pattern = r'^\d{8}T\d{6}_[a-f0-9]{8}\.json$'
    if not re.match(pattern, filename):
        logger.warning(f"[normalize_filename] Filename doesn't match expected pattern: {filename}")

    logger.info(f"[normalize_filename] Normalized '{os.path.basename(input_key)}' -> '{filename}'")
    return filename


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


def get_status(bucket_name: str, filename: str, agent_type: str):
    """
    Get processing status from S3 by direct filename lookup.

    Args:
        bucket_name: S3 bucket name
        filename: Full filename (e.g., '20251031T003447_f312ea72.json')
        agent_type: Agent type ('ResearchAgent' or 'TransportAgent') - REQUIRED

    Returns:
        Status dict or None if not found
    """
    if not agent_type or agent_type not in AGENT_PREFIXES:
        logger.error(f"Invalid agent_type: {agent_type}. Must be 'ResearchAgent' or 'TransportAgent'")
        return None

    status_prefix = AGENT_PREFIXES[agent_type]['status']
    status_key = f"{status_prefix}{filename}"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=status_key)
        status_data = json.loads(response['Body'].read())
        return status_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        logger.error(f"Failed to get status for {filename}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to get status for {filename}: {e}")
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


def check_processed_result(bucket_name: str, filename: str, agent_type: str):
    """
    Check if processed result exists for the filename.

    New logic:
    1. Check status file first (in /status folder) by direct filename lookup
    2. If status == "completed" with output_key, fetch result from /processed
    3. Return combined response with status metadata + result data

    Args:
        bucket_name: S3 bucket name
        filename: Full filename (e.g., '20251031T003447_f312ea72.json')
        agent_type: Agent type ('ResearchAgent' or 'TransportAgent') - REQUIRED

    Returns:
        dict with status + result data if completed, None otherwise
    """
    logger.info(f"[check_processed_result] Starting for filename: {filename}, agent_type: {agent_type}")

    if not agent_type or agent_type not in AGENT_PREFIXES:
        logger.error(f"[check_processed_result] Invalid agent_type: {agent_type}. Must be 'ResearchAgent' or 'TransportAgent'")
        return None

    status_prefix = AGENT_PREFIXES[agent_type]['status']
    status_key = f"{status_prefix}{filename}"

    # Extract session_id from filename for response
    session_id = filename.replace('.json', '').split('_')[-1]
    logger.info(f"[check_processed_result] Session ID: {session_id}, Status key: {status_key}")

    try:
        # Step 1: Read status file directly by filename
        logger.info(f"[check_processed_result] Step 1: Reading status file from S3: {status_key}")
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=status_key)
            status_content = response['Body'].read()
            logger.info(f"[check_processed_result] Status file found, size: {len(status_content)} bytes")

            status_data = json.loads(status_content)
            logger.info(f"[check_processed_result] Status data parsed, keys: {list(status_data.keys())}")
            logger.info(f"[check_processed_result] Status value: {status_data.get('status')}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.info(f"[check_processed_result] Status file not found: {status_key}")
                return None
            logger.error(f"[check_processed_result] S3 ClientError accessing status file: {error_code} - {str(e)}")
            raise

        # Step 2: Check if status is "completed"
        logger.info(f"[check_processed_result] Step 2: Checking if status is 'completed'")
        if status_data.get('status') != 'completed':
            logger.info(f"[check_processed_result] Status is '{status_data.get('status')}', not 'completed'. Returning None.")
            return None
        logger.info(f"[check_processed_result] Status is 'completed'")

        # Step 3: Get output_key from status
        logger.info(f"[check_processed_result] Step 3: Getting output_key from status data")
        output_key = status_data.get('output_key')
        if not output_key:
            logger.warning(f"[check_processed_result] Status is completed but no output_key found for {filename}")
            logger.warning(f"[check_processed_result] Status data keys: {list(status_data.keys())}")
            return None
        logger.info(f"[check_processed_result] Output key found: {output_key}")

        # Step 4: Fetch result from /processed folder
        logger.info(f"[check_processed_result] Step 4: Fetching result file from S3: {output_key}")
        try:
            result_response = s3_client.get_object(Bucket=bucket_name, Key=output_key)
            result_content = result_response['Body'].read()
            logger.info(f"[check_processed_result] Result file found, size: {len(result_content)} bytes")

            if not result_content.strip():
                logger.warning(f"[check_processed_result] Result file is empty!")
                return None

            result_data = json.loads(result_content)
            logger.info(f"[check_processed_result] Result data parsed successfully")
            logger.info(f"[check_processed_result] Result data type: {type(result_data).__name__}")
            if isinstance(result_data, dict):
                logger.info(f"[check_processed_result] Result data keys: {list(result_data.keys())}")

            # Extract agent-specific nested data
            if agent_type == 'ResearchAgent':
                if 'retrieval' in result_data:
                    logger.info(f"[check_processed_result] Extracting 'retrieval' key for ResearchAgent")
                    result_data = result_data['retrieval']
                else:
                    logger.warning(f"[check_processed_result] 'retrieval' key not found in result data for ResearchAgent. Available keys: {list(result_data.keys())}")
            elif agent_type == 'TransportAgent':
                if 'transport' in result_data:
                    logger.info(f"[check_processed_result] Extracting 'transport' key for TransportAgent")
                    result_data = result_data['transport']
                else:
                    logger.warning(f"[check_processed_result] 'transport' key not found in result data for TransportAgent. Available keys: {list(result_data.keys())}")

            # Step 5: Return combined response
            logger.info(f"[check_processed_result] Step 5: Building combined response")
            combined_response = {
                'status': 'completed',
                'session_id': session_id,
                'filename': filename,
                'output_key': output_key,
                'output_location': status_data.get('output_location', f"s3://{bucket_name}/{output_key}"),
                'started_at': status_data.get('started_at'),
                'completed_at': status_data.get('completed_at'),
                'result': result_data
            }
            logger.info(f"[check_processed_result] SUCCESS: Returning combined response with result data")
            return combined_response

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"[check_processed_result] S3 ClientError fetching result from {output_key}: {error_code} - {str(e)}")
            if error_code == 'NoSuchKey':
                logger.error(f"[check_processed_result] Result file does not exist at: {output_key}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"[check_processed_result] JSON decode error for result file: {str(e)}")
            logger.error(f"[check_processed_result] First 200 chars of content: {result_content[:200] if result_content else 'empty'}")
            return None
        except Exception as e:
            logger.error(f"[check_processed_result] Unexpected error fetching result from {output_key}: {type(e).__name__} - {str(e)}")
            import traceback
            logger.error(f"[check_processed_result] Traceback: {traceback.format_exc()}")
            return None

    except Exception as e:
        logger.error(f"[check_processed_result] Unexpected error in outer try block: {type(e).__name__} - {str(e)}")
        import traceback
        logger.error(f"[check_processed_result] Traceback: {traceback.format_exc()}")
        return None

