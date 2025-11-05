"""
GET /research endpoint handler to check research task status.

Enhanced workflow:
1. Check processed files first (fast path for completed jobs)
2. If not found, check status files (for in-progress jobs)
3. Detect timeouts (jobs processing > 5 minutes)
4. Return appropriate responses
"""

import json
import logging
from datetime import datetime, timezone
from shared_utils import check_processed_result, get_status

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Timeout threshold in seconds (5 minutes)
TIMEOUT_THRESHOLD = 5 * 60


def lambda_handler(event, context):
    """
    Check status of research task with enhanced logic.

    Expected query parameters:
    - session: Session ID (e.g., 'f312ea72') - REQUIRED
    - bucket_name: S3 bucket name (default: 'iss-travel-planner')

    Workflow:
    1. Check if processed result exists (job completed)
    2. If not, check status file (job in progress)
    3. Detect timeout if job stuck (processing > 5 minutes)
    4. Return not_found if neither exists

    Returns:
        Status information with appropriate HTTP status code
    """
    try:
        # Parse query string parameters (GET request)
        query_params = event.get('queryStringParameters') or {}

        session_id = query_params.get('session', '').strip()
        bucket_name = query_params.get('bucket_name', 'iss-travel-planner').strip()

        # Validate required parameters
        if not session_id:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required parameter: session'
                })
            }

        # Step 1: Check processed files first (fast path)
        processed_result = check_processed_result(
            bucket_name=bucket_name,
            session_id=session_id,
            agent_type='ResearchAgent'
        )

        if processed_result:
            logger.info(f"Found completed result for session {session_id}")
            # Case 1: Completed - return full result with processed data
            response_data = {
                'status': 'completed',
                'session_id': session_id,
                'completed_at': processed_result.get('completed_at'),
                'output_location': processed_result.get('output_location'),
                'result': processed_result.get('result')
            }
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(response_data)
            }

        # Step 2: Check status file (job in progress)
        status_data = get_status(
            bucket_name=bucket_name,
            session_id=session_id,
            agent_type='ResearchAgent'
        )

        if status_data:
            # Calculate elapsed time
            timestamp_str = status_data.get('timestamp')
            elapsed_seconds = None

            if timestamp_str:
                try:
                    started_at = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    elapsed_seconds = int((now - started_at).total_seconds())

                    if elapsed_seconds > TIMEOUT_THRESHOLD:
                        logger.warning(f"Job timeout detected for session {session_id}: {elapsed_seconds}s elapsed")
                        return {
                            'statusCode': 408,  # Request Timeout
                            'headers': {
                                'Content-Type': 'application/json',
                                'Access-Control-Allow-Origin': '*'
                            },
                            'body': json.dumps({
                                'status': 'timeout',
                                'session_id': session_id,
                                'message': 'Job appears stuck (processing > 5 minutes)',
                                'started_at': timestamp_str,
                                'elapsed_seconds': elapsed_seconds,
                                'agent': status_data.get('agent', 'ResearchAgent')
                            })
                        }
                except Exception as e:
                    logger.error(f"Error parsing timestamp: {e}")

            # Case 2: Still Processing - return clean processing status
            logger.info(f"Job in progress for session {session_id}")
            response_data = {
                'status': status_data.get('status', 'processing'),
                'session_id': session_id,
                'started_at': status_data.get('timestamp') or status_data.get('started_at'),
                'agent': status_data.get('agent', 'ResearchAgent')
            }

            # Add elapsed_seconds if calculated
            if elapsed_seconds is not None:
                response_data['elapsed_seconds'] = elapsed_seconds

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(response_data)
            }

        # Case 3: Not Found - neither processed nor status file found
        logger.info(f"No job found for session {session_id}")
        return {
            'statusCode': 404,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'not_found',
                'session_id': session_id,
                'message': 'No job found with this session ID'
            })
        }

    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f"Failed to retrieve status: {str(e)}"
            })
        }
