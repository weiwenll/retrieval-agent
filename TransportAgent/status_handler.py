"""
GET /transport endpoint handler to check transport task status.
"""

import json
import os
import logging
from shared_utils import get_status

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    Check status of transport task.

    Expected query parameters:
    - session: Session ID (e.g., 'f312ea72')
    - filename: Optional full filename (e.g., '20251031T003447_f312ea72.json')

    If filename not provided, searches for any file containing session ID.

    Returns:
        Status information or 404 if not found
    """
    try:
        # Parse query string parameters (GET request)
        query_params = event.get('queryStringParameters') or {}

        session_id = query_params.get('session', '').strip()
        filename = query_params.get('filename', '').strip()
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

        # If filename provided, construct input_key for direct lookup
        input_key = None
        if filename:
            input_key = f"transport_agent/active/{filename}"

        # Get status from S3
        status_data = get_status(
            bucket_name=bucket_name,
            session_id=session_id,
            input_key=input_key,
            agent_type='TransportAgent'
        )

        if not status_data:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'session_id': session_id,
                    'status': 'not_found',
                    'message': 'No status found for this session in TransportAgent'
                })
            }

        # If completed, include output location
        if status_data.get('status') == 'completed':
            output_key = status_data.get('output_key')
            if output_key:
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
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f"Failed to retrieve status: {str(e)}"
            })
        }
