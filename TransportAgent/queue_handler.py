"""
API Gateway handler that queues transport tasks to SQS.
Returns 202 Accepted immediately to avoid API Gateway timeout.
"""

import json
import os
import logging
import boto3
import uuid
from datetime import datetime
from shared_utils import write_status, log_structured

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients outside handler for reuse across warm invocations
sqs_client = boto3.client('sqs')
s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """
    Queue transport task to SQS and return immediately.

    Expected payload:
    {
        "bucket_name": "iss-travel-planner",
        "key": "transport_agent/active/20251031T003447_f312ea72.json",
        "sender_agent": "Planner Agent",
        "session": "f312ea72"
    }

    Returns:
        202 Accepted with task_id and status check info
    """
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        # Extract and validate parameters
        bucket_name = body.get('bucket_name', '').strip()
        input_key = body.get('key', '').strip()
        sender_agent = body.get('sender_agent', 'API Gateway').strip()
        session_id = body.get('session', '').strip()

        # Validate required parameters
        if not bucket_name:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Missing required parameter: bucket_name'})
            }

        if not input_key:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Missing required parameter: key'})
            }

        if not session_id:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Missing required parameter: session'})
            }

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        log_structured('INFO', 'Queueing transport task',
            session_id=session_id,
            stage='queue',
            task_id=task_id,
            bucket_name=bucket_name,
            key=input_key,
            sender_agent=sender_agent,
            session=session_id)

        # Prepare SQS message
        task_message = {
            'task_id': task_id,
            'bucket_name': bucket_name,
            'key': input_key,
            'sender_agent': sender_agent,
            'session': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'api_request_id': context.aws_request_id
        }

        # Get queue URL from environment
        queue_url = os.environ.get('QUEUE_URL')
        if not queue_url:
            raise ValueError("QUEUE_URL environment variable not set")

        # Send message to SQS (standard queue, no deduplication ID)
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(task_message),
            MessageAttributes={
                'task_type': {
                    'StringValue': 'transport_task',
                    'DataType': 'String'
                },
                'session': {
                    'StringValue': session_id,
                    'DataType': 'String'
                }
            }
        )

        log_structured('INFO', 'Task queued to SQS',
            session_id=session_id,
            stage='queue',
            task_id=task_id,
            sqs_message_id=response['MessageId'])

        # Write initial status to S3 (non-blocking, don't fail if this errors)
        try:
            # Extract filename from input_key for status tracking
            filename = os.path.basename(input_key)

            write_status(
                bucket_name, filename, 'queued',
                session_id=session_id,
                task_id=task_id,
                input_key=input_key,
                sender_agent=sender_agent,
                created_at=datetime.utcnow().isoformat(),
                sqs_message_id=response['MessageId'],
                agent_type='TransportAgent'
            )
        except Exception as status_error:
            # Log error but don't fail the request
            logger.warning(f"Failed to write status to S3 (non-critical): {status_error}")

        # Return 202 Accepted immediately
        # Extract filename for status location (same logic as write_status)
        status_filename = os.path.basename(input_key)

        return {
            'statusCode': 202,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'accepted',
                'message': 'Transport task queued successfully',
                'task_id': task_id,
                'session_id': session_id,
                'sqs_message_id': response['MessageId'],
                'status_location': f"s3://{bucket_name}/transport_agent/status/{status_filename}",
                'estimated_completion': 'within 15 minutes'
            })
        }

    except Exception as e:
        logger.error(f"Error queueing task: {e}", exc_info=True)

        # Try to extract session for logging
        try:
            session_id = body.get('session', 'unknown')
        except:
            session_id = 'unknown'

        log_structured('ERROR', 'Failed to queue task',
            session_id=session_id,
            stage='queue_error',
            error=str(e))

        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Failed to queue transport task',
                'details': str(e)
            })
        }
