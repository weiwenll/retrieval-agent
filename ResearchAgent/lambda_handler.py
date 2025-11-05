"""
Lambda handler for ResearchAgent with S3 input/output and async support

Expected API Gateway payload:
{
    "bucket_name": "iss-travel-planner",
    "key": "retrieval_agent/20251101120000.json",
    "sender_agent": "Intent Agent",
    "session": "A52321B",
    "async": true  // Optional: enable async processing (default: false)
}

Parameters:
- bucket_name: S3 bucket name where input file is located
- key: Full S3 key path to input file (e.g., "retrieval_agent/20251101120000.json")
- sender_agent: Name of the agent that previously processed this request (for logging/tracking)
- session: Session ID to track related requests across agents and name output files
- async: Boolean flag to enable async processing (returns 202 immediately)

Input location: s3://{bucket_name}/{key}
Output location: s3://{bucket_name}/retrieval_agent/processed/{filename}.json
Status location (async): s3://{bucket_name}/retrieval_agent/status/{session_id}.json
"""

import json
import os
import logging
import tempfile
import boto3
import time
from datetime import datetime
from botocore.exceptions import ClientError

# Import the research_places function from main.py
from main import research_places

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

# Output prefix for research agent output (goes into processed folder)
OUTPUT_PREFIX = 'retrieval_agent/processed/'
STATUS_PREFIX = 'retrieval_agent/status/'


def write_status(bucket_name: str, session_id: str, status: str, **kwargs):
    """
    Write processing status to S3 for async tracking.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID
        status: Status string ('processing', 'completed', 'failed')
        **kwargs: Additional status data
    """
    status_key = f"{STATUS_PREFIX}{session_id}.json"
    status_data = {
        'session_id': session_id,
        'status': status,
        'timestamp': datetime.utcnow().isoformat(),
        'agent': 'ResearchAgent',
        **kwargs
    }

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=status_key,
            Body=json.dumps(status_data),
            ContentType='application/json'
        )
        logger.info(f"Status updated: {status} for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to write status for {session_id}: {e}")


def get_status(bucket_name: str, session_id: str):
    """
    Get processing status from S3.

    Args:
        bucket_name: S3 bucket name
        session_id: Session ID

    Returns:
        Status dict or None if not found
    """
    status_key = f"{STATUS_PREFIX}{session_id}.json"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=status_key)
        status_data = json.loads(response['Body'].read())
        return status_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise


def lambda_handler(event, context):
    """
    AWS Lambda handler for ResearchAgent

    Args:
        event: API Gateway event containing:
            - bucket_name: S3 bucket name
            - key: S3 key path to input file
            - sender_agent: Previous agent name (for logging)
            - session: Session ID for tracking
        context: Lambda context

    Returns:
        API Gateway response with status and result
    """
    try:
        # Start timing for metrics
        request_start_time = time.time()

        print(f"Raw event received: {json.dumps(event, default=str)}")

        # Parse request body if it's a string (API Gateway)
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        # Extract parameters (strip whitespace)
        bucket_name = body.get('bucket_name', '').strip()
        input_key = body.get('key', '').strip()
        sender_agent = body.get('sender_agent', '').strip()
        session_id = body.get('session', '').strip()
        async_mode = body.get('async', True)  # Default to async to prevent timeouts

        # Log when defaulting to async mode
        if 'async' not in body:
            log_structured('INFO', 'Defaulting to async mode to prevent timeout',
                session_id=session_id,
                stage='initialization')

        # Structured logging: Request received
        log_structured('INFO', 'Request received',
            session_id=session_id,
            stage='initialization',
            bucket_name=bucket_name,
            input_key=input_key,
            sender_agent=sender_agent,
            async_mode=async_mode,
            workflow_chain=f"{sender_agent} -> ResearchAgent")

        # Validate required parameters
        if not bucket_name:
            log_structured('ERROR', 'Missing required parameter: bucket_name',
            session_id=session_id, stage='validation')
            return {
                'statusCode': 400,
                'body': json.dumps({
                'error': 'Missing required parameter: bucket_name'
            })
        }

        if not input_key:
            log_structured('ERROR', 'Missing required parameter: key',
            session_id=session_id, stage='validation')
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: key'
                })
            }

        if not session_id:
            log_structured('ERROR', 'Missing required parameter: session',
            session_id=session_id, stage='validation')
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: session'
                })
            }

        # Extract filename from input_key and construct output S3 path
        filename = os.path.basename(input_key)
        output_s3_key = f"{OUTPUT_PREFIX}{filename}"

        # Structured logging: Configuration
        log_structured('INFO', 'S3 paths configured',
            session_id=session_id,
            stage='configuration',
            input_s3=f"s3://{bucket_name}/{input_key}",
            output_s3=f"s3://{bucket_name}/{output_s3_key}",
            filename=filename)

        # Write initial status for async mode
        if async_mode:
            write_status(
                bucket_name, session_id, 'processing',
                input_key=input_key,
                output_key=output_s3_key,
                sender_agent=sender_agent,
                started_at=datetime.utcnow().isoformat()
            )
            log_structured('INFO', 'Async mode: Status set to processing',
            session_id=session_id, stage='async_status')

        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as input_tmp:
            input_tmp_path = input_tmp.name

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as output_tmp:
            output_tmp_path = output_tmp.name

        try:
            # Download input file from S3
            download_start = time.time()
            log_structured('INFO', 'Starting S3 download',
                session_id=session_id,
                stage='s3_download',
                s3_uri=f"s3://{bucket_name}/{input_key}")

            s3_client.download_file(bucket_name, input_key, input_tmp_path)
            download_duration = time.time() - download_start

            log_structured('INFO', 'S3 download completed',
                session_id=session_id,
                stage='s3_download',
                duration_ms=round(download_duration * 1000, 2))

            # Run the research agent with session_id
            processing_start = time.time()
            log_structured('INFO', 'Starting research processing',
                session_id=session_id,
                stage='research_processing',
                previous_agent=sender_agent or 'N/A')

            result = research_places(input_tmp_path, output_tmp_path, session_id=session_id)
            processing_duration = time.time() - processing_start

            # Check if research was successful
            if 'error' in result:
                log_structured('ERROR', 'Research processing failed',
                    session_id=session_id,
                    stage='research_processing',
                    error=result['error'],
                    duration_ms=round(processing_duration * 1000, 2))

                # Update status for async mode
                if async_mode:
                    write_status(
                        bucket_name, session_id, 'failed',
                        error=result['error'],
                        failed_at=datetime.utcnow().isoformat()
                    )

                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': f"ResearchAgent failed: {result['error']}"
                    })
                }

            log_structured('INFO', 'Research processing completed',
                session_id=session_id,
                stage='research_processing',
                duration_ms=round(processing_duration * 1000, 2),
                places_found=result['retrieval']['places_found'],
                attractions_count=result['retrieval']['attractions_count'],
                food_count=result['retrieval']['food_count'])

            # Upload output file to S3
            upload_start = time.time()
            log_structured('INFO', 'Starting S3 upload',
                session_id=session_id,
                stage='s3_upload',
                s3_uri=f"s3://{bucket_name}/{output_s3_key}")

            s3_client.upload_file(output_tmp_path, bucket_name, output_s3_key)
            upload_duration = time.time() - upload_start

            log_structured('INFO', 'S3 upload completed',
                session_id=session_id,
                stage='s3_upload',
                duration_ms=round(upload_duration * 1000, 2))

            # Calculate total request duration
            total_duration = time.time() - request_start_time

            # Prepare response
            response_body = {
                'status': 'success',
                'message': 'Research completed successfully',
                'session_id': session_id,
                'sender_agent': sender_agent,
                'bucket_name': bucket_name,
                'key': output_s3_key,
                'input_location': f"s3://{bucket_name}/{input_key}",
                'output_location': f"s3://{bucket_name}/{output_s3_key}",
                'summary': {
                    'retrieval_id': result['retrieval']['retrieval_id'],
                    'places_found': result['retrieval']['places_found'],
                    'attractions_count': result['retrieval']['attractions_count'],
                    'food_count': result['retrieval']['food_count'],
                    'processing_time_seconds': result['retrieval']['time_elapsed']
                },
                'metrics': {
                    'total_duration_ms': round(total_duration * 1000, 2),
                    'download_duration_ms': round(download_duration * 1000, 2),
                    'processing_duration_ms': round(processing_duration * 1000, 2),
                    'upload_duration_ms': round(upload_duration * 1000, 2)
                }
            }

            # Update status for async mode
            if async_mode:
                write_status(
                    bucket_name, session_id, 'completed',
                    output_key=output_s3_key,
                    output_location=f"s3://{bucket_name}/{output_s3_key}",
                    places_found=result['retrieval']['places_found'],
                    attractions_count=result['retrieval']['attractions_count'],
                    food_count=result['retrieval']['food_count'],
                    completed_at=datetime.utcnow().isoformat(),
                    duration_ms=round(total_duration * 1000, 2)
                )
                
                return {
                    'statusCode': 202,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'status': 'accepted',
                        'session_id': session_id,
                        'message': 'Request accepted for async processing',
                        'status_location': f"s3://{bucket_name}/{STATUS_PREFIX}{session_id}.json",
                        'check_status_endpoint': '/status',
                        'expected_duration_seconds': '60-90'
                    })
                }

            # Structured logging: Request completed successfully
            log_structured('INFO', 'Request completed successfully',
                session_id=session_id,
                stage='completion',
                total_duration_ms=round(total_duration * 1000, 2),
                places_found=result['retrieval']['places_found'],
                success=True,
                next_agent='PlannerAgent',
                workflow_chain=f"{sender_agent} -> ResearchAgent -> PlannerAgent")

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(response_body)
            }

        finally:
            # Clean up temporary files
            try:
                os.unlink(input_tmp_path)
                os.unlink(output_tmp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")

    except ClientError as e:
        # Get session_id for logging
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)
        session_id = body.get('session', 'unknown')
        bucket = body.get('bucket_name', 'unknown')
        key = body.get('key', 'unknown')
        async_mode = body.get('async', False)

        error_code = e.response.get('Error', {}).get('Code', 'Unknown')

        if error_code == 'NoSuchKey':
            log_structured('ERROR', 'S3 file not found',
                session_id=session_id,
                stage='s3_error',
                error_code=error_code,
                s3_uri=f"s3://{bucket}/{key}",
                success=False)

            # Update status for async mode
            if async_mode and bucket != 'unknown':
                write_status(bucket, session_id, 'failed',
                error=f"Input file not found: s3://{bucket}/{key}",
                error_code=error_code,
                failed_at=datetime.utcnow().isoformat())

            return {
                'statusCode': 404,
                'body': json.dumps({
                    'status': 'error',
                    'session_id': session_id,
                    'error': f"Input file not found: s3://{bucket}/{key}"
                })
            }

        log_structured('ERROR', 'S3 error occurred',
            session_id=session_id,
            stage='s3_error',
            error_code=error_code,
            error_message=str(e),
            success=False)

        # Update status for async mode
        if async_mode and bucket != 'unknown':
            write_status(bucket, session_id, 'failed',
            error=f"S3 error: {str(e)}",
            error_code=error_code,
            failed_at=datetime.utcnow().isoformat())

        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'session_id': session_id,
                'error': f"S3 error: {str(e)}"
            })
        }

    except Exception as e:
        # Try to get session_id for logging
        try:
            body = event.get('body', event)
            if isinstance(body, str):
                body = json.loads(body)
            session_id = body.get('session', 'unknown')
            bucket = body.get('bucket_name', 'unknown')
            async_mode = body.get('async', False)
        except:
            session_id = 'unknown'
            bucket = 'unknown'
            async_mode = False

        log_structured('ERROR', 'Unexpected error occurred',
            session_id=session_id,
            stage='error',
            error_type=type(e).__name__,
            error_message=str(e),
            success=False)
        logger.error(f"Unexpected error: {e}", exc_info=True)

        # Update status for async mode
        if async_mode and bucket != 'unknown':
            write_status(bucket, session_id, 'failed',
            error=f"Internal server error: {str(e)}",
            error_type=type(e).__name__,
            failed_at=datetime.utcnow().isoformat())

        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'session_id': session_id,
                'error': f"Internal server error: {str(e)}"
            })
        }


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


# For local testing
# if __name__ == "__main__":
#     test_event = {
#         'body': {
#             'bucket_name': 'iss-travel-planner',
#             'key': 'retrieval_agent/20251101120000.json',
#             'sender_agent': 'Retrieval Agent',
#             'session': 'A52321B'
#         }
#     }

#     result = lambda_handler(test_event, None)
#     print(json.dumps(result, indent=2))
