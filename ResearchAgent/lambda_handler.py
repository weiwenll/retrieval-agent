"""
Lambda handler for ResearchAgent with S3 input/output

Expected API Gateway payload:
{
    "bucket_name": "iss-travel-planner",
    "key": "retrieval_agent/20251101120000.json",
    "sender_agent": "Intent Agent",
    "session": "A52321B"
}

Parameters:
- bucket_name: S3 bucket name where input file is located
- key: Full S3 key path to input file (e.g., "planner_agent/20251101120000.json")
- sender_agent: Name of the agent that previously processed this request (for logging/tracking)
- session: Session ID to track related requests across agents and name output files

Input location: s3://{bucket_name}/{key}
Output location: s3://{bucket_name}/planner_agent/{filename}.json
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

# Output prefix for research agent output
OUTPUT_PREFIX = 'planner_agent/'


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

        # Structured logging: Request received
        log_structured('INFO', 'Request received',
            session_id=session_id,
            stage='initialization',
            bucket_name=bucket_name,
            input_key=input_key,
            sender_agent=sender_agent,
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

        error_code = e.response.get('Error', {}).get('Code', 'Unknown')

        if error_code == 'NoSuchKey':
            log_structured('ERROR', 'S3 file not found',
                session_id=session_id,
                stage='s3_error',
                error_code=error_code,
                s3_uri=f"s3://{bucket}/{key}",
                success=False)
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
        except:
            session_id = 'unknown'

        log_structured('ERROR', 'Unexpected error occurred',
            session_id=session_id,
            stage='error',
            error_type=type(e).__name__,
            error_message=str(e),
            success=False)
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'session_id': session_id,
                'error': f"Internal server error: {str(e)}"
            })
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        'body': {
            'bucket_name': 'iss-travel-planner',
            'key': 'retrieval_agent/20251101120000.json',
            'sender_agent': 'Retrieval Agent',
            'session': 'A52321B'
        }
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
