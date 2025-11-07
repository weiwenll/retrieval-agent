"""
SQS-triggered Lambda handler for processing research tasks asynchronously.
This handler is invoked by SQS and processes tasks in the background.
"""

import json
import os
import logging
import tempfile
import boto3
import time
from datetime import datetime
from botocore.exceptions import ClientError

# Import existing functions
from main import research_places
from shared_utils import write_status, log_structured, delete_status, normalize_filename, check_duplicate_session

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')
OUTPUT_PREFIX = 'retrieval_agent/processed/'


def lambda_handler(event, context):
    """
    Process research tasks from SQS queue.

    Args:
        event: SQS event with Records array
        context: Lambda context

    Returns:
        Processing result
    """
    logger.info(f"Received {len(event['Records'])} messages from SQS")

    # Process each SQS record
    for record in event['Records']:
        try:
            # Parse SQS message
            message_body = json.loads(record['body'])
            task_id = message_body['task_id']
            bucket_name = message_body['bucket_name']
            input_key = message_body['key']
            session_id = message_body['session']
            sender_agent = message_body.get('sender_agent', 'API')

            # Normalize filename from input_key
            filename = normalize_filename(input_key)

            # ============================================================
            # LOG: New session received
            # ============================================================
            logger.info(f"{'='*80}")
            logger.info(f"[OK] NEW SESSION RECEIVED")
            logger.info(f"     PROCESSING FILE: {filename}")
            logger.info(f"     SESSION ID: {session_id}")
            logger.info(f"     TASK ID: {task_id}")
            logger.info(f"     SQS MESSAGE ID: {record['messageId']}")
            logger.info(f"{'='*80}")

            log_structured('INFO', 'NEW SESSION RECEIVED',
                session_id=session_id,
                stage='session_start',
                task_id=task_id,
                filename=filename,
                sqs_message_id=record['messageId'])

            # ============================================================
            # CHECK: Duplicate session detection
            # ============================================================
            from shared_utils import check_duplicate_session

            duplicate_check = check_duplicate_session(
                bucket_name=bucket_name,
                filename=filename,
                agent_type='ResearchAgent'
            )

            if duplicate_check['is_duplicate']:
                existing_status = duplicate_check['existing_status']
                current_status = existing_status.get('status', 'unknown')

                # ============================================================
                # LOG: Duplicate detected - REJECT
                # ============================================================
                logger.warning(f"{'='*80}")
                logger.warning(f"[WARNING] DUPLICATE SESSION DETECTED")
                logger.warning(f"          FILE: {filename}")
                logger.warning(f"          SESSION ID: {session_id}")
                logger.warning(f"          CURRENT STATUS: {current_status.upper()}")
                logger.warning(f"          ORIGINAL TASK ID: {existing_status.get('task_id')}")
                logger.warning(f"          STARTED AT: {existing_status.get('started_at')}")
                logger.warning(f"          ACTION: REJECTING DUPLICATE REQUEST (IDEMPOTENCY)")
                logger.warning(f"{'='*80}")

                log_structured('WARNING', 'DUPLICATE SESSION REJECTED',
                    session_id=session_id,
                    stage='duplicate_rejection',
                    task_id=task_id,
                    filename=filename,
                    current_status=current_status,
                    existing_task_id=existing_status.get('task_id'))

                # Skip processing this duplicate
                continue

            else:
                # ============================================================
                # LOG: Session validated - PROCEED
                # ============================================================
                if duplicate_check['existing_status']:
                    old_status = duplicate_check['existing_status'].get('status', 'unknown')
                    logger.info(f"[OK] SESSION RETRY ALLOWED")
                    logger.info(f"     FILE: {filename}")
                    logger.info(f"     PREVIOUS STATUS: {old_status.upper()}")
                    logger.info(f"     ACTION: PROCEEDING WITH RETRY")

                    log_structured('INFO', 'SESSION RETRY ALLOWED',
                        session_id=session_id,
                        stage='session_retry',
                        task_id=task_id,
                        previous_status=old_status)
                else:
                    logger.info(f"[OK] SESSION VALIDATED")
                    logger.info(f"     FILE: {filename}")
                    logger.info(f"     STATUS: NEW UNIQUE SESSION")
                    logger.info(f"     ACTION: PROCEEDING WITH PROCESSING")

                    log_structured('INFO', 'SESSION VALIDATED - UNIQUE',
                        session_id=session_id,
                        stage='session_validated',
                        task_id=task_id)

            # Update status to processing
            write_status(
                bucket_name, filename, 'processing',
                session_id=session_id,
                task_id=task_id,
                input_key=input_key,
                sender_agent=sender_agent,
                started_at=datetime.utcnow().isoformat(),
                sqs_message_id=record['messageId'],
                agent_type='ResearchAgent'
            )

            # Process the research task
            result = process_research_task(
                bucket_name=bucket_name,
                input_key=input_key,
                session_id=session_id,
                task_id=task_id,
                filename=filename
            )

            if result['status'] == 'success':
                log_structured('INFO', 'SQS task completed successfully',
                    session_id=session_id,
                    stage='sqs_completion',
                    task_id=task_id,
                    places_found=result.get('places_found', 0))
            else:
                log_structured('ERROR', 'SQS task failed',
                    session_id=session_id,
                    stage='sqs_error',
                    task_id=task_id,
                    error=result.get('error'))

        except Exception as e:
            logger.error(f"Error processing SQS message: {e}", exc_info=True)
            # Re-raise to trigger SQS retry mechanism
            raise

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Processed successfully'})
    }


def process_research_task(bucket_name: str, input_key: str, session_id: str, task_id: str, filename: str):
    """
    Process a single research task.

    Args:
        bucket_name: S3 bucket name
        input_key: S3 key for input file
        session_id: Session ID
        task_id: Unique task ID
        filename: Filename for status tracking (e.g., '20251031T003447_f312ea72.json')

    Returns:
        Result dictionary with status and details
    """
    processing_start = time.time()

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as input_tmp:
        input_tmp_path = input_tmp.name

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as output_tmp:
        output_tmp_path = output_tmp.name

    try:
        # Download input file from S3
        download_start = time.time()
        log_structured('INFO', 'Downloading input from S3',
            session_id=session_id,
            stage='s3_download',
            s3_uri=f"s3://{bucket_name}/{input_key}")

        s3_client.download_file(bucket_name, input_key, input_tmp_path)
        download_duration = time.time() - download_start

        log_structured('INFO', 'S3 download completed',
            session_id=session_id,
            stage='s3_download',
            duration_ms=round(download_duration * 1000, 2))

        # Run research processing
        research_start = time.time()
        log_structured('INFO', 'Starting research processing',
            session_id=session_id,
            stage='research_processing')

        result = research_places(input_tmp_path, output_tmp_path, session_id=session_id)
        research_duration = time.time() - research_start

        # Check for errors
        if 'error' in result:
            log_structured('ERROR', 'Research processing failed',
                session_id=session_id,
                stage='research_processing',
                error=result['error'])

            write_status(
                bucket_name, filename, 'failed',
                session_id=session_id,
                task_id=task_id,
                error=result['error'],
                failed_at=datetime.utcnow().isoformat(),
                duration_ms=round((time.time() - processing_start) * 1000, 2),
                agent_type='ResearchAgent'
            )

            return {
                'status': 'failed',
                'error': result['error']
            }

        log_structured('INFO', 'Research processing completed',
            session_id=session_id,
            stage='research_processing',
            duration_ms=round(research_duration * 1000, 2),
            places_found=result['retrieval']['places_found'])

        # Upload output file to S3
        upload_start = time.time()
        output_key = f"{OUTPUT_PREFIX}{filename}"

        log_structured('INFO', 'Uploading output to S3',
            session_id=session_id,
            stage='s3_upload',
            s3_uri=f"s3://{bucket_name}/{output_key}")

        s3_client.upload_file(output_tmp_path, bucket_name, output_key)
        upload_duration = time.time() - upload_start

        log_structured('INFO', 'S3 upload completed',
            session_id=session_id,
            stage='s3_upload',
            duration_ms=round(upload_duration * 1000, 2))

        # Calculate total duration
        total_duration = time.time() - processing_start

        # Update status to completed (status file kept permanently)
        write_status(
            bucket_name, filename, 'completed',
            session_id=session_id,
            task_id=task_id,
            output_key=output_key,
            output_location=f"s3://{bucket_name}/{output_key}",
            places_found=result['retrieval']['places_found'],
            attractions_count=result['retrieval']['attractions_count'],
            food_count=result['retrieval']['food_count'],
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=round(total_duration * 1000, 2),
            processing_time_seconds=result['retrieval']['time_elapsed'],
            agent_type='ResearchAgent'
        )

        # Status file is now kept permanently as audit record
        log_structured('INFO', 'Status file updated to completed',
            session_id=session_id,
            stage='completion')

        return {
            'status': 'success',
            'output_key': output_key,
            'output_location': f"s3://{bucket_name}/{output_key}",
            'places_found': result['retrieval']['places_found'],
            'attractions_count': result['retrieval']['attractions_count'],
            'food_count': result['retrieval']['food_count'],
            'duration_ms': round(total_duration * 1000, 2)
        }

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        log_structured('ERROR', 'S3 error',
            session_id=session_id,
            stage='s3_error',
            error_code=error_code,
            error_message=str(e))

        write_status(
            bucket_name, filename, 'failed',
            session_id=session_id,
            task_id=task_id,
            error=f"S3 error: {str(e)}",
            error_code=error_code,
            failed_at=datetime.utcnow().isoformat(),
            agent_type='ResearchAgent'
        )

        return {
            'status': 'failed',
            'error': f"S3 error: {str(e)}"
        }

    except Exception as e:
        log_structured('ERROR', 'Unexpected error',
            session_id=session_id,
            stage='error',
            error_type=type(e).__name__,
            error_message=str(e))

        write_status(
            bucket_name, filename, 'failed',
            session_id=session_id,
            task_id=task_id,
            error=f"Internal error: {str(e)}",
            error_type=type(e).__name__,
            failed_at=datetime.utcnow().isoformat(),
            agent_type='ResearchAgent'
        )

        return {
            'status': 'failed',
            'error': str(e)
        }

    finally:
        # Clean up temporary files
        try:
            os.unlink(input_tmp_path)
            os.unlink(output_tmp_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")