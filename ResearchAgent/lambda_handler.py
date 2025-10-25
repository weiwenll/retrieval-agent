"""
Lambda handler for ResearchAgent with S3 input/output

Expected API Gateway payload:
{
    "input_s3_key": "inputs/trip-request-123.json",
    "output_s3_key": "outputs/attractions-123.json",
    "s3_bucket": "retrieval-agent-data"  # optional, uses env var if not provided
}

Input file in S3 should contain the same structure as local input.json
Output will be written to S3 in the same format as local output
"""

import json
import os
import logging
import tempfile
import boto3
from botocore.exceptions import ClientError

# Import the research_places function from main.py
from main import research_places, load_input_file

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

# Get S3 bucket from environment variable
DEFAULT_S3_BUCKET = os.environ.get('S3_BUCKET', 'retrieval-agent-data')


def lambda_handler(event, context):
    """
    AWS Lambda handler for ResearchAgent

    Args:
        event: API Gateway event containing:
            - input_s3_key: S3 key for input file
            - output_s3_key: S3 key for output file
            - s3_bucket: (optional) S3 bucket name
        context: Lambda context

    Returns:
        API Gateway response with status and result
    """
    try:
        # Parse request body if it's a string (API Gateway)
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        # Extract parameters
        input_s3_key = body.get('input_s3_key')
        output_s3_key = body.get('output_s3_key')
        s3_bucket = body.get('s3_bucket', DEFAULT_S3_BUCKET)

        # Validate required parameters
        if not input_s3_key:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: input_s3_key'
                })
            }

        if not output_s3_key:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: output_s3_key'
                })
            }

        logger.info(f"Processing request - Bucket: {s3_bucket}, Input: {input_s3_key}, Output: {output_s3_key}")

        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as input_tmp:
            input_tmp_path = input_tmp.name

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as output_tmp:
            output_tmp_path = output_tmp.name

        try:
            # Download input file from S3
            logger.info(f"Downloading input from S3: s3://{s3_bucket}/{input_s3_key}")
            s3_client.download_file(s3_bucket, input_s3_key, input_tmp_path)

            # Run the research agent
            logger.info("Running ResearchAgent...")
            result = research_places(input_tmp_path, output_tmp_path)

            # Check if research was successful
            if 'error' in result:
                logger.error(f"ResearchAgent error: {result['error']}")
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': f"ResearchAgent failed: {result['error']}"
                    })
                }

            # Upload output file to S3
            logger.info(f"Uploading output to S3: s3://{s3_bucket}/{output_s3_key}")
            s3_client.upload_file(output_tmp_path, s3_bucket, output_s3_key)

            # Prepare response
            response_body = {
                'message': 'Research completed successfully',
                'input_s3_key': input_s3_key,
                'output_s3_key': output_s3_key,
                's3_bucket': s3_bucket,
                'summary': {
                    'retrieval_id': result['retrieval']['retrieval_id'],
                    'places_found': result['retrieval']['places_found'],
                    'attractions_count': result['retrieval']['attractions_count'],
                    'food_count': result['retrieval']['food_count'],
                    'time_elapsed': result['retrieval']['time_elapsed']
                }
            }

            logger.info(f"Success: Found {result['retrieval']['places_found']} places")

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
        logger.error(f"S3 error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"S3 error: {str(e)}"
            })
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Internal server error: {str(e)}"
            })
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        'body': {
            'input_s3_key': 'inputs/test-input.json',
            'output_s3_key': 'outputs/test-output.json',
            's3_bucket': 'retrieval-agent-data'
        }
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
