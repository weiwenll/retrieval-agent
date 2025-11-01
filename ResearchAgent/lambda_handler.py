"""
Lambda handler for ResearchAgent with S3 input/output

Expected API Gateway payload:
{
    "json_filename": "sessions/f312ea72.json",
    "session_id": "f312ea72"
}

Input location: s3://iss-travel-planner/retrieval_agent/{json_filename}
Output location: s3://iss-travel-planner/planner_agent/{session_id}-output.json
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

# S3 bucket and paths
INPUT_BUCKET = 'iss-travel-planner'
INPUT_PREFIX = 'retrieval_agent/'
OUTPUT_PREFIX = 'planner_agent/'


def lambda_handler(event, context):
    """
    AWS Lambda handler for ResearchAgent

    Args:
        event: API Gateway event containing:
            - json_filename: Input json_filename in retrieval_agent folder
            - session_id: Unique session identifier
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
        json_filename = body.get('json_filename')
        session_id = body.get('session_id')

        # Validate required parameters
        if not json_filename:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: json_filename'
                })
            }

        if not session_id:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: session_id'
                })
            }

        # Construct S3 paths
        input_s3_key = f"{INPUT_PREFIX}{json_filename}"
        output_s3_key = f"{OUTPUT_PREFIX}{session_id}-output.json"

        logger.info(f"Processing request - Session: {session_id}, Input: s3://{INPUT_BUCKET}/{input_s3_key}, Output: s3://{INPUT_BUCKET}/{output_s3_key}")

        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as input_tmp:
            input_tmp_path = input_tmp.name

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as output_tmp:
            output_tmp_path = output_tmp.name

        try:
            # Download input file from S3
            logger.info(f"Downloading input from S3: s3://{INPUT_BUCKET}/{input_s3_key}")
            s3_client.download_file(INPUT_BUCKET, input_s3_key, input_tmp_path)

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
            logger.info(f"Uploading output to S3: s3://{INPUT_BUCKET}/{output_s3_key}")
            s3_client.upload_file(output_tmp_path, INPUT_BUCKET, output_s3_key)

            # Prepare response
            response_body = {
                'status': 'success',
                'message': 'Research completed successfully',
                'session_id': session_id,
                'input_location': f"s3://{INPUT_BUCKET}/{input_s3_key}",
                'output_location': f"s3://{INPUT_BUCKET}/{output_s3_key}",
                'summary': {
                    'retrieval_id': result['retrieval']['retrieval_id'],
                    'places_found': result['retrieval']['places_found'],
                    'attractions_count': result['retrieval']['attractions_count'],
                    'food_count': result['retrieval']['food_count'],
                    'processing_time_seconds': result['retrieval']['time_elapsed']
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
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'status': 'error',
                    'error': f"Input file not found: s3://{INPUT_BUCKET}/{INPUT_PREFIX}{json_filename}"
                })
            }
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': f"S3 error: {str(e)}"
            })
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
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
