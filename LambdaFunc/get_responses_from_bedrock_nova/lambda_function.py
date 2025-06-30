import json # Used for JSON serialization and deserialization
import os # Used to access environment variables
import boto3 # AWS SDK for Python
from textwrap import dedent # Used to remove common leading whitespace from multiline strings
from botocore.exceptions import ClientError # Specific exception for AWS service errors

# Initialize AWS clients
s3 = boto3.client('s3') # S3 client for interacting with S3 buckets
events_client = boto3.client('events') # EventBridge client for publishing events

# Retrieve environment variables for configuration
event_bus_name = os.environ['EVENT_BUS_NAME'] # Name of the EventBridge event bus
s3_bucket_name = os.environ['S3_BUCKET_NAME'] # S3 bucket to store LLM responses
# Get the AWS region for Bedrock, default to 'us-east-1' if not set
aws_region = os.environ.get('AWS_REGION', 'us-east-1') 

# Initialize Bedrock Runtime client specific to the region
bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)

def format_prompt_with_context(question: str, context: str):
    """
    Formats the user's question and an optional context into a structured prompt
    for the LLM to generate an answer. This helps guide the LLM to use the provided
    context appropriately and improves response quality.
    """
    if context and context.strip():
        # If context is provided and is not empty, embed it along with the question
        return dedent(
            f"""
            Use the following context to answer the question:
            <context>
            {context}
            </context>

            <question>
            {question}
            </question>

            Your answer should be based solely on the provided context. If the answer cannot be found in the context, state that.
            Answer:
            """
        )
    else:
        # If no context is provided, just present the question
        return dedent(
            f"""
            Answer the following question:
            <question>
            {question}
            </question>

            Answer:
            """
        )

def lambda_handler(event, context):
    """
    AWS Lambda handler function for invoking the AWS Bedrock Nova Lite LLM.
    It processes an evaluation event from EventBridge, checks if Bedrock Nova Lite is requested,
    formats the prompt, calls the Bedrock `converse` API, saves the LLM response to S3,
    and publishes an EventBridge event indicating the success or failure of the invocation.
    """
    llm_name = "Bedrock Nova Lite" # Define the specific LLM name this Lambda function is responsible for
    evaluation_id = "N/A" # Default value for evaluation_id, used in error logs if ID extraction fails early

    try:
        # Extract details from the incoming EventBridge event.
        # This event is typically of DetailType 'EvaluationInitiated' and contains
        # the original prompt details and which LLMs are requested.
        detail = event['detail']
        evaluation_id = detail['evaluation_id'] # Unique ID for the overall evaluation run
        prompt_text = detail['prompt'] # The core prompt text
        context_text = detail.get('context', '') # Optional context, defaults to empty string if not present

        # Retrieve the list of requested LLMs for this specific evaluation instance.
        # This list determines which LLMs should actually process this event.
        requested_llms_in_event = detail.get('selected_llms', [])
        
        # --- Crucial self-filtering logic for fan-out EventBridge rules ---
        # If this Lambda's specific LLM ('Bedrock Nova Lite') is NOT in the list of requested LLMs for this event,
        # it means this invocation is not relevant for this particular evaluation prompt.
        # In this case, the Lambda logs a message and exits gracefully, preventing unnecessary
        # API calls and resource usage.
        if llm_name not in requested_llms_in_event:
            print(f"Skipping invocation for {llm_name}: Not requested in this evaluation ({evaluation_id}). Requested LLMs: {requested_llms_in_event}")
            return {
                'statusCode': 200, # Return 200 OK because the skip is an expected, successful outcome
                'body': f'{llm_name} not requested for this evaluation, skipping.'
            }
            
        # Format the prompt using the helper function. This prepares the text
        # for the LLM API call, optionally including context.
        formatted_prompt = format_prompt_with_context(prompt_text, context_text)

        # Get the Bedrock model ID from environment variables, with a default.
        # Ensure this model ID is enabled in your AWS account and region.
        model_id = os.environ.get('BEDROCK_MODEL_ID', 'amazon.titan-text-express-v1')

        print(f"Invoking {llm_name} model: {model_id} for eval ID: {evaluation_id}")

        # Construct the messages payload for the Bedrock `converse` API.
        # The `converse` API expects a specific format for messages and system prompts.
        messages_for_bedrock = [
            {"role": "user", "content": [{"text": formatted_prompt}]}
        ]

        system_messages_for_bedrock = [
            {"text": "You are a helpful assistant. Provide concise and accurate answers based on the provided context."}
        ]

        # Make the API call to Bedrock's `converse` endpoint
        response = bedrock_runtime.converse(
            modelId=model_id, # The specific Bedrock model to use
            messages=messages_for_bedrock, # User messages
            system=system_messages_for_bedrock, # System instructions for the model
            inferenceConfig={"temperature": 0.0, "maxTokens": 512} # Inference parameters (temperature for determinism, max tokens)
        )

        # Extract the generated text content from the Bedrock response.
        # The structure can be nested depending on the response format.
        llm_response = response['output']['message']['content'][0]['text']

        # Construct the S3 key for storing this LLM's response.
        # Format: evaluation_id/llm_name_response.json (e.g., 'abc-123/bedrock_nova_lite_response.json')
        # Replaces spaces with underscores and converts to lowercase for valid S3 keys.
        s3_key = f"{evaluation_id}/{llm_name.lower().replace(' ', '_')}_response.json"
        
        # Save the LLM's raw response and the original prompt sent to S3
        s3.put_object(
            Bucket=s3_bucket_name, # The S3 bucket configured via environment variable
            Key=s3_key, # The unique S3 key for this response
            Body=json.dumps({'response': llm_response, 'original_prompt_sent_to_llm': formatted_prompt}) # Store as JSON
        )
        print(f"Saved {llm_name} response to S3: {s3_key}")

        # Publish an EventBridge event to notify downstream services that
        # this LLM's response has been successfully received and saved.
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval', # Custom source identifier for your evaluation system
                    'DetailType': 'LLMResponseReceived', # Indicates a successful LLM response
                    'Detail': json.dumps({ # Payload containing key details for downstream processing
                        'evaluation_id': evaluation_id,
                        'llm_name': llm_name,
                        's3_key': s3_key, # S3 key to retrieve the response later
                        'status': 'RECEIVED' # Status indicating success
                    }),
                    'EventBusName': event_bus_name # The target EventBridge bus
                }
            ]
        )
        print(f"Published LLMResponseReceived event for {llm_name}.")

        # Return a successful HTTP response for the Lambda invocation
        return {'statusCode': 200, 'body': f'{llm_name} invocation successful'}

    except ClientError as e:
        # Handle specific AWS Boto3 client errors (e.g., API throttling, permission issues)
        error_message = f"Bedrock ClientError: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        print(error_message)
        
        # Publish an EventBridge event to signal that this LLM's invocation failed due to an AWS service error.
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval',
                    'DetailType': 'LLMResponseFailed', # Indicates a failed LLM response
                    'Detail': json.dumps({ # Payload with error details
                        'evaluation_id': evaluation_id,
                        'llm_name': llm_name,
                        'error': error_message # Include the specific Bedrock error message
                    }),
                    'EventBusName': event_bus_name
                }
            ]
        )
        # Return an HTTP 500 error response
        return {'statusCode': 500, 'body': json.dumps({'error': error_message})}
    except Exception as e:
        # Catch any other unexpected errors during the LLM invocation process
        print(f"Error invoking {llm_name} for evaluation {evaluation_id}: {e}")
        
        # Publish a generic EventBridge event for other types of failures.
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval',
                    'DetailType': 'LLMResponseFailed', # Indicates a failed LLM response
                    'Detail': json.dumps({ # Payload with error details
                        'evaluation_id': evaluation_id,
                        'llm_name': llm_name,
                        'error': str(e) # Include the generic error message
                    }),
                    'EventBusName': event_bus_name
                }
            ]
        )
        # Return an HTTP 500 error response
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}