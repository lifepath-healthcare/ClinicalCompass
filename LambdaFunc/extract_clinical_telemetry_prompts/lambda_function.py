import json 
import os 
import uuid 
from datetime import datetime 
import boto3 

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE_NAME'])

# Initialize EventBridge client
events_client = boto3.client('events')
event_bus_name = os.environ['EVENT_BUS_NAME']

# NEW: Get default selected LLMs from environment variable during Lambda initialization (cold start)
default_selected_llms_str = os.environ.get('DEFAULT_SELECTED_LLMS')
if not default_selected_llms_str:
    # If the environment variable is not set, raise an error immediately.
    # This ensures the Lambda won't start if a crucial configuration is missing.
    raise EnvironmentError("DEFAULT_SELECTED_LLMS environment variable is not set for InitiateEvaluationLambda.")

# Parse the comma-separated string into a list of LLM names, stripping whitespace
DEFAULT_SELECTED_LLMS = [llm.strip() for llm in default_selected_llms_str.split(',')]
print(f"Configured default LLMs for evaluation: {DEFAULT_SELECTED_LLMS}")

def lambda_handler(event, context):
    """
    AWS Lambda handler function to initiate LLM evaluations.
    It processes a request payload (single object or array of objects),
    stores evaluation details in DynamoDB, and publishes an EventBridge event
    for each evaluation.
    """
    try:
        # Attempt to parse the request body.
        # API Gateway often sends the body as a string, so it needs to be loaded as JSON.
        if isinstance(event.get('body'), str):
            request_payload = json.loads(event['body'])
        else:
            # If 'body' is already a dict/list or not present, use it directly (or an empty list)
            request_payload = event.get('body', [])

        # Ensure the request_payload is a list for consistent processing.
        # If a single dictionary is passed, wrap it in a list.
        if not isinstance(request_payload, list):
            if isinstance(request_payload, dict):
                request_payload = [request_payload]
            else:
                # If the payload is neither a list nor a dict, it's an invalid format.
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'Request body must be a JSON array of evaluation objects or a single evaluation object.'})
                }

        processed_evaluations = [] 
        errors_occurred = [] 

        # Iterate through each evaluation request in the payload
        for i, evaluation_request in enumerate(request_payload):
            try:
                # Extract required fields from the current evaluation request
                prompt = evaluation_request.get('prompt')
                reference_answer = evaluation_request.get('reference_answer')
                context_text = evaluation_request.get('context') # Optional context

                
                selected_llms_for_this_item = DEFAULT_SELECTED_LLMS 

                # Validate required inputs
                if not prompt:
                    raise ValueError(f"Prompt is required for item {i}.")
                if not selected_llms_for_this_item: # Check if the default list is empty
                    raise ValueError(f"No LLMs configured for evaluation (DEFAULT_SELECTED_LLMS env var is empty or missing).")

                # Generate a unique ID and timestamp for the evaluation
                evaluation_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()

                # Prepare the item to be stored in DynamoDB
                item = {
                    'evaluation_id': evaluation_id,
                    'prompt': prompt,
                    'reference_answer': reference_answer,
                    'context': context_text,
                    'requested_llms': selected_llms_for_this_item, # Store the selected LLMs from environment
                    'llm_responses_received': {llm: 'PENDING' for llm in selected_llms_for_this_item}, # Initialize response status for each LLM
                    'status': 'INITIATED', # Initial overall status of the evaluation
                    'timestamp': timestamp
                }

                # Put the item into the DynamoDB table
                table.put_item(Item=item)
                print(f"Evaluation {evaluation_id} initiated and saved to DynamoDB for item {i}.")

                # Prepare the detail payload for the EventBridge event
                event_detail = {
                    'evaluation_id': evaluation_id,
                    'prompt': prompt,
                    'reference_answer': reference_answer,
                    'context': context_text,
                    'selected_llms': selected_llms_for_this_item # Include selected LLMs in the event for downstream consumers
                }

                # Publish the 'EvaluationInitiated' event to EventBridge
                events_client.put_events(
                    Entries=[
                        {
                            'Source': 'com.verdict.eval', # Custom source for your application events
                            'DetailType': 'EvaluationInitiated', # Type of event, used by EventBridge rules
                            'Detail': json.dumps(event_detail), # The actual event payload as a JSON string
                            'EventBusName': event_bus_name # The name of your custom EventBridge bus
                        }
                    ]
                )
                print(f"Published EvaluationInitiated event for {evaluation_id} (item {i}).")
                
                # Record successful initiation
                processed_evaluations.append({'item_index': i, 'evaluation_id': evaluation_id, 'status': 'INITIATED'})

            except Exception as item_e:
                # Catch and log errors for individual items within the batch
                error_message = f"Error processing item {i}: {str(item_e)}"
                print(error_message)
                errors_occurred.append({'item_index': i, 'error': error_message, 'input': evaluation_request})

        # Prepare the final HTTP response body
        response_body = {
            'message': 'Batch evaluation initiation completed.',
            'total_items_received': len(request_payload),
            'items_initiated_successfully': len(processed_evaluations),
            'initiated_evaluations': processed_evaluations,
            'errors': errors_occurred # Include any errors for the client to review
        }

        # Set HTTP status code: 200 if all successful, 207 (Multi-Status) if some errors occurred
        status_code = 200 if not errors_occurred else 207

        # Return the HTTP response
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # Required for CORS if called from a web frontend
            },
            'body': json.dumps(response_body)
        }
    except Exception as e:
        # Catch any fatal errors that prevent overall batch processing
        print(f"Fatal error in InitiateEvaluationLambda (overall batch processing): {e}")
        return {
            'statusCode': 500, # Internal Server Error
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': f"An unexpected error occurred during batch processing: {str(e)}"})
        }