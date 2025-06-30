import json 
import os 
import boto3 
import google.generativeai as genai 
from textwrap import dedent 

# Initialize AWS clients
s3 = boto3.client('s3') 
events_client = boto3.client('events') # EventBridge client for publishing events

# Retrieve environment variables for configuration
event_bus_name = os.environ['EVENT_BUS_NAME'] 
s3_bucket_name = os.environ['S3_BUCKET_NAME'] 

def format_prompt_with_context(question: str, context: str):
    """
    Formats the user's question and an optional context into a structured prompt
    for the LLM to generate an answer. This helps guide the LLM to use the provided
    context appropriately and improves response quality.
    """
    if context and context.strip():
       
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
    AWS Lambda handler function for invoking the Google Gemini LLM.
    It processes an evaluation event from EventBridge, checks if Gemini is requested,
    formats the prompt, calls the Gemini API, saves the LLM response to S3,
    and publishes an EventBridge event indicating the success or failure of the invocation.
    """
    llm_name = "Gemini" 
    evaluation_id = "N/A" 

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
        # If this Lambda's specific LLM ('Gemini') is NOT in the list of requested LLMs for this event,
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

        # Configure the Google Generative AI client with the API key from environment variables
        api_key = os.environ['GEMINI_API_KEY']
        genai.configure(api_key=api_key)
        
        # Get the Gemini model name from environment variables, with a default
        model_name = os.environ.get('GEMINI_MODEL_NAME', 'gemini-1.5-flash')
        model = genai.GenerativeModel(model_name) # Instantiate the GenerativeModel

        print(f"Invoking {llm_name} model: {model_name} for eval ID: {evaluation_id}")

        # Make the content generation API call to the Gemini LLM
        response = model.generate_content(formatted_prompt)
        llm_response = response.text # Extract the generated text content from the response

        # Construct the S3 key for storing this LLM's response.
        # Format: evaluation_id/llm_name_response.json (e.g., 'abc-123/gemini_response.json')
        # Converts LLM name to lowercase and replaces spaces with underscores for valid S3 keys.
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

    except Exception as e:
        # Catch any unexpected errors during the LLM invocation process
        print(f"Error invoking {llm_name} for evaluation {evaluation_id}: {e}")
        
        # Publish an EventBridge event to signal that this LLM's invocation failed.
        # This allows downstream services (e.g., status update, completion check) to react.
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval',
                    'DetailType': 'LLMResponseFailed', # Indicates a failed LLM response
                    'Detail': json.dumps({ # Payload with error details
                        'evaluation_id': evaluation_id,
                        'llm_name': llm_name,
                        'error': str(e) # Include the error message for debugging
                    }),
                    'EventBusName': event_bus_name
                }
            ]
        )
        # Return an HTTP 500 error response to indicate a server-side failure
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}