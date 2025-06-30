import json
import os
import boto3
from openai import OpenAI 
from textwrap import dedent # Used to remove common leading whitespace from multiline strings

# Initialize AWS clients
s3 = boto3.client('s3')
events_client = boto3.client('events')

# Retrieve environment variables for configuration
event_bus_name = os.environ['EVENT_BUS_NAME'] 
s3_bucket_name = os.environ['S3_BUCKET_NAME'] 

def format_prompt_with_context(question: str, context: str):
    """
    Formats the user's question and an optional context into a structured prompt
    for the LLM to generate an answer. This improves the LLM's ability to focus
    on the provided information.
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
    AWS Lambda handler function for invoking the Perplexity LLM.
    It receives an evaluation event, checks if Perplexity is requested,
    formats the prompt, calls the Perplexity API, saves the response to S3,
    and publishes an EventBridge event indicating success or failure.
    """
    llm_name = "Perplexity" # Define the name of this specific LLM
    evaluation_id = "N/A" # Default value for evaluation_id, in case of early error

    try:
        # Extract details from the incoming EventBridge event
        detail = event['detail']
        evaluation_id = detail['evaluation_id']
        prompt_text = detail['prompt']
        context_text = detail.get('context', '') # Get context, default to empty string if not present

        # Get the list of LLMs specifically requested for this evaluation from the event detail
        requested_llms_in_event = detail.get('selected_llms', [])
        
        # Check if THIS specific LLM (e.g., "Perplexity") is in the list of requested LLMs for this evaluation.
        # This allows a single EventBridge rule to fan out to multiple LLM invocation lambdas,
        # with each lambda self-filtering if it should process the event.
        if llm_name not in requested_llms_in_event:
            print(f"Skipping invocation for {llm_name}: Not requested in this evaluation ({evaluation_id}). Requested LLMs: {requested_llms_in_event}")
            return {
                'statusCode': 200, # Return 200 OK because the skip is an expected, successful outcome
                'body': f'{llm_name} not requested for this evaluation, skipping.'
            }
            
        # Format the prompt using the extracted prompt_text and context_text
        formatted_prompt = format_prompt_with_context(prompt_text, context_text)

        # Initialize the Perplexity AI client using API key and base URL from environment variables
        client = OpenAI(
            api_key=os.environ['PERPLEXITY_API_KEY'],
            base_url=os.environ.get('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai') # Use default if not provided
        )
        # Get the Perplexity model name from environment variables, with a default
        model_name = os.environ.get('PERPLEXITY_MODEL', 'sonar-medium-online')

        print(f"Invoking {llm_name} model: {model_name} for eval ID: {evaluation_id}")

        # Make the API call to Perplexity AI
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_prompt}], # Send the formatted prompt as a user message
            temperature=0.0 # Set temperature to 0 for deterministic and consistent responses
        )
        llm_response = response.choices[0].message.content # Extract the generated content from the response

        # Construct the S3 key for storing the LLM's response
        # Example key: 'your-evaluation-id/perplexity_response.json'
        s3_key = f"{evaluation_id}/{llm_name.lower()}_response.json"
        
        # Save the LLM's raw response and the original prompt sent to S3
        s3.put_object(
            Bucket=s3_bucket_name,
            Key=s3_key,
            Body=json.dumps({'response': llm_response, 'original_prompt_sent_to_llm': formatted_prompt})
        )
        print(f"Saved {llm_name} response to S3: {s3_key}")

        # Publish an EventBridge event indicating that the LLM's response has been received
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval', # Custom source
                    'DetailType': 'LLMResponseReceived', # Event type for response reception
                    'Detail': json.dumps({ # Payload with evaluation details and S3 key
                        'evaluation_id': evaluation_id,
                        'llm_name': llm_name,
                        's3_key': s3_key,
                        'status': 'RECEIVED'
                    }),
                    'EventBusName': event_bus_name
                }
            ]
        )
        print(f"Published LLMResponseReceived event for {llm_name}.")

        # Return a successful HTTP response
        return {'statusCode': 200, 'body': f'{llm_name} invocation successful'}

    except Exception as e:
        # Catch any exceptions that occur during the process
        print(f"Error invoking {llm_name} for evaluation {evaluation_id}: {e}")
        
        # Publish an EventBridge event indicating that the LLM's response failed
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval',
                    'DetailType': 'LLMResponseFailed', # Event type for failed response
                    'Detail': json.dumps({ # Payload with evaluation details and error message
                        'evaluation_id': evaluation_id,
                        'llm_name': llm_name,
                        'error': str(e)
                    }),
                    'EventBusName': event_bus_name
                }
            ]
        )
        # Return an HTTP 500 error response to indicate failure
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}