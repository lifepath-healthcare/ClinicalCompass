import json 
import os 
import boto3 
from openai import OpenAI 
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
    AWS Lambda handler function for invoking the OpenAI LLM.
    It processes an evaluation event from EventBridge, checks if OpenAI is requested,
    formats the prompt, calls the OpenAI/Azure OpenAI API, saves the LLM response to S3,
    and publishes an EventBridge event indicating the success or failure of the invocation.
    """
    llm_name = "OpenAI" 
    evaluation_id = "N/A" 

    try:
        # Extract details from the incoming EventBridge event.
        detail = event['detail']
        evaluation_id = detail['evaluation_id'] 
        prompt_text = detail['prompt'] 
        context_text = detail.get('context', '') 

        # Retrieve the list of requested LLMs for this specific evaluation instance.
        requested_llms_in_event = detail.get('selected_llms', [])
        
        # --- Crucial self-filtering logic for fan-out EventBridge rules ---
        if llm_name not in requested_llms_in_event:
            print(f"Skipping invocation for {llm_name}: Not requested in this evaluation ({evaluation_id}). Requested LLMs: {requested_llms_in_event}")
            return {
                'statusCode': 200, 
                'body': f'{llm_name} not requested for this evaluation, skipping.'
            }
            
      
        formatted_prompt = format_prompt_with_context(prompt_text, context_text)

        # Initialize parameters for the OpenAI client based on environment variables.
        api_key = os.environ.get('OPENAI_API_KEY')
        client_kwargs = {} 
        model_name = "" 

        # Check if Azure OpenAI specific environment variables are set
        if os.environ.get('AZURE_OPENAI_ENDPOINT'):
            client_kwargs['azure_endpoint'] = os.environ.get('AZURE_OPENAI_ENDPOINT')
            client_kwargs['api_key'] = os.environ.get('AZURE_OPENAI_API_KEY')
            client_kwargs['api_version'] = os.environ.get('AZURE_OPENAI_API_VERSION')
            model_name = os.environ.get('AZURE_OPENAI_MODEL_NAME')
            print(f"Using Azure OpenAI endpoint: {client_kwargs['azure_endpoint']}")
        elif api_key:
           
            client_kwargs['api_key'] = api_key
            model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o') 
            print(f"Using standard OpenAI API key.")
        else:
        
            raise ValueError("No OpenAI or Azure OpenAI API key/endpoint configured for InvokeOpenAILambda.")

        # Instantiate the OpenAI client with the determined parameters
        client = OpenAI(**client_kwargs)

        print(f"Invoking {llm_name} model: {model_name} for eval ID: {evaluation_id}")

        # Make the chat completion API call to the LLM
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_prompt}], 
            temperature=0.0 
        )
        # Extract the generated content from the LLM's response
        llm_response = response.choices[0].message.content

        # Construct the S3 key for storing this LLM's response.
        s3_key = f"{evaluation_id}/{llm_name.lower().replace(' ', '_')}_response.json"
        
        # Save the LLM's raw response and the original prompt sent to S3
        s3.put_object(
            Bucket=s3_bucket_name, 
            Key=s3_key, 
            Body=json.dumps({'response': llm_response, 'original_prompt_sent_to_llm': formatted_prompt}) 
        )
        print(f"Saved {llm_name} response to S3: {s3_key}")

        # Publish an EventBridge event to notify downstream services that
        # this LLM's response has been successfully received and saved.
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval', 
                    'DetailType': 'LLMResponseReceived', 
                    'Detail': json.dumps({ 
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

        
        return {'statusCode': 200, 'body': f'{llm_name} invocation successful'}

    except Exception as e:
       
        print(f"Error invoking {llm_name} for evaluation {evaluation_id}: {e}")
        
        # Publish an EventBridge event to signal that this LLM's invocation failed.
        # This allows downstream services (e.g., status update, completion check) to react.
        events_client.put_events(
            Entries=[
                {
                    'Source': 'com.verdict.eval',
                    'DetailType': 'LLMResponseFailed', 
                    'Detail': json.dumps({ 
                        'evaluation_id': evaluation_id,
                        'llm_name': llm_name,
                        'error': str(e) 
                    }),
                    'EventBusName': event_bus_name
                }
            ]
        )
        # Return an HTTP 500 error response to indicate a server-side failure
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}