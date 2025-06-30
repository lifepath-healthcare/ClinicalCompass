import json 
import os 
import boto3 

# Initialize DynamoDB client
# The table name for storing evaluation data is retrieved from environment variables.
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE_NAME'])

def lambda_handler(event, context):
    """
    AWS Lambda handler function for updating the status of LLM responses
    within a DynamoDB evaluation record.
    
    This Lambda is triggered by EventBridge events of DetailType 'LLMResponseReceived'
    or 'LLMResponseFailed', which are published by the InvokeLLM Lambdas.
    """
    try:
        # Extract details from the incoming EventBridge event
        detail = event['detail']
        evaluation_id = detail['evaluation_id'] 
        llm_name = detail['llm_name'] 
       
        llm_status = detail.get('status', 'RECEIVED') # Can be 'RECEIVED' or 'FAILED'

        # Define the DynamoDB UpdateExpression to update a specific attribute within a map.
       
        update_expression = "SET #responses.#llm = :val"
        
        # Define ExpressionAttributeNames for mapping placeholder names to actual attribute names
        expression_attribute_names = {
            '#responses': 'llm_responses_received', 
            '#llm': llm_name 
        }
        
        # Define ExpressionAttributeValues for mapping placeholder values to actual data
        expression_attribute_values = {':val': llm_status} 

        # Perform the update operation on the DynamoDB item.
        # The 'evaluation_id' is the Partition Key used to identify the item.
        table.update_item(
            Key={'evaluation_id': evaluation_id}, 
            UpdateExpression=update_expression, 
            ExpressionAttributeNames=expression_attribute_names, 
            ExpressionAttributeValues=expression_attribute_values 
        )
        print(f"Updated DynamoDB for {llm_name} (status: {llm_status}) in evaluation {evaluation_id}")
        
        # Return a successful HTTP response
        return {'statusCode': 200, 'body': 'Status updated successfully'}
    
    except Exception as e:
        # Catch any unexpected errors during the DynamoDB update process
        print(f"Error in StatusUpdateLLMResponseLambda: {e}")
        # Return an HTTP 500 error response to indicate a server-side failure
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}