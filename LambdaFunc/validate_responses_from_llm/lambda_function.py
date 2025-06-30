import json
import os
import boto3
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE_NAME'])
events_client = boto3.client('events')
event_bus_name = os.environ['EVENT_BUS_NAME']

def lambda_handler(event, context):
    try:
        # Scan for evaluations that are INITIATED or PROCESSING (if a check was already fired)
        # For production, consider using Global Secondary Index or query specific IDs if needed for large tables.
        response = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('status').eq('INITIATED') | 
                             boto3.dynamodb.conditions.Attr('status').eq('PROCESSING')
        )

        items = response.get('Items', [])

        for item in items:
            evaluation_id = item['evaluation_id']
            requested_llms = set(item.get('requested_llms', []))
            llm_responses_received = item.get('llm_responses_received', {})
            current_status = item.get('status')

            # Count how many of the requested LLMs have successfully responded
            received_llms_count = sum(1 for llm in requested_llms if llm_responses_received.get(llm) == 'RECEIVED')

            # Check if all requested LLMs have successfully responded
            if len(requested_llms) > 0 and received_llms_count == len(requested_llms):
                logger.info(f"All {len(requested_llms)} LLM responses received for evaluation: {evaluation_id}.")

                # Prevent re-triggering by immediately updating status to PROCESSING
                # The CalculateLLMMetricsLambda will update it to COMPLETED or FAILED eventually.
                if current_status == 'INITIATED': # Only update if it's currently INITIATED
                    table.update_item(
                        Key={'evaluation_id': evaluation_id},
                        UpdateExpression="SET #s = :new_status",
                        ExpressionAttributeNames={'#s': 'status'},
                        ExpressionAttributeValues={':new_status': 'PROCESSING'}
                    )
                    logger.info(f"Updated evaluation {evaluation_id} status to PROCESSING to prevent re-checks.")

                # Publish AllLLMResponsesReady event
                events_client.put_events(
                    Entries=[
                        {
                            'Source': 'com.verdict.eval',
                            'DetailType': 'AllLLMResponsesReady',
                            'Detail': json.dumps({'evaluation_id': evaluation_id}),
                            'EventBusName': event_bus_name
                        }
                    ]
                )
                logger.info(f"Published AllLLMResponsesReady event for {evaluation_id}.")
            elif current_status in ('INITIATED', 'PROCESSING'):
                # If some responses are still missing or failed, just log and continue for the next scheduled check
                missing_llms = [llm for llm in requested_llms if llm_responses_received.get(llm) != 'RECEIVED']
                logger.info(f"Evaluation {evaluation_id} still pending or processing. Missing: {missing_llms} ({received_llms_count}/{len(requested_llms)} received).")
                # In a real production system, you'd add a timeout here to mark evaluations as FAILED if they don't complete within a threshold.

        return {'statusCode': 200, 'body': 'Completion check successful'}

    except Exception as e:
        logger.error(f"Error in CheckEvaluationCompletionLambda: {e}", exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}