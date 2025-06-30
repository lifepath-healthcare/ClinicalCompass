import json
import os
import boto3
import logging
from datetime import datetime
from botocore.exceptions import ClientError
import subprocess # NEW
import sys # NEW

# Add /tmp to the Python path so downloaded modules can be imported
sys.path.insert(1, '/tmp') # NEW

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# NEW: S3 client for downloading dependencies
s3_client_dep = boto3.client('s3')

# --- Dependency Download and Setup (executed on cold start) ---
# Environment variables for dependency S3 bucket and key
# NEW: Function to download a file from S3 to /tmp/ and optionally unzip
def download_and_unzip_dependency(bucket_name: str, key: str):
    local_zip_path = os.path.join('/tmp/', os.path.basename(key))
    
    try:
        print(f"Attempting to download s3://{bucket_name}/{key} to {local_zip_path}")
        s3_client_dep.download_file(bucket_name, key, local_zip_path)
        print(f"Successfully downloaded {key}")

        print(f"Unzipping {local_zip_path} to /tmp/")
        # Use subprocess.run for unzip. '-o' overwrites existing files.
        subprocess.run(['unzip', '-o', local_zip_path, '-d', '/tmp/'], check=True)
        print(f"Successfully unzipped {os.path.basename(key)}.")
        
        # Clean up the zip file after extraction
        os.remove(local_zip_path)
        print(f"Removed temporary zip file: {local_zip_path}")

    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"S3 object s3://{bucket_name}/{key} not found.", exc_info=True)
            raise FileNotFoundError(f"Dependency file not found: s3://{bucket_name}/{key}")
        else:
            logger.error(f"Error downloading or unzipping dependency from S3: {e}", exc_info=True)
            raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Unzip command failed for {local_zip_path}: {e.stderr}", exc_info=True)
        raise RuntimeError(f"Failed to unzip dependency {os.path.basename(key)}") from e
    except Exception as e:
        logger.error(f"Unexpected error during S3 download/unzip: {e}", exc_info=True)
        raise

# --- Dependency Download and Setup (executed on cold start) ---
# Environment variables for dependency S3 bucket and a comma-separated list of keys
DEPENDENCIES_S3_BUCKET = os.environ.get('DEPENDENCIES_S3_BUCKET')
# Example: 'dependencies/llm-metrics-deps.zip,dependencies/nltk-data.zip'
DEPENDENCIES_S3_KEYS_STR = os.environ.get('DEPENDENCIES_S3_KEYS')

_dependencies_setup_complete = False # Flag to ensure this runs only once on cold start

if not _dependencies_setup_complete:
    if DEPENDENCIES_S3_BUCKET and DEPENDENCIES_S3_KEYS_STR:
        dependencies_keys = [k.strip() for k in DEPENDENCIES_S3_KEYS_STR.split(',')]
        
        # Using a marker file to prevent re-downloading on warm starts
        marker_file_path = '/tmp/deepeval_deps_ready'
        if not os.path.exists(marker_file_path):
            logger.info("Cold start or dependencies not detected. Starting dependency download and extraction.")
            try:
                for key in dependencies_keys:
                    download_and_unzip_dependency(DEPENDENCIES_S3_BUCKET, key)
                
                # Create marker file after all dependencies are successfully extracted
                with open(marker_file_path, 'w') as f:
                    f.write('ready')
                _dependencies_setup_complete = True
                logger.info("All dependencies successfully downloaded and unzipped.")
            except Exception as e:
                logger.error(f"Critical error during dependency setup: {e}", exc_info=True)
                raise RuntimeError("Failed to set up necessary dependencies.") from e
        else:
            logger.info("Dependencies already unzipped in /tmp/, skipping download on warm start.")
            _dependencies_setup_complete = True
    else:
        logger.warning("DEPENDENCIES_S3_BUCKET or DEPENDENCIES_S3_KEYS environment variables not fully set. "
                       "Skipping S3 dependency download. Ensure all required libraries are in Lambda layers or provided otherwise.")

# --- DeepEval and NLTK imports (after /tmp is in sys.path) ---
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        HallucinationMetric,
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        ContextualRecallMetric,
        BiasMetric,
        ToxicityMetric,
        GEval
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.models import DeepEvalBaseLLM

    from openai import AzureOpenAI # Make sure this is still correct for your critic LLM

    import nltk
    # NLTK data download: Check if the zip file contains NLTK data or if we still need to download it
    # If your code.zip contains 'wordnet' and 'punkt' under a path like 'nltk_data/',
    # you might need to adjust nltk.data.path or remove this download section.
    # Assuming '/tmp/' is already in nltk.data.path due to 'sys.path.insert(1, '/tmp/')'
    # and if the zip extracts to /tmp/ directly or /tmp/nltk_data etc.
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK data (wordnet and punkt) already present.")
    except nltk.downloader.DownloadError:
        logger.warning("NLTK data (wordnet and punkt) not found. Attempting to download to /tmp/...")
        try:
            nltk.download('wordnet', download_dir='/tmp/', quiet=True)
            nltk.download('punkt', download_dir='/tmp/', quiet=True)
            # Ensure /tmp/ is in NLTK's data path if not already added by sys.path.insert
            if '/tmp/' not in nltk.data.path:
                nltk.data.path.append('/tmp/')
            logger.info("NLTK data (wordnet and punkt) downloaded successfully to /tmp/.")
        except Exception as download_e:
            logger.error(f"Failed to download NLTK data: {download_e}", exc_info=True)
            logger.error("WARNING: METEOR and other NLTK-dependent metrics might not function correctly.")
    except Exception as e:
        logger.error(f"Error checking NLTK data: {e}", exc_info=True)


    from evaluate import load

except ImportError as e:
    logger.error(f"Failed to import a core library. Ensure dependencies are correctly zipped and specified: {e}", exc_info=True)
    # If imports fail after trying to set up /tmp/, it's a critical error
    raise

# Rest of your global variables and functions remain the same
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE_NAME'])

s3_bucket_name = os.environ['S3_BUCKET_NAME'] # This is for storing LLM responses and final metrics

# --- DeepEval Critic LLM Configuration and Initialization ---
critic_openai_key = os.environ.get('CRITIC_OPENAI_KEY')
critic_openai_deployment = os.environ.get('CRITIC_OPENAI_DEPLOYMENT')
critic_openai_api_base = os.environ.get('CRITIC_OPENAI_API_BASE')
critic_openai_api_version = os.environ.get('CRITIC_OPENAI_API_VERSION')

critic_openai_client = None
if critic_openai_key and critic_openai_deployment and critic_openai_api_base and critic_openai_api_version:
    try:
        critic_openai_client = AzureOpenAI(
            api_key=critic_openai_key,
            azure_endpoint=critic_openai_api_base,
            api_version=critic_openai_api_version
        )
        logger.info("Azure OpenAI critic client initialized for DeepEval.")
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI critic client for DeepEval: {e}", exc_info=True)
else:
    logger.warning("DeepEval critic LLM environment variables not fully set. DeepEval critic metrics will be skipped.")

class AzureCriticModel(DeepEvalBaseLLM):
    def __init__(self, openai_client_instance: AzureOpenAI, deployment_name: str):
        self.client = openai_client_instance
        self.deployment_name = deployment_name

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        if not self.client:
            logger.error("Critic LLM client not initialized in AzureCriticModel for generation.")
            return "Error: Critic LLM not available."

        messages = [
            {"role": "system", "content": "You are a fair and knowledgeable LLM evaluator. Your task is to provide concise and objective judgments based on the given criteria."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating from critic model '{self.deployment_name}': {e}", exc_info=True)
            return f"Error: Critic LLM generation failed ({e})"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.deployment_name

critic_model = None
if critic_openai_client:
    critic_model = AzureCriticModel(openai_client_instance=critic_openai_client, deployment_name=critic_openai_deployment)
    logger.info(f"DeepEval critic model '{critic_openai_deployment}' instantiated.")
else:
    logger.warning("DeepEval critic model will not be instantiated as its client failed or env vars are missing. DeepEval metrics will be skipped.")


# --- Load the Evaluation Metrics (once per Lambda instance lifetime) ---
logger.info("Loading core evaluation metrics (HuggingFace Evaluate & SentenceTransformer) globally...")
bleu_metric = None
rouge_metric = None
meteor_metric = None

try:
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")
    meteor_metric = load("meteor")
    logger.info("All core HuggingFace 'evaluate' metrics loaded.")
except Exception as e:
    logger.error(f"Error loading HuggingFace 'evaluate' metrics: {e}. Some basic metrics will be skipped.", exc_info=True)

def calculate_metrics_for_generative_text(llm_response_str: str, reference_answer_str: str, prompt_text: str, context_text: str) -> dict: # Changed to dict for type hint
    scores = {}

    test_case = LLMTestCase(
        input=prompt_text,
        actual_output=llm_response_str,
        expected_output=reference_answer_str,
        context=[context_text],
        retrieval_context=[context_text] if context_text else []
    )

    if rouge_metric:
        try:
            rouge_results = rouge_metric.compute(
                predictions=[llm_response_str],
                references=[reference_answer_str]
            )
            scores["ROUGE-1_F1"] = round(rouge_results["rouge1"], 4)
            scores["ROUGE-2_F1"] = round(rouge_results["rouge2"], 4)
            scores["ROUGE-L_F1"] = round(rouge_results["rougeL"], 4)
        except Exception as e:
            scores["ROUGE_Scores_Error"] = f"Error: {e}"
            logger.error(f"Error calculating ROUGE: {e}", exc_info=True)
    else:
        scores["ROUGE_Scores_Status"] = "Metric not loaded."

    if bleu_metric:
        try:
            tokenized_candidate = nltk.word_tokenize(llm_response_str)
            tokenized_reference = nltk.word_tokenize(reference_answer_str)
            bleu_results = bleu_metric.compute(
                predictions=[tokenized_candidate],
                references=[[tokenized_reference]]
            )
            scores["BLEU"] = round(bleu_results["bleu"], 4)
        except Exception as e:
            scores["BLEU_Error"] = f"Error: {e}"
            logger.error(f"Error calculating BLEU: {e}", exc_info=True)
    else:
        scores["BLEU_Status"] = "Metric not loaded."

    if meteor_metric:
        try:
            meteor_results = meteor_metric.compute(
                predictions=[llm_response_str],
                references=[[reference_answer_str]]
            )
            scores["METEOR"] = round(meteor_results["meteor"], 4)
        except Exception as e:
            scores["METEOR_Error"] = f"Error: {e}"
            logger.error(f"Error calculating METEOR: {e}", exc_info=True)
    else:
        scores["METEOR_Status"] = "Metric not loaded."
    
    if critic_model:
        try:
            deepeval_metrics_list = [
                GEval(
                    name="Coherence",
                    criteria="Coherence - the collective quality of all sentences in the actual output",
                    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                    model=critic_model
                ),
                HallucinationMetric(threshold=0.5, model=critic_model),
                AnswerRelevancyMetric(threshold=0.5, model=critic_model),
                FaithfulnessMetric(threshold=0.5, model=critic_model),
                ContextualRelevancyMetric(threshold=0.7, model=critic_model),
                ContextualRecallMetric(threshold=0.7, model=critic_model),
            ]

            results = evaluate([test_case], deepeval_metrics_list, print_results=False)

            if results and len(results) > 0 and len(results[0].metrics) > 0:
                for metric_result in results[0].metrics:
                    scores[metric_result.metric_name] = {
                        "score": metric_result.score,
                    }
            else:
                scores["DeepEval_Metrics_Status"] = "DeepEval evaluation returned no metrics."
                logger.warning("DeepEval evaluation completed but returned no metrics.")

        except Exception as e:
            logger.error(f"Error calculating DeepEval metrics: {e}", exc_info=True)
            scores["DeepEval_Metrics_Error"] = f"Error: {e}"
    else:
        scores["DeepEval_Metrics_Status"] = "Critic model not initialized, DeepEval metrics skipped."

    return scores


def lambda_handler(event, context):
    evaluation_id = None
    try:
        detail = event['detail']
        evaluation_id = detail['evaluation_id'] # This Lambda expects just evaluation_id in its event

        logger.info(f"Starting metrics calculation for evaluation: {evaluation_id}")

        db_response = table.get_item(Key={'evaluation_id': evaluation_id})
        evaluation_item = db_response.get('Item')

        if not evaluation_item:
            logger.error(f"Evaluation ID {evaluation_id} not found in DynamoDB.")
            return {'statusCode': 404, 'body': 'Evaluation not found'}

        prompt = evaluation_item['prompt']
        reference_answer = evaluation_item['reference_answer']
        context_text = evaluation_item.get('context', "")
        selected_llms = evaluation_item['requested_llms']
        llm_responses_received = evaluation_item['llm_responses_received']

        final_metrics_result = {
            "evaluation_id": evaluation_id,
            "prompt": prompt,
            "reference_answer": reference_answer,
            "context": context_text,
            "model_comparisons": {}
        }

        for llm_name in selected_llms:
            # ONLY process if this LLM's response is marked as RECEIVED
            if llm_responses_received.get(llm_name) == 'RECEIVED':
                # S3 key format must match what Invoke*Lambdas save
                s3_key = f"{evaluation_id}/{llm_name.lower().replace(' ', '_')}_response.json"
                llm_response_text = None
                try:
                    s3_object = s3_client_dep.get_object(Bucket=s3_bucket_name, Key=s3_key) # Use s3_client_dep here
                    response_content = json.loads(s3_object['Body'].read().decode('utf-8'))
                    llm_response_text = response_content.get('response')
                    logger.info(f"Retrieved {llm_name} response from S3: {s3_key}")

                    if llm_response_text:
                        scores = calculate_metrics_for_generative_text(
                            llm_response_text,
                            reference_answer,
                            prompt,
                            context_text
                        )

                        final_metrics_result["model_comparisons"][llm_name] = {
                            "status": "COMPLETED",
                            "llm_response": llm_response_text,
                            "metrics": scores
                        }
                        logger.info(f"Finished metrics for {llm_name} for evaluation {evaluation_id}.")
                    else:
                        final_metrics_result["model_comparisons"][llm_name] = {
                            "status": "EMPTY_RESPONSE",
                            "error": "LLM response in S3 object was empty or malformed."
                        }
                        logger.warning(f"Empty LLM response for {llm_name} in evaluation {evaluation_id}.")

                except Exception as e:
                    logger.error(f"Failed to retrieve or process {llm_name} response for {evaluation_id} from S3: {e}", exc_info=True)
                    final_metrics_result["model_comparisons"][llm_name] = {
                        "status": "ERROR_PROCESSING_S3_RESPONSE",
                        "error": str(e)
                    }
            else:
                final_metrics_result["model_comparisons"][llm_name] = {
                    "status": "SKIPPED_NOT_RECEIVED",
                    "message": f"LLM response for {llm_name} was not marked as RECEIVED in DynamoDB."
                }
                logger.info(f"Skipping {llm_name} as its response was not marked as RECEIVED.")

        final_s3_key = f"{evaluation_id}/final_metrics.json"
        s3_client_dep.put_object( # Use s3_client_dep here for consistency
            Bucket=s3_bucket_name,
            Key=final_s3_key,
            Body=json.dumps(final_metrics_result, indent=2)
        )
        logger.info(f"Saved final aggregated metrics to S3: {final_s3_key}")

        table.update_item(
            Key={'evaluation_id': evaluation_id},
            UpdateExpression="SET #s = :status, final_s3_key = :s3_key, completion_timestamp = :ts",
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={
                ':status': 'COMPLETED',
                ':s3_key': final_s3_key,
                ':ts': datetime.now().isoformat()
            }
        )
        logger.info(f"Evaluation {evaluation_id} marked as COMPLETED in DynamoDB.")

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Metrics calculated and saved successfully!', 'evaluation_id': evaluation_id, 'final_s3_key': final_s3_key})
        }

    except Exception as e:
        logger.error(f"Fatal error in CalculateLLMMetricsLambda for evaluation {evaluation_id}: {e}", exc_info=True)
        if evaluation_id:
            try:
                table.update_item(
                    Key={'evaluation_id': evaluation_id},
                    UpdateExpression="SET #s = :status, error_message = :msg, completion_timestamp = :ts",
                    ExpressionAttributeNames={'#s': 'status'},
                    ExpressionAttributeValues={
                        ':status': 'FAILED',
                        ':msg': f"Overall Calculation Failure: {str(e)}",
                        ':ts': datetime.now().isoformat()
                    }
                )
                logger.info(f"Evaluation {evaluation_id} marked as FAILED in DynamoDB due to overall error.")
            except Exception as db_e:
                logger.error(f"Failed to update DynamoDB status to FAILED for {evaluation_id}: {db_e}", exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
