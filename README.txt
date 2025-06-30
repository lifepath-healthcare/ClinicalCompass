#Clinical Compass

## LLM Evaluation Workflow on AWS Lambda

This application leverages a serverless, event-driven architecture on AWS Lambda to automate the evaluation of Large Language Models (LLMs) against specific prompts and reference answers.

### Core Architecture

The system is built around AWS Lambda functions orchestrated by Amazon EventBridge, with data persistence in Amazon DynamoDB and storage of LLM responses and evaluation results in Amazon S3.

### Key Lambda Functions and Their Roles:

#### extract_clinical_telemetry_prompts Lambda
* **Trigger**: S3 PutObject event (when an evaluation prompt file is uploaded).
* **Function**: Reads evaluation prompts from the S3 input file. For each prompt, it initializes a record in DynamoDB (setting LLM response statuses to 'PENDING') and publishes an `EvaluationInitiated` event to EventBridge.

#### get_responses_from_<LLMname> Lambda (e.g., InvokeOpenAILambda, InvokeGeminiLambda, InvokeBedrockNovaLiteLambda, InvokePerplexityLambda)
* **Trigger**: EventBridge rule on `EvaluationInitiated` events. (A single rule fans out to all `Invoke` Lambdas).
* **Function**: Each specific `Invoke` Lambda checks if its corresponding LLM was requested for the evaluation. If so, it formats the prompt, calls the respective LLM API, saves the LLM's raw response to S3, and publishes an `LLMResponseReceived` or `LLMResponseFailed` event.

#### update_microevents_to_ui Lambda
* **Trigger**: EventBridge rule on `LLMResponseReceived` and `LLMResponseFailed` events.
* **Function**: Updates the DynamoDB record for the specific evaluation and LLM, setting its response status (e.g., 'RECEIVED' or 'FAILED').

#### validate_responses_from_llm Lambda
* **Trigger**: EventBridge rule on `LLMResponseReceived` and `LLMResponseFailed` events.
* **Function**: Checks if all requested LLMs for a given evaluation have submitted their responses (either 'RECEIVED' or 'FAILED'). If all responses are in, it updates the overall evaluation status in DynamoDB and publishes an `AllLLMResponsesReady` event.

#### evaluate_llm_metric_scores Lambda
* **Trigger**: EventBridge rule on `AllLLMResponsesReady` events.
* **Function**: On cold start, it downloads and unzips necessary dependencies (like NLTK data or DeepEval components) from S3. It then retrieves LLM responses from S3, calculates various metrics (e.g., ROUGE, BLEU, DeepEval metrics) against the reference answer, saves the comprehensive metrics to S3, and updates the final evaluation status in DynamoDB to 'COMPLETED' or 'FAILED' with error details.

### How it Works (Data Flow):

1.  An input file with prompts is uploaded to an S3 bucket.
2.  `InitiateEvaluationLambda` processes this file, creating DynamoDB entries and emitting an `EvaluationInitiated` event for each prompt.
3.  The `EvaluationInitiated` event triggers all `Invoke*Lambdas`. Each `Invoke*Lambda` selectively calls its respective LLM if it's listed in the event's `selected_llms`.
4.  Successful or failed LLM responses result in `LLMResponseReceived` or `LLMResponseFailed` events, triggering `StatusUpdateLLMResponseLambda` (to update DynamoDB) and `CheckEvaluationCompletionLambda` (to monitor overall progress).
5.  Once `CheckEvaluationCompletionLambda` confirms all LLM responses for an evaluation are in, it emits an `AllLLMResponsesReady` event.
6.  `CalculateLLMMetricsLambda` then processes this event, fetching data, running evaluation metrics, and storing the final results.

This architecture ensures a decoupled, scalable, and resilient workflow for robust LLM evaluation.