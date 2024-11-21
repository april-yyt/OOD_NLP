import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import argparse
import os
from tqdm import tqdm
import pandas as pd
import warnings
import requests
import json
from time import sleep
from datetime import datetime
from langchain_ollama.llms import OllamaLLM
import logging
from src.utils import PROCESSOR, DATASET_LIST, VERBALIZER
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_llm import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ood_evaluation.log'),
        logging.StreamHandler()
    ]
)

class OODEvaluator:
    def __init__(self, model_id: str, checkpoint_dir: str = "checkpoints/ood", test_mode: bool = False):
        self.model_id = model_id
        self.checkpoint_dir = checkpoint_dir
        self.test_mode = test_mode
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            self.model = OllamaLLM(model=model_id)
            logging.info(f"Successfully initialized model: {model_id}")
            if test_mode:
                logging.info("Running in TEST MODE with reduced samples")
        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            raise

    def save_checkpoint(self, results: dict, task: str, dataset_name: str, phase: str):
        """Save evaluation checkpoint"""
        try:
            checkpoint_file = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_{self.model_id}_{task}_{dataset_name}_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")

    def evaluate_dataset(self, wrapped_dataset, labels, task, dataset_name, setting):
        """Evaluate model on a single dataset with checkpointing"""
        
        # If in test mode, only use first 5 samples
        if self.test_mode:
            wrapped_dataset = wrapped_dataset[:5]
            labels = labels[:5]
            logging.info(f"TEST MODE: Using only {len(wrapped_dataset)} samples")
            
        # Check for existing results
        output_path = os.path.join("llm_temp_output", task, dataset_name, self.model_id, f"{setting}.tsv")
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, sep='\t')
            prediction_list = df["prediction"].astype(str).tolist()
            if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
                reference_list = df["reference"].tolist()
            elif task in ["NameEntityRecognition", "QuestionAnswering"]:
                reference_list = [eval(ref) for ref in df["reference"].astype(str).tolist()]
            else:
                logging.error(f"Unknown task: {task}")
                return None
        else:
            prediction_list = []
            reference_list = []

        # Process remaining samples
        total = len(wrapped_dataset)
        checkpoint_interval = max(10, total // 10)  # Save every 10% of progress
        inference_results = []  # Collect all inference results

        # Define the inference results file path
        inference_output_path = os.path.join("llm_temp_output", task, dataset_name, self.model_id, f"{setting}_all_inferences.tsv")
        os.makedirs(os.path.dirname(inference_output_path), exist_ok=True)

        if len(prediction_list) < total:
            for i, (input, label) in enumerate(tqdm(zip(wrapped_dataset, labels)), start=len(prediction_list)):
                if i % checkpoint_interval == 0:
                    logging.info(f"{dataset_name} Evaluating {i}/{total}")
                    # Save progress to the single inference results file
                    if inference_results:
                        df = pd.DataFrame(inference_results)
                        df.to_csv(inference_output_path, sep="\t", index=None, mode='a', header=not os.path.exists(inference_output_path))

                # Query with retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        result = self.model.invoke(input).strip()
                        break
                    except Exception as e:
                        logging.error(f"Error on attempt {retry + 1}: {str(e)}")
                        if retry == max_retries - 1:
                            raise
                        sleep(1)

                prediction_list.append(result)
                reference_list.append(label)

                # Collect results for this inference
                inference_results.append({
                    "input": input,
                    "prediction": result,
                    "reference": label
                })

        # Save any remaining inference results at the end
        if inference_results:
            df = pd.DataFrame(inference_results)
            df.to_csv(inference_output_path, sep="\t", index=None, mode='a', header=not os.path.exists(inference_output_path))
            logging.info(f"All inference results saved to: {inference_output_path}")

        # Save final results
        df = pd.DataFrame({"prediction": prediction_list, "reference": reference_list})
        os.makedirs(os.path.join("llm_temp_output", task, dataset_name, self.model_id), exist_ok=True)
        df.to_csv(output_path, sep="\t", index=None)
        
        # Process results based on task type
        try:
            if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
                verbalizer = VERBALIZER[task][dataset_name]
                predictions = []
                references = []
                count = 0
                
                
                # Add debug logging
                logging.info(f"Verbalizer for {task}/{dataset_name}: {verbalizer}")
                
                for pred, ref in zip(prediction_list, reference_list):
                    logging.debug(f"Raw prediction: '{pred}'")
                    try:
                        # Extract the last word/line which usually contains the actual prediction
                        pred_cleaned = pred.strip().lower().split()[-1].strip('.:!,')
                        
                        # Remove common wrapper text
                        pred_cleaned = pred_cleaned.replace('prediction:', '').strip()
                        
                        logging.debug(f"Cleaned prediction: '{pred_cleaned}'")
                        
                        # Try exact match first
                        if pred_cleaned in verbalizer:
                            predictions.append(verbalizer.index(pred_cleaned))
                            references.append(ref)
                            continue
                            
                        # Try finding any verbalizer term in the text
                        matched = False
                        for i, verb in enumerate(verbalizer):
                            if verb in pred_cleaned:
                                predictions.append(i)
                                references.append(ref)
                                matched = True
                                logging.debug(f"Found verbalizer term '{verb}' in prediction")
                                break
                                
                        if not matched:
                            count += 1
                            logging.warning(f"No verbalizer match found in: '{pred_cleaned}'")
                            
                    except Exception as e:
                        count += 1
                        logging.warning(f"Failed to process prediction: '{pred}', Error: {str(e)}")
                        continue
                
                logging.info(f"{count}/{len(prediction_list)} format errors!")
                
                # Add check for empty predictions
                if not predictions:
                    logging.error("No valid predictions after processing!")
                    return {
                        "error": "No valid predictions",
                        "total_samples": len(prediction_list),
                        "format_errors": count
                    }
                    
            elif task == "NameEntityRecognition":
                predictions = [extract_entity_from_gpt_output(pred) for pred in prediction_list]
                references = reference_list
            elif task == "QuestionAnswering":
                predictions = prediction_list
                references = reference_list
            else:
                raise NotImplementedError

            results = compute_metric(task, predictions, references)
            
            # Save final processed results
            final_results = {
                "task": task,
                "dataset": dataset_name,
                "setting": setting,
                "metrics": results,
                "format_errors": count if 'count' in locals() else 0
            }
            self.save_checkpoint(final_results, task, dataset_name, "final")
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing results: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="zero-shot", 
                       choices=["zero-shot", "in-context", "ood-in-context"])
    parser.add_argument('--models', nargs='+', default=["llama2", "mixtral"],
                       help="List of models to evaluate")
    parser.add_argument('--test', action='store_true',
                       help="Run in test mode with only 5 samples per dataset")
    parser.add_argument('--test_tasks', nargs='+', 
                       default=["SentimentAnalysis"],
                       help="Specify tasks to test (only used in test mode)")
    args = parser.parse_args()

    if args.test:
        logging.info("Running in TEST MODE")
        tasks = args.test_tasks
    else:
        tasks = ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference", 
                "NameEntityRecognition", "QuestionAnswering"]

    for model_id in args.models:
        logging.info(f"Starting evaluation of {model_id}")
        evaluator = OODEvaluator(model_id=model_id, test_mode=args.test)
        
        for task in tasks:
            processor = PROCESSOR[task]()
            
            # Evaluate all datasets for each task in test mode
            dataset_list = DATASET_LIST[task]
            
            for dataset_name in dataset_list:
                logging.info(f"Evaluating {dataset_name}")

                if args.setting == "ood-in-context" and dataset_name not in PROMPT[args.setting].keys():
                    continue

                try:
                    dataset_path = os.path.join("datasets", "process", task, dataset_name)
                    dataset = processor.get_examples(dataset_path, "test")

                    # In test mode, add suffix to output directories
                    result_dir = "llm_results_test" if args.test else "llm_results"
                    os.makedirs(os.path.join(result_dir, task, dataset_name, model_id, args.setting), exist_ok=True)
                    
                    wrapped_dataset, labels = wrap_dataset(dataset, args.setting, task, dataset_name)
                    result = evaluator.evaluate_dataset(wrapped_dataset, labels, task, dataset_name, args.setting)
                    
                    if result:
                        result_path = os.path.join(result_dir, task, dataset_name, model_id, args.setting, "0.tsv")
                        pd.DataFrame([result]).to_csv(result_path, sep="\t", index=False)
                        logging.info(f"{dataset_name} Result: {result}")
                    
                except Exception as e:
                    logging.error(f"Error processing {dataset_name}: {str(e)}")
                    continue

            logging.info(f"Finished task: {task}")
        logging.info(f"Finished evaluating {model_id}")

if __name__ == "__main__":
    main()