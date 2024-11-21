#!/bin/bash

# Set up paths and logging
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ood_eval_${TIMESTAMP}.log"

# Function to log messages
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# Function to check if Ollama is running
check_ollama() {
    if ! curl -s http://localhost:11434/api/version > /dev/null; then
        log "Error: Ollama is not running. Starting Ollama..."
        ollama serve &
        sleep 5  # Wait for Ollama to start
        if ! curl -s http://localhost:11434/api/version > /dev/null; then
            log "Failed to start Ollama. Please check installation."
            exit 1
        fi
    fi
    log "Ollama is running"
}

# Function to check if model exists and pull if needed
check_model() {
    local model=$1
    if ! curl -s "http://localhost:11434/api/tags" | grep -q "\"name\":\"$model\""; then
        log "Model $model not found. Pulling from Ollama..."
        ollama pull $model
        if [ $? -ne 0 ]; then
            log "Failed to pull model $model"
            exit 1
        fi
    fi
    log "Model $model is available"
}

# Function to switch active model in Ollama
switch_model() {
    local model=$1
    log "Switching to model: $model"
    
    # Stop any running model
    curl -s -X DELETE http://localhost:11434/api/stop > /dev/null
    sleep 2
    
    # Test model availability
    local test_response=$(curl -s -X POST http://localhost:11434/api/generate -d "{
        \"model\": \"$model\",
        \"prompt\": \"test\",
        \"stream\": false
    }")
    
    if ! echo "$test_response" | grep -q "response"; then
        log "Failed to switch to model $model"
        return 1
    fi
    
    log "Successfully switched to model $model"
    return 0
}

# Add to argument parsing section:
TEST_MODE=0
TEST_TASKS="SentimentAnalysis"

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=1
            shift
            ;;
        --test_tasks)
            TEST_TASKS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Function to run evaluation for a model
run_evaluation() {
    local model=$1
    local setting=$2
    
    log "Starting evaluation of $model with setting: $setting"
    
    # Switch to the correct model
    switch_model $model
    if [ $? -ne 0 ]; then
        log "Failed to switch to model $model, skipping evaluation"
        return 1
    fi
    
    # Build command based on test mode
    local cmd="${ROOT_DIR}/src/evaluations/ood-eval.py --setting $setting --models $model"
    if [ $TEST_MODE -eq 1 ]; then
        cmd="$cmd --test --test_tasks $TEST_TASKS"
        log "Running in TEST MODE with tasks: $TEST_TASKS"
    fi
    
    # Run the evaluation script
    python $cmd 2>&1 | tee -a $LOG_FILE
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        log "Successfully completed evaluation of $model"
        return 0
    else
        log "Error during evaluation of $model"
        return 1
    fi
}

# Main execution
main() {
    cd "$ROOT_DIR"  # Change to root directory
    
    # Check if Ollama is running
    check_ollama

    # Settings to evaluate
    SETTINGS=("zero-shot" "in-context" "ood-in-context")
    
    # Models to evaluate
    MODELS=("llama2" "mixtral")
    
    # Check and pull models if needed
    for model in "${MODELS[@]}"; do
        check_model $model
    done
    
    # Run evaluations for each setting and model
    for setting in "${SETTINGS[@]}"; do
        log "Starting evaluations for setting: $setting"
        for model in "${MODELS[@]}"; do
            # Try running evaluation up to 3 times
            max_retries=3
            for ((retry=1; retry<=max_retries; retry++)); do
                log "Attempt $retry of $max_retries for $model"
                if run_evaluation $model $setting; then
                    break
                elif [ $retry -eq $max_retries ]; then
                    log "Failed to evaluate $model after $max_retries attempts"
                else
                    log "Retrying evaluation of $model in 10 seconds..."
                    sleep 10
                fi
            done
            
            # Add a delay between models to allow system to stabilize
            sleep 10
        done
    done
    
    log "All evaluations completed. Check $LOG_FILE for details."
}

# Run main function with error handling
{
    main
} || {
    log "Script failed with error. Check logs for details."
    exit 1
}