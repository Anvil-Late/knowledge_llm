#!/bin/bash

# Get the query from the command-line argument
query="$1"

# Launch foo.py with argument --query
if ! prompt=$(python src/query_llm.py --query "$query" --top_k 1 --return_output True); then
    echo "Error running query_llm.py"
    exit 1
fi

# Run the macOS terminal command
/Users/anvil/Documents/llm/llama.cpp/main \
    -t 8 \
    -m /Users/anvil/Documents/llm/llama.cpp/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_0.bin \
    --color \
    -c 15000 \
    --temp 0.7 \
    --repeat_penalty 1.1 \
    -n -1 \
    -p "$prompt" \
    -ngl 1 