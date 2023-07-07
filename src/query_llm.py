import os
from query_index import Ue5DocSearch
import torch
import logging
import re
from utils.parse_tools import remove_tabbed_lines
logging.disable(logging.INFO)

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

def main(
    query,
    embedder = "instructor",
    top_k = None, 
    block_types = None, 
    score = False, 
    open_url = True,
    print_output = True,
    return_output = False
    ):

    # Set up query
    query_machine = Ue5DocSearch(
        embedder=embedder,
        top_k=top_k,
        block_types=block_types,
        score=score,
        open_url=open_url,
        print_output=print_output
    )

    query_output = query_machine(query)

    #print(f'query_output: {query_output}')

    prompt_folder = '/Users/anvil/Documents/llm/llama.cpp/prompts'

    # Set up prompt template
    prompt = f"""
Below is an relevant documentation and a query. Write a response that appropriately completes the query based on the relevant documentation provided.

Relevant documentation: {remove_tabbed_lines(query_output)}


Query: {query}

Response: Here's the answer to your query:"""
#Answer: Here's how to solve your problem:
    # Return prompt if return_output is True
    if return_output:
        print(prompt)
        return prompt
    
    # Save prompt to file if return_output is False
    else:
        with open(os.path.join(prompt_folder, "custom_prompt.txt"), "w") as f:
            f.write(prompt)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--block_types', type=str, default='text')
    parser.add_argument('--score', type=bool, default=False)
    parser.add_argument('--open_url', type=bool, default=False)
    parser.add_argument('--embedder', type=str, default='instructor')
    parser.add_argument('--print_output', type=bool, default=False)
    parser.add_argument('--return_output', type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))