import openai
from openai.error import InvalidRequestError
from InstructorEmbedding import INSTRUCTOR
import torch
import glob
import argparse
import pickle
import json
import os
from tqdm import tqdm

def embed(subsection_dict_path, embedder, security, rebuild):
    """Embed the files in the directory.

    Args:
        subsection_dict_path (dict): Path to the dictionary containing the subsections.
        security (str): Security setting. Either "activated" or "deactivated".
                        prevents the function from running if not "deactivated" 
                        and avoids unexpected costs.

    Returns:
        embeddings (dict): Dictionary containing the embeddings.
    """

    # If embeddings already exist, load them
    
    if os.path.exists(os.path.join("./embeddings", f'{embedder}_embeddings.json')) and not rebuild:
        print("Embeddings already exist. Loading them.")
        with open(os.path.join("./embeddings", f'{embedder}_embeddings.json'), 'r') as f:
            embeddings = json.load(f)

    else:
        # initialize dictionary to store embeddings
        embeddings = {}

    # check security if embedder is openai
    if security != "deactivated":
        if embedder == 'openai':
            raise Exception("Security is not deactivated.")
    
    # load subsections
    with open(subsection_dict_path, 'r') as f:
        subsection_dict = json.load(f)


    # For debugging purposes only
    # Compute average text length to embed
    dict_len = len(subsection_dict)
    total_text_len = 0
    for url, subsection in subsection_dict.items():
        total_text_len += len(subsection['content'])
    avg_text_len = total_text_len / dict_len

    # initialize openai api if embedder is 'openai'
    if embedder == "openai":
        openai_model = "text-embedding-ada-002"
        # Fetch API key from environment variable or prompt user for it
        api_key = os.getenv('API_KEY')
        if api_key is None:
            api_key = input("Please enter your OpenAI API key: ")
        openai.api_key = api_key

    # initialize instructor model if embedder is 'instructor'
    elif embedder == "instructor":
        instructor_model = INSTRUCTOR('hkunlp/instructor-xl')

        # set device to gpu if available
        if (torch.backends.mps.is_available()) and (torch.backends.mps.is_built()):
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    else:
        raise ValueError(f"Embedder must be 'openai' or 'instructor'. Not {embedder}")
    
    # loop through subsections
    for url, subsection in tqdm(subsection_dict.items()):
        subsection_name = subsection['title']
        text_to_embed = subsection['content']

        # skip if already embedded
        if url in embeddings.keys():
            continue

        # make request for embedding
        # case 1: openai
        if embedder == 'openai':
            try:
                response = openai.Embedding.create(
                    input=text_to_embed,
                    model=openai_model
                )
                embedding = response['data'][0]['embedding']

            except InvalidRequestError as e:
                print(f"Error with url {url}")
                print('The server couldn\'t fulfill the request.')
                print('Error code: ', e.code)
                print(f'Tried to embed {len(text_to_embed)} characters while average is {avg_text_len}')
                continue

        # case 2: instructor
        elif embedder == 'instructor':
            instruction = "Represent the PEP content for later documentation retrieval:"
            embedding = instructor_model.encode([[instruction, text_to_embed]], device=device)
            embedding = [float(x) for x in embedding.squeeze().tolist()]

        else:
            raise ValueError(f"Embedder must be 'openai' or 'instructor'. Not {embedder}")
        
        # add embedding to dictionary
        embeddings[url] = {
            "title": subsection_name,
            "embedding": embedding,
            "content": text_to_embed
        }

        # save dictionary every 100 iterations
        if len(embeddings) % 100 == 0:
            print(f"Saving embeddings after {len(embeddings)} iterations.")
            # save embeddings to pickle file
            with open(os.path.join("./embeddings", f'{embedder}_embeddings.pkl'), 'wb') as f:
                pickle.dump(embeddings, f)
            # save embeddings to json file
            with open(os.path.join("./embeddings", f'{embedder}_embeddings.json'), 'w') as f:
                json.dump(embeddings, f)

    return embeddings

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', type=str, default='instructor')
    parser.add_argument('--subsections_path', type=str, default='./resources/peps.json')
    parser.add_argument('--security', type=str, default='activated')
    parser.add_argument('--rebuild', type=bool, default=True)
    args = parser.parse_args()
    embeddings = embed(args.subsections_path, args.embedder, args.security, args.rebuild)
    # save embeddings to pickle file
    with open(os.path.join("./embeddings", f'{args.embedder}_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
    # save embeddings to json file
    with open(os.path.join("./embeddings", f'{args.embedder}_embeddings.json'), 'w') as f:
        json.dump(embeddings, f)

