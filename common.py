from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pyvi.ViTokenizer import tokenize
import os
import torch
import time
import json
import gdown

def download_files(url="https://drive.google.com/drive/folders/1Ial6KkjHfYdkU30SZRibfSifgnRl-myA"):
    gdown.download_folder(url, quiet=False, use_cookies=False)

def load_data(data_file='es-exported-index.json', 
                passages_file='es-passages.json',
                min_words = 50,
                max_words = 500,
                data_folder='data/'):
    '''
    Load json data from json file that export from Elastic Search
    '''
    passages = []
    passages_path = data_folder + passages_file
    data_path = data_folder + data_file
    
    if os.path.exists(passages_path):
        print('Load passages from ', passages_path)
        with open(passages_path, 'r') as json_file:
            passages = json.load(json_file)
    else:
        print("Build passages dataset")
        with open(data_path, 'rt', encoding='utf8') as fIn:
            for line in fIn:
                data = json.loads(line.strip())
                nomalized_content = re.sub(r'\s+', ' ', data['_source']['content'])
                words = nomalized_content.split(" ")

                # ignore content less than min_words
                if len(words) <= min_words:
                    continue

                # accept content len > min_words and content len <= max_words
                if len(words) <= max_words:
                    passages.append([data['_source']['book_name'], nomalized_content])
                    continue

                i = 0
                while i < len(words):

                    start_id = i
                    temp_id = start_id + max_words

                    # exist if this is the last batch of passage
                    if temp_id > len(words):
                        sub_words = words[start_id:]
                        passages.append([data['_source']['book_name'], " ".join(sub_words)])
                        break

                    # find the commplete sentence (which token has '.')
                    break_id = temp_id
                    while break_id < len(words):
                        if "." in words[break_id]:
                            break
                        break_id = break_id + 1

                    # make sure the rest of content has at least min_words
                    if len(words[break_id+1:]) <= min_words:
                        passages.append([data['_source']['book_name'], " ".join(words[start_id:])])
                        break
                    else:
                        passages.append([data['_source']['book_name'], " ".join(words[start_id:break_id+1])])

                    # increase i
                    i = break_id + 1

        # store tokenized data
        with open(passages_path, 'w') as json_file:
            json.dump(passages, json_file)
            print('Store passages input data at ', passages_path)
    return passages

def load_tokenized_data(tokenized_data='tokenized-es-exported-index.json', 
                        data_raw="es-exported-index.json", 
                        data_folder='data/', 
                        min_sentence=5):
    '''
    Load pre-tokeinzed data
    '''
    inputs = []
    tokenized_path = data_folder + tokenized_data
    if os.path.exists(tokenized_path):
        print('Load tokenized input data ', tokenized_path)
        with open(tokenized_path, 'r') as json_file:
            inputs = json.load(json_file)
    else:
        print('Tokenizing input data')
        data_path = data_folder + data_raw
        with open(data_path, 'rt', encoding='utf8') as fIn:
            for line in fIn:
                data = json.loads(line.strip())
                if (len(data['_source']['content'].split('.')) >= min_sentence):
                    inputs.append([data['_source']['book_name'], tokenize(data['_source']['content'])])
        
        # store tokenized data
        with open(tokenized_path, 'w') as json_file:
            json.dump(inputs, json_file)
            print('Store tokenized input data at ', tokenized_path)
    return inputs

def load_model(passages, model_name='keepitreal/vietnamese-sbert', data_folder='data/'):
    '''
    Load pre-trained model if it exists, 
    otherwise train from scratch then backup it for reusing next time
    '''
    bi_encoder = SentenceTransformer(model_name)
    embeddings_filepath = model_name + '.pt'

    # replace character '/' by '-'
    embeddings_filepath = data_folder + embeddings_filepath.replace("/", "-")
    print(embeddings_filepath)

    if os.path.exists(embeddings_filepath):
        print('Load model from ', embeddings_filepath)
        corpus_embeddings = torch.load(embeddings_filepath, map_location=torch.device('cpu'))
        corpus_embeddings = corpus_embeddings.float()
        if torch.cuda.is_available():
            corpus_embeddings = corpus_embeddings.to('cuda')
    else:
        print('Building model from scratch')
        corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
        torch.save(corpus_embeddings, embeddings_filepath)
        print('Save model at ', embeddings_filepath)
    
    return bi_encoder, corpus_embeddings

def search(bi_encoder, corpus_embeddings, 
            query, passages, top_k=10, is_tokenize=False):
    # Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()

    if is_tokenize:
        question_embedding = bi_encoder.encode(tokenize(query), convert_to_tensor=True)
    else:
        question_embedding = bi_encoder.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query
    end_time = time.time()

    # Output of top-k hits
    print("Input question:", query)
    print("Results (after {:.3f} seconds):".format(end_time - start_time))
    results = []
    for hit in hits:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']]))
        results.append(passages[hit['corpus_id']])
    return results, hits

def ranking(hits, query, passages, top_k=10, 
    cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    cross_encoder = CrossEncoder(cross_encoder_name)
    cross_inp = [[query, passages[hit['corpus_id']][1]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    results = []
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']]))
        results.append(passages[hit['corpus_id']])
    return results, hits