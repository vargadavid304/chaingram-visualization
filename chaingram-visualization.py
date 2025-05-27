import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize, ngrams
nltk.download('punkt_tab')
from scipy.sparse import vstack
from sklearn.feature_extraction.text import CountVectorizer

import math
import pickle
import re

import pyvis
from matplotlib import pyplot as plt
from pyvis.network import Network
import networkx as nx
from tqdm import tqdm

def remove_node_and_connect(G, node_to_remove):
    if node_to_remove not in G:
        return G

    parents = list(G.predecessors(node_to_remove))
    children = list(G.successors(node_to_remove))
    G.remove_node(node_to_remove)

    for parent in parents:
        for child in children:
            if parent != child and child not in nx.descendants(G, parent):
                G.add_edge(parent, child)

    return G


def remove_least_important_nodes(G, keep_nodes_number, weight='weight'):
    if keep_nodes_number == 0:
        return G
    G = G.copy()
    node_counts = {node: G.nodes[node].get('count', 1.0) for node in G.nodes()}
    total_count = sum(node_counts.values())
    personalization = {node: count / total_count for node, count in node_counts.items()}
    pagerank = nx.pagerank(G, personalization=personalization, weight=weight)
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    nodes_to_keep = set(sorted_nodes[:keep_nodes_number])

    for node in sorted_nodes[keep_nodes_number:]:
        if node not in nodes_to_keep:
            remove_node_and_connect(G, node)

    return G


def visualize_graph(subG, output_filename='graph.html'):
    nt = Network('800px', '1400px', notebook=False, directed=True, )

    for node, node_attrs in subG.nodes(data=True):
        nt.add_node(node, title=node, size=node_attrs.get('size', 10), color=node_attrs.get('color', '#97c2fc'), font={'size': 24} )

    for source, target, attrs in subG.edges(data=True):
        nt.add_edge(source, target, value=attrs.get('weight', 1))

    nt.show_buttons(filter_=['physics'])
    nt.show(output_filename, notebook=False)



def build_graph3(data):
    G = nx.DiGraph()
    base_size = 10

    for item in data:
        phrase = item['_id']
        count = item['counts_by_itself']
        log_size = base_size + math.log1p(count) * 10
        G.add_node(phrase, size=log_size, title=phrase, count=count)

    for item in data:
        subgram = item['_id']

        for supergram in item['term_counts']:
            if subgram != supergram:
                G.add_edge(supergram, subgram, weight=item['term_counts'][supergram])

    G_reduced = nx.transitive_reduction(G)

    for node in G_reduced.nodes():
        if node in G.nodes():
            G_reduced.nodes[node].update(G.nodes[node])

    for u, v in G_reduced.edges():
        if G.has_edge(u, v):
            G_reduced.edges[u, v].update(G.edges[u, v])

    return G_reduced


def build_subgraph(G, root_nodes):
    subgraph_nodes = set()

    for root_node in root_nodes:
        pattern = r'\b' + re.escape(root_node) + r'\b'
        compiled_pattern = re.compile(pattern)

        related_nodes = [node for node in G if compiled_pattern.search(node)]
        subgraph_nodes.update(related_nodes)

    subgraph = G.subgraph(subgraph_nodes)
    return subgraph


def find_chain_links_given_by_ngrams(text, ngram_list):
    chain_links = []
    text_length = len(text)

    for ngram in ngram_list:
        ngram_words = ngram.split()
        ngram_len = len(ngram_words)

        for i in range(text_length - ngram_len + 1):
            if text[i:i + ngram_len] == ngram_words:
                chain_link = {
                    'from': i,
                    'to': i + ngram_len,
                    'ngrams_counts': [{'ngram': ngram, 'count': 1}],
                    'text': ' '.join(ngram_words)
                }
                chain_links.append(chain_link)

    return chain_links


def merge_chain_links_into_chains(chain_links, original_text):
    if not chain_links:
        return []

    chains = []
    current_chain_link = chain_links[0]

    for next_chain_link in chain_links[1:]:
        if current_chain_link['to'] >= next_chain_link['from']:
            current_chain_link['to'] = max(current_chain_link['to'], next_chain_link['to'])

            for ngram_count in next_chain_link['ngrams_counts']:
                ngram = ngram_count['ngram']
                count = ngram_count['count']
                found = False
                for current_ngram in current_chain_link['ngrams_counts']:
                    if current_ngram['ngram'] == ngram:
                        current_ngram['count'] += count
                        found = True
                        break
                if not found:
                    current_chain_link['ngrams_counts'].append({'ngram': ngram, 'count': count})
        else:
            current_chain_link['text'] = ' '.join(original_text[current_chain_link['from']:current_chain_link['to']])
            chains.append(current_chain_link)
            current_chain_link = next_chain_link

    current_chain_link['text'] = ' '.join(original_text[current_chain_link['from']:current_chain_link['to']])
    chains.append(current_chain_link)

    return chains


def create_combined_ngram_chains(text, ngram_list):
    chain_links = find_chain_links_given_by_ngrams(text, ngram_list)
    chain_links.sort(key=lambda x: x['from'])
    merged_chains = merge_chain_links_into_chains(chain_links, text)
    return merged_chains


def generate_sub_ngrams(ngram_list):
    ngram_dict = {}

    for ngram in ngram_list:
        words = ngram.split()

        sub_ngrams = set()  #

        for start in range(len(words)):
            for end in range(start + 1, len(words) + 1):
                sub_ngram = ' '.join(words[start:end])
                if sub_ngram in ngram_list:
                    sub_ngrams.add(sub_ngram)

        ngram_dict[ngram] = list(sub_ngrams)

    return ngram_dict


def count_specified_ngrams_with_vectorizer_verbose(texts, ngram_list, batch_size=100):
    max_ngram_number = max(len(ngram.split()) for ngram in ngram_list)
    vectorizer = CountVectorizer(tokenizer=word_tokenize, vocabulary=ngram_list, ngram_range=(1, max_ngram_number))
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Total batches: {total_batches}")

    result_accumulator = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        X_batch = vectorizer.fit_transform(batch_texts)

        result_accumulator.append(X_batch)

        print(f"Processed batch {i // batch_size + 1}/{total_batches}")

    if result_accumulator:
        full_result = vstack(result_accumulator)
    else:
        full_result = None

    ngram_counts = np.array(full_result.sum(axis=0)).flatten()
    ngram_features = vectorizer.get_feature_names_out()
    ngram_count_dict = dict(zip(ngram_features, ngram_counts))
    sorted_ngram_count_dict = dict(sorted(ngram_count_dict.items(), key=lambda item: (item[1], -len(item[0].split()))))

    return sorted_ngram_count_dict

def count_overlapping_phrases(ngram, subgram):
    subgram_length = len(subgram.split())

    def custom_tokenizer(ngram):
        words = ngram.split()
        return [' '.join(words[i:i + subgram_length]) for i in range(len(words) - subgram_length + 1)]

    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, vocabulary=[subgram])

    transformed_data = vectorizer.fit_transform([ngram])

    return int(transformed_data.toarray()[0][0])

def get_all_chains(tokenized_texts, ngram_list):
    i = 1
    chains_dict = {}
    for text in tokenized_texts:
        print("decision: ", i)
        i += 1
        chains = create_combined_ngram_chains(text, ngram_list)
        for chain in chains:
            if chain['text'] in chains_dict:
                chains_dict[chain['text']] += 1
            else:
                chains_dict[chain['text']] = 1
    return chains_dict


def analyze_ngrams(texts, ngram_list):
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    chains = get_all_chains(tokenized_texts, ngram_list)

    expanded_ngram_set = set(chains.keys()).union(set(ngram_list))
    ngram_counts = count_specified_ngrams_with_vectorizer_verbose(texts, list(expanded_ngram_set))

    original_ngram_counts = ngram_counts.copy()
    all_ngram_subgrams = generate_sub_ngrams(list(expanded_ngram_set))
    ngram_analyzed_dict = {}

    for ngram in list(expanded_ngram_set):
        ngram_analyzed_dict[ngram] = {}

    filtered_dict = {key: value for key, value in ngram_counts.items() if value != 0}
    filtered_dict = dict(sorted(filtered_dict.items(), key=lambda item: (item[1], -len(item[0].split()))))
    for ngram in filtered_dict:
        all_ngram_occurences = ngram_counts[ngram]
        subgrams = all_ngram_subgrams[ngram]
        for subgram in subgrams:
            count_of_subgram_inside_ngram = count_overlapping_phrases(ngram, subgram)
            ngram_counts[subgram] = ngram_counts[subgram] - (all_ngram_occurences * count_of_subgram_inside_ngram)
            ngram_analyzed_dict[subgram].update({ngram: all_ngram_occurences})

    return ngram_counts, ngram_analyzed_dict, original_ngram_counts


def chaingram_visualization(ngram_list, texts, root_nodes=[], keep_nodes_number=0):
    ngram_counts_minused, ngram_analyzed_dict, original_ngram_counts = analyze_ngrams(texts, ngram_list)
    filtered_dict = {key: value for key, value in ngram_analyzed_dict.items() if value != {}}
    data = []

    for item in filtered_dict:
        if item in sorted(filtered_dict.keys()):
            term_counts = filtered_dict[item]
            converted_term_counts = {key: int(value) for key, value in
                                     term_counts.items()}  # because of np int64 does not work with mongo
            counts_by_itself = converted_term_counts[item]
            total_counts = 0
            for term in converted_term_counts:
                total_counts += count_overlapping_phrases(term, item) * converted_term_counts[term]
            counts_as_subgram = total_counts - counts_by_itself

            data.append({"_id": item,
                         "counts_by_itself": converted_term_counts[item],
                         "counts_as_subgram": counts_as_subgram,
                         "total_counts": total_counts,
                         "term_counts": converted_term_counts})

    G = build_graph3(data)
    if root_nodes is not None and root_nodes != []:
        G = build_subgraph(G, root_nodes)

    visualize_graph(remove_least_important_nodes(G, keep_nodes_number))


#
root_nodes = ['brown']
texts_foxes = [
    "A brown fox was seen leaping over brown water. The big brown fox runs in the forest. A big brown fox runs around with big brown hat and brown pants.",
    "Near the river, the big brown fox jumps around with agility and grace."
]
ngram_list_foxes = [
    "around",
    "brown",
    "jumps",
    "around with",
    "big brown",
    "brown fox",
    "brown wood",
    "fox brown",
    "jumps around",
    "brown fox jumps",
    "brown fox runs",
    "huge brown fox",
    "world is brown"
]
chaingram_visualization(ngram_list_foxes, texts_foxes)
#chaingram_visualization(ngram_list_foxes, texts_foxes, root_nodes)


