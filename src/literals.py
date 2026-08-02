"""
FLORA literal embedding computation utilities.

This module can be run independently from FLORA's main loop. 
The main loop of FLORA can run on CPU, while computing literal embeddings benefits from GPU acceleration.
Therefore, this module can be used to precompute literal embeddings on a GPU machine, save them, and reuse 
them later in the CPU-only FLORA loop.
"""

import re
import os
import pickle
import argparse
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from Prefixes import *
import utils
import Announce
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # safe


# Load pre-trained models for strings
DEFAULT_EMBEDDING_MODEL = 'Lihuchen/pearl_small' # 'sentence-transformers/LaBSE'
model_name = None
Str_model = None
Str_tokenizer = None
# # Multilingual BERT
# Str_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
# Str_model = AutoModel.from_pretrained('sentence-transformers/LaBSE')
# Monolingual BERT
# Str_tokenizer = AutoTokenizer.from_pretrained('Lihuchen/pearl_small')
# Str_model = AutoModel.from_pretrained('Lihuchen/pearl_small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_string_model(embedding_model=None):
    global model_name, Str_model, Str_tokenizer
    selected_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    if Str_model is not None and Str_tokenizer is not None and model_name == selected_model:
        return Str_model, Str_tokenizer

    model_name = selected_model
    Str_model = AutoModel.from_pretrained(model_name)
    Str_tokenizer = AutoTokenizer.from_pretrained(model_name)
    Str_model.to(device) # gpu
    Str_model.eval()
    Announce.doing(f"Pretrained Model {model_name} loaded successfully...")
    Announce.done()
    return Str_model, Str_tokenizer


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_text(model, input_texts):
    if Str_tokenizer is None:
        load_string_model()
    # Tokenize the input texts
    batch_dict = Str_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def string_similarity(input_texts):
    '''
    input_texts: list of strings [source, target1, target2, ...]
    '''
    model, _ = load_string_model()
    embeddings = encode_text(model, input_texts)
    scores = (embeddings[:1] @ embeddings[1:].T) # no * 100
    return scores.tolist()


def is_readable(txt):
    if not txt or len(txt) == 0:
        return False
    # not a web link
    if txt.startswith('http'):
        return False
    non_alpha_ratio = sum(not c.isalpha() for c in txt) / len(txt) 
    return non_alpha_ratio < 0.5



# Regex for literals
literalRegex=re.compile('"([^"]*)"(@([a-z-]+))?(\\^\\^(.*))?')

# Regex for int values
intRegex=re.compile('^"?[+-]?[0-9]+"?$')

# Regex for float values
floatRegex = re.compile('^"?([+-])?([0-9.]+)"?$')
sciFloatRegex = re.compile('^"?([+-])?([0-9.]+[Ee][+-]?[0-9]+)"?$')

# Regex for numbers: post code, phone number, etc. 
# A normalized-number candidate must contain at least one digit.
numberRegex = re.compile(r'(?=.*\d)[\d\W]+')
identifierRegex = re.compile(r'([a-zA-Z]+)(\d+)') # TBD if needed

DATE_DATATYPES = {'xsd:date','xsd:gYear', 'xsd:gYearMonth', 'xsd:dateTime', 'xsd:gMonthDay'}

def isLiteral(term):
    return re.match(literalRegex,term) or re.match(floatRegex,term)

def normalize_datatype(datatype):
    """Normalize the datatype to a standard format. e.g., "http://www.w3.org/2001/XMLSchema#string" -> "xsd:string"."""
    if datatype is None:
        return None
    datatype = datatype.strip()
    if datatype.startswith('<') and datatype.endswith('>'):
        datatype = datatype[1:-1]
    xmlschema_prefix = 'http://www.w3.org/2001/XMLSchema#'
    if datatype.startswith(xmlschema_prefix):
        datatype = 'xsd:' + datatype[len(xmlschema_prefix):]
    return datatype

def unicode(txt):
    # e.g., "Coamo,_Puerto_Rico" -> "Coamo_u002C_Puerto_Rico"
    encoded_str = re.sub(r'[^a-zA-Z0-9_]', lambda x: "_u{:04X}_".format(ord(x.group())), txt)
    return encoded_str

def decode_unicode(encoded_str):
    decoded_string = re.sub(r'_u([0-9A-F]{4})_', lambda x: chr(int(x.group(1), 16)), encoded_str)
    if '\\u' in decoded_string:
        try:
            s = decoded_string.encode('utf-8').decode('unicode_escape')
            return s.encode('utf-16', 'surrogatepass').decode('utf-16')
        except Exception:
            return decoded_string
    else:
        return decoded_string


def splitLiteral(term):
    """ Returns String value, int value, language, and datatype of a term (or None, None, None, None). No good backslash handling """
    
    literal, _, lang, _, datatype = re.match(literalRegex, term).groups()
    datatype = normalize_datatype(datatype)
    # Dates
    if datatype in DATE_DATATYPES:
        return (literal, None, lang, datatype)
    
    # Numbers
    floatmatch = re.match(floatRegex, literal)
    scifloatmatch = re.match(sciFloatRegex, literal)
    if (floatmatch or scifloatmatch) and lang is None:
        try:
            # some identifiers are integers
            Value=int(literal.strip('"'))
            if len(str(Value)) != len(literal.strip('"')):
                # e.g., "06" /= 6, "+3" /= 3
                return (literal, None, lang, datatype) # datatype 'none'
            return (literal, Value, lang, datatype)
        except:
            try:
                # e.g., code version 1.0 /= 1
                Value=float(literal.strip('"'))
                return (literal, Value, lang, datatype)
            except ValueError:
                # e.g. "23.78.9" version
                return (literal, None, lang, datatype)
    
    # Cases for normalized numbers, e.g., phone numbers, post codes
    # Would put it to the string bucket
    matchNumber=re.fullmatch(numberRegex, literal)
    # Check if the string is a number type, e.g., "818/762-1221"
    if matchNumber:
        Value = numeric_normalization(literal)
        if len(Value) >= 10: # e.g., phone numbers, not: ', , , ,'; "1.2"@fr
            return (literal, 'normalized_'+Value, lang, datatype)
    
    # Strings: lowecasing, order-agnostic, decode unicode
    # Pre-processing for string literals
    de_literal = decode_unicode(literal)
    return (de_literal, None, lang, 'xsd:string')


def embedding_strings(kb, batch_size=64, embedding_model=None, emb_file=None):
    model, _ = load_string_model(embedding_model)
    literal2id = {} # {literal:id}
    literals = []
    cnt = 0
    if hasattr(kb, 'iter_literal_object_ids') and hasattr(kb, 'entity_for_id'): # for CompactGraph
        literal_objects = (kb.entity_for_id(object_id) for object_id in kb.iter_literal_object_ids())
    else:
        literal_objects = (object for object in kb.objects() if isLiteral(object))

    for object in literal_objects:
        if isLiteral(object):
            term, _, _, type = splitLiteral(object)
            if len(term) <= 1:
                # print("Empty/Short literal: ", term, object)
                continue
            if (not is_readable(term)) and type == 'xsd:string':
                # print("Unreadable literal: ", term, object)
                continue
            if type == 'xsd:string' and is_readable(term):
                if term in literal2id:
                    continue
                literal2id[term] = cnt
                literals.append(term)
                cnt += 1
    # Check correctness
    for ltr in literals:
        id = literal2id[ltr]
        assert literals[id] == ltr
    
    if not literals:
        hidden_size = getattr(model.config, "hidden_size", 0)
        empty_emb = np.empty((0, hidden_size), dtype=np.float32)
        if emb_file:
            np.save(emb_file, empty_emb)
            return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
        return {"id": literal2id, "emb": empty_emb}

    embedding_matrix = None
    for i in tqdm(range(0, len(literals), batch_size),
                  desc="        Computing embeddings"):
        batch_literals = literals[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = encode_text(model, batch_literals).cpu().numpy().astype(np.float32, copy=False)

        # Preallocate once and fill by slice.
        if embedding_matrix is None:
            shape = (len(literals), batch_embeddings.shape[1])
            # Use memory-mapped file to save memory for large embeddings.
            if emb_file:
                embedding_matrix = np.lib.format.open_memmap(emb_file, mode="w+", dtype=np.float32, shape=shape)
            else:
                embedding_matrix = np.empty(shape, dtype=np.float32)
        embedding_matrix[i:i + len(batch_literals)] = batch_embeddings
        del batch_embeddings

    if emb_file:
        embedding_matrix.flush()
        return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
    return {"id": literal2id, "emb": embedding_matrix}

# Memory-efficient approach. It's useful for large knowledge graphs where loading the entire graph may not be feasible due to memory constraints.
def iter_ttl_object_literals(path):
    """Yield literal objects from one-triple-per-line TTL files."""
    with open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("@prefix") or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) != 3 or not parts[2].endswith("."):
                continue
            obj = parts[2][:-1].strip()
            if obj.startswith('"'):
                yield obj

# Memory-efficient approach. It's useful for large knowledge graphs where loading the entire graph may not be feasible due to memory constraints.
def embedding_strings_from_ttl(path, batch_size=64, embedding_model=None, emb_file=None):
    """Compute embeddings by streaming TTL literals without loading the full KG."""
    model, _ = load_string_model(embedding_model)
    literal2id = {}
    literals = []

    for obj in tqdm(iter_ttl_object_literals(path), 
                    desc=f"        Scanning literals from {os.path.basename(path)}"):
        if not isLiteral(obj):
            continue
        term, _, _, type = splitLiteral(obj)
        if len(term) <= 1:
            continue
        if (not is_readable(term)) and type == 'xsd:string':
            continue
        if type == 'xsd:string' and is_readable(term):
            if term in literal2id:
                continue
            literal2id[term] = len(literals)
            literals.append(term)

    if not literals:
        hidden_size = getattr(model.config, "hidden_size", 0)
        empty_emb = np.empty((0, hidden_size), dtype=np.float32)
        if emb_file:
            np.save(emb_file, empty_emb)
            return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
        return {"id": literal2id, "emb": empty_emb}

    embedding_matrix = None
    for i in tqdm(range(0, len(literals), batch_size), 
                  desc="        Computing embeddings"):
        batch_literals = literals[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = encode_text(model, batch_literals).cpu().numpy().astype(np.float32, copy=False)

        if embedding_matrix is None:
            shape = (len(literals), batch_embeddings.shape[1])
            if emb_file:
                embedding_matrix = np.lib.format.open_memmap(emb_file, mode="w+", dtype=np.float32, shape=shape)
            else:
                embedding_matrix = np.empty(shape, dtype=np.float32)
        embedding_matrix[i:i + len(batch_literals)] = batch_embeddings
        del batch_embeddings

    if emb_file:
        embedding_matrix.flush()
        return {"id": literal2id, "emb_path": os.path.basename(emb_file)}
    return {"id": literal2id, "emb": embedding_matrix}
    

def numeric_normalization(term):
    term = term.strip('"')
    # Normalize the input string by removing all non-digit characters, except the sign
    # sign = term[0] if term.startswith('-') or term.startswith('+') else None
    normalized = re.sub(r'[^0-9]', '', term)
    return normalized

def jaccard_similarity(str1, str2):
    s1 = set(str1.split())
    s2 = set(str2.split())
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union

def compute_literal_embeddings(kb1, kb2, emb_path, batch_size=128, embedding_model=None):
    if not os.path.exists(emb_path):
        os.makedirs(emb_path)

    kb1_emb = embedding_strings(kb1, batch_size, embedding_model, emb_file=os.path.join(emb_path, "kb1.npy"))
    with open(os.path.join(emb_path, "kb1.pkl"), "wb") as f: # save the literal embeddings for kb1
        pickle.dump(kb1_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    del kb1_emb # free memory
    gc.collect()

    kb2_emb = embedding_strings(kb2, batch_size, embedding_model, emb_file=os.path.join(emb_path, "kb2.npy"))
    with open(os.path.join(emb_path, "kb2.pkl"), "wb") as f: # save the literal embeddings for kb2
        pickle.dump(kb2_emb, f, protocol=pickle.HIGHEST_PROTOCOL)

# Memory-efficient approach. It's useful for large knowledge graphs where loading the entire graph may not be feasible due to memory constraints.
def compute_literal_embeddings_streaming(kg1, kg2, emb_path, batch_size=128, embedding_model=None):
    if not os.path.exists(emb_path):
        os.makedirs(emb_path)

    kb1_emb = embedding_strings_from_ttl(kg1, batch_size, embedding_model, emb_file=os.path.join(emb_path, "kb1.npy"))
    with open(os.path.join(emb_path, "kb1.pkl"), "wb") as f: # save the literal embeddings for kb1
        pickle.dump(kb1_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    del kb1_emb # free memory
    gc.collect()

    kb2_emb = embedding_strings_from_ttl(kg2, batch_size, embedding_model, emb_file=os.path.join(emb_path, "kb2.npy"))
    with open(os.path.join(emb_path, "kb2.pkl"), "wb") as f: # save the literal embeddings for kb2
        pickle.dump(kb2_emb, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_params():
    parser = argparse.ArgumentParser(description="Pre-compute FLORA literal embeddings for two Turtle KGs.")
    parser.add_argument("kg1", help="Path to KG1 Turtle file")
    parser.add_argument("kg2", help="Path to KG2 Turtle file")
    parser.add_argument("emb_path", help="Output folder for kb1.pkl and kb2.pkl")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace model used to encode string literals. Default: Lihuchen/pearl_small. For multilingual embeddings, use sentence-transformers/LaBSE.")
    parser.add_argument("--batch_size", type=int, default=128, help="Literal encoding batch size")
    parser.add_argument("--streaming", action="store_true", help="Stream TTL files and extract literals without loading full Graphs (for memory-efficient processing).")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_params()

    if args.streaming:
        compute_literal_embeddings_streaming(
            args.kg1,
            args.kg2,
            args.emb_path,
            batch_size=args.batch_size,
            embedding_model=args.embedding_model,
        )
    else:
        Announce.doing("Loading Knowledge Bases")
        kb1 = utils.graphFromTurtleFile(args.kg1)
        kb2 = utils.graphFromTurtleFile(args.kg2)
        Announce.done()
        compute_literal_embeddings(kb1, kb2, args.emb_path, batch_size=args.batch_size, embedding_model=args.embedding_model)