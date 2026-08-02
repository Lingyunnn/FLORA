"""
FLORA literal matching initialization utilities.

This module can be run independently from FLORA's main loop. 
The main loop of FLORA can run on CPU, while literal matching benefits from 
GPU acceleration when FAISS GPU support is available. Keeping this file executable
therefore makes it convenient to precompute literal matching scores separately
on a GPU machine, save them, and reuse them later in the CPU-only FLORA loop.
"""

import re
import os
import math
import time
import logging
import argparse
import unicodedata
from collections import defaultdict
import pickle
import numpy as np
from urllib.parse import unquote
import faiss


#################################################################
#            Literal parsing and datatype helpers               #
#################################################################

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

DATE_DATATYPES = {'xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:dateTime', 'xsd:gMonthDay'}


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

def numeric_normalization(term):
    term = term.strip('"')
    # Normalize the input string by removing all non-digit characters, except the sign
    # sign = term[0] if term.startswith('-') or term.startswith('+') else None
    normalized = re.sub(r'[^0-9]', '', term)
    return normalized

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
    
    matchNumber=re.fullmatch(numberRegex, literal)
    # Check if the string is a number type, e.g., "818/762-1221"
    if matchNumber:
        Value = numeric_normalization(literal)
        if len(Value) > 0:
            return (literal, 'normalized_'+Value, lang, datatype)
    # Strings: lowecasing, order-agnostic, decode unicode
    # Pre-processing for string literals
    de_literal = decode_unicode(literal)
    return (de_literal, None, lang, 'xsd:string')


#################################################################
#          Literal normalization and candidate filters          #
#################################################################

def reorder_string_with_brackets(input):
    bracket_content = re.findall(r'\(.*?\)', input)
    content_without_brackets = re.sub(r'\(.*?\)', '', input).split()
    return (' '.join(set(content_without_brackets)) + ' ' + ' '.join(bracket_content)).strip()

def is_punctuation_only_literal(txt):
    """Check if a string literal consists only of punctuation characters."""
    txt = decode_unicode(txt).strip()
    return bool(txt) and not any(char.isalnum() for char in txt)

def is_human_readable(txt):
    if not txt or len(txt) == 0:
        return False
    # not a web link
    if txt.startswith('http'):
        return False
    non_alpha_ratio = sum(not c.isalpha() for c in txt) / len(txt) 
    return non_alpha_ratio < 0.5

def is_faiss_literal_candidate(txt):
    """Filter string literals before FAISS search."""
    if not txt:
        return False
    txt = decode_unicode(txt).strip()
    if not txt:
        return False
    return is_human_readable(txt)

def normalize_string_literal(value):
    value = decode_unicode(value) # decode unicode characters
    value = unquote(value) # decode URL-encoded characters
    value = value.strip() # remove spaces
    value = value.lower() # convert to lowercase

    # Normalize common separators.
    value = value.replace('_', ' ') # replace underscores with spaces
    value = re.sub(r'\s+', ' ', value) # replace multiple spaces with a single space
    return value.strip()

def is_latin_alpha(char):
    if not char.isalpha():
        return False
    try:
        return 'LATIN' in unicodedata.name(char)
    except ValueError:
        return False

def is_english_string_literal(literal, lang=None, min_latin_ratio=0.8):
    """Check if a string literal is likely to be English based on its language tag and the ratio of Latin letters."""
    if lang is not None:
        lang = lang.lower()
        if lang != 'en' and not lang.startswith('en-'):
            return False

    letters = [char for char in decode_unicode(literal) if char.isalpha()]
    if not letters:
        return True

    latin_letters = sum(1 for char in letters if is_latin_alpha(char))
    return latin_letters / len(letters) >= min_latin_ratio


#################################################################
#                FAISS embedding index helpers                  #
#################################################################

def normalize_embedding_matrix(embedding_matrix):
    """ Normalize the embedding matrix to have unit length vectors. """
    matrix = np.asarray(embedding_matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return matrix / safe_norms

def create_faiss_inner_product_index(dim, index_type='flat', hnsw_m=32, hnsw_ef_search=64, hnsw_ef_construction=200):
    """Create a FAISS IP index.
    ``flat`` uses exact-search and uses FAISS GPU when available. 
    ``hnsw`` uses approximate CPU search, which is usually much faster for large literal candidate sets.
    """
    if index_type not in {'flat', 'hnsw'}:
        raise ValueError("Unsupported FAISS literal index type: %s" % index_type)

    if index_type == 'hnsw':
        try:
            index = faiss.IndexHNSWFlat(dim, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            index = faiss.IndexHNSWFlat(dim, int(hnsw_m))
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = int(hnsw_ef_construction)
        index.hnsw.efSearch = int(hnsw_ef_search)
        logging.info(
            "FAISS literal search using HNSW CPU index | M=%s | efSearch=%s | efConstruction=%s",
            hnsw_m, hnsw_ef_search, hnsw_ef_construction,
        )
        print(
            "   FAISS literal search using HNSW CPU index "
            "(M=%s, efSearch=%s, efConstruction=%s)." % (
                hnsw_m, hnsw_ef_search, hnsw_ef_construction,
            ),
            flush=True,
        )
        return index, None

    index = faiss.IndexFlatIP(dim)

    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        message = "This FAISS build has no GPU support. Using CPU."
        logging.warning(message)
        print("   Warning: " + message, flush=True)
        return index, None

    try:
        gpu_count = faiss.get_num_gpus()
    except Exception:
        gpu_count = 0
    if gpu_count <= 0:
        logging.info("FAISS literal search using CPU.")
        print("   FAISS literal search using CPU.", flush=True)
        return index, None

    gpu_res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    logging.info("FAISS literal search using GPU cuda:0.")
    print("   FAISS literal search using GPU cuda:0.", flush=True)
    return index, gpu_res


#################################################################
#                    IDF weighting helpers                      #
#################################################################

def _iter_literal_facts(kb):
    if (hasattr(kb, 'num_entities')
        and hasattr(kb, 'triplesWithSubjectIds')
        and hasattr(kb, 'predicate_for_id')
        and hasattr(kb, 'entity_for_id')
        and hasattr(kb, 'is_literal_id')):
        for subject_id in range(kb.num_entities()):
            for _, predicate_id, object_id in kb.triplesWithSubjectIds(subject_id):
                if kb.is_literal_id(object_id):
                    yield subject_id, kb.predicate_for_id(predicate_id), kb.entity_for_id(object_id)
        return

    for subject, predicate, obj in kb:
        if isLiteral(obj):
            yield subject, predicate, obj


def compute_literal_idf_weights(kb):
    """
    Return {literal_object: weight} using subject-level IDF.
    The weight is computed as log((N + 1) / df) / log(N + 1), 
    where N is the total number of subjects and df is the number of subjects that have the literal object.
    """
    literal_subjects = defaultdict(set)
    all_subjects = set()

    for subject, _, obj in _iter_literal_facts(kb):
        all_subjects.add(subject)
        literal_subjects[obj].add(subject)

    subject_count = len(all_subjects)
    max_idf = math.log(subject_count + 1)
    if subject_count <= 0 or max_idf <= 0:
        return {literal_obj: 1.0 for literal_obj in literal_subjects}

    weights = {}
    for literal_obj, subjects in literal_subjects.items():
        df = max(1, len(subjects))
        normalized_idf = math.log((subject_count + 1) / df) / max_idf
        weights[literal_obj] = max(0.0, min(1.0, normalized_idf))
    return weights


def update_literal_score(scores, literal1, literal2, score, weights1=None, weights2=None, min_score=None):
    if weights1 is not None and weights2 is not None:
        score *= math.sqrt(weights1.get(literal1, 1.0) * weights2.get(literal2, 1.0))
    if min_score is not None and score < min_score:
        return False
    row = scores.setdefault(literal1, {})
    if score > row.get(literal2, 0):
        row[literal2] = score
        return True
    return False


#################################################################
#                Bucket-level literal comparison                #
#################################################################

# Get buckets based on types
def getLiteralBuckets(kb, literal_english_filter=False):
    quantityBucket = defaultdict(list) # e.g., integers, floats
    digitBucket = defaultdict(list) # e.g., IDs, code versions, identifier
    strBucket = defaultdict(list) # e.g., strings
    dateBucket = defaultdict(list) # e.g., dates
    if hasattr(kb, 'iter_literal_object_ids') and hasattr(kb, 'entity_for_id'): # for compact graph
        literal_objects = (kb.entity_for_id(object_id) for object_id in kb.iter_literal_object_ids())
    else:
        literal_objects = (object1 for object1 in kb.objects() if isLiteral(object1))

    for object1 in literal_objects:
        literal, numValue, lang, datatype = splitLiteral(object1)
        if is_punctuation_only_literal(literal):
            continue
        # dates handling
        if datatype in DATE_DATATYPES:
            dateBucket[literal].append(object1)
            continue
        # numeric handling
        if numValue is not None:
            if isinstance(numValue, (int, float)):
                if datatype is not None: # strict for the specified the datatype
                    quantityBucket[numValue].append(object1)
                    continue # TBD: may delete
                    # numberBucket[numValue].append(object1)
                # e.g., "0" vs "0"^^xsd:decimal is different
                # serve it as a string, e.g., code version 1.0, 01
                digitBucket[literal].append(object1)
                continue
            # xsd:normalizedString for phone numbers, post codes, etc.
            if len(numValue) > 0:
                strBucket[numValue].append(object1)
            continue
        # string handling
        if literal_english_filter and not is_english_string_literal(literal, lang):
            continue
        # if is_human_readable(literal):
        strBucket[literal].append(object1)
        normalized_literal = normalize_string_literal(literal)
        if normalized_literal != literal:
            strBucket[normalized_literal].append(object1)
            
    return quantityBucket, digitBucket, strBucket, dateBucket

# Compare literals in same type buckets
def compareLiterals(sameAsScores, bucket1, bucket2, datatype=None, weights1=None, weights2=None, min_score=None):
    if datatype is None:
        raise ValueError('Datatype must be specified')
    # strings, dates
    if datatype == 'string':
        pass
    elif datatype == 'quantity':
        for key in bucket1:
            for key2 in bucket2:
                if math.isclose(key, key2):
                    for object1 in bucket1[key]:
                        for object2 in bucket2[key2]:
                            update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)
    elif datatype == 'date':
        for key1 in bucket1:
            if key1 in bucket2:
                for object1 in bucket1[key1]:
                    for object2 in bucket2[key1]:
                        update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)
                continue
            last_dash = key1.rfind('-')
            while last_dash != -1:
                key1_ = key1[:last_dash]
                if key1_ in bucket2:
                    for object1 in bucket1[key1]:
                        for object2 in bucket2[key1_]:
                            update_literal_score(sameAsScores, object1, object2, (key1_.count('-')+1) / 3, weights1, weights2, min_score)
                    break
                last_dash = key1.rfind('-', 0, last_dash)       
    else: # digits, e.g., IDs, code versions, identifier
        for key in bucket1:
            if key in bucket2 and len(key.strip('"')) > 0:
                for object1 in bucket1[key]:
                    for object2 in bucket2[key]:
                        update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)

def compareLiterals_identity(sameAsScores, bucket1, bucket2, weights1=None, weights2=None, min_score=None):
    for key in bucket1:
        if key in bucket2:
            for object1 in bucket1[key]:
                for object2 in bucket2[key]:
                    update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)


def has_perfect_literal_match(literal, *score_maps):
    for score_map in score_maps:
        if not score_map:
            continue
        scores = score_map.get(literal)
        if scores and max(scores.values(), default=0) >= 1.0:
            return True
    return False


def all_literals_have_perfect_match(literals, *score_maps):
    return all(has_perfect_literal_match(literal, *score_maps) for literal in literals)


def _bucket_stats(bucket):
    return len(bucket), sum(len(objects) for objects in bucket.values())


#################################################################
#               Main literal matching function                  #
#################################################################

def load_emb(path):
    with open(path, 'rb') as file:
        emb = pickle.load(file)
    literal2id = emb['id']
    if 'emb_path' in emb:
        embedding_matrix = np.load(os.path.join(os.path.dirname(path), emb['emb_path']), mmap_mode='r')
    else:
        embedding_matrix = emb['emb']
    return literal2id, embedding_matrix

def mapLiterals(
    kb1,
    kb2,
    path_emb,
    sameAsScore,
    literal_identity_only=False,
    threshold=0.5,
    chunk_size=32768,
    top_k=1,
    literal_english_filter=False,
    literal_idf=False,
    literal_faiss_index='flat',
    literal_hnsw_m=32,
    literal_hnsw_ef_search=64,
    literal_hnsw_ef_construction=200,
):
    total_start = time.time()
    mapScores = {}
    logging.info(
        "Literal matching started | identity_only=%s | threshold=%s | chunk_size=%s | top_k=%s | "
        "english_filter=%s | faiss_index=%s | hnsw_m=%s | hnsw_ef_search=%s | "
        "hnsw_ef_construction=%s | embedding=%s",
        literal_identity_only, threshold, chunk_size, top_k,
        literal_english_filter, literal_faiss_index, literal_hnsw_m, literal_hnsw_ef_search,
        literal_hnsw_ef_construction, path_emb,
    )
    stage_start = time.time()
    quantityBucket1, digitBucket1, strBucket1, dateBucket1 = getLiteralBuckets(kb1, literal_english_filter=literal_english_filter)
    quantityBucket2, digitBucket2, strBucket2, dateBucket2 = getLiteralBuckets(kb2, literal_english_filter=literal_english_filter)
    logging.info(
        "Literal buckets built | elapsed_min=%.3f | "
        "kb1 quantity=%s/%s digit=%s/%s string=%s/%s date=%s/%s | "
        "kb2 quantity=%s/%s digit=%s/%s string=%s/%s date=%s/%s",
        (time.time() - stage_start) / 60,
        *_bucket_stats(quantityBucket1), *_bucket_stats(digitBucket1), *_bucket_stats(strBucket1), *_bucket_stats(dateBucket1),
        *_bucket_stats(quantityBucket2), *_bucket_stats(digitBucket2), *_bucket_stats(strBucket2), *_bucket_stats(dateBucket2),
    )
    weights1 = weights2 = None
    if literal_idf:
        stage_start = time.time()
        weights1 = compute_literal_idf_weights(kb1)
        weights2 = compute_literal_idf_weights(kb2)
        logging.info(
            "Literal IDF weights computed | elapsed_min=%.3f | kb1=%s | kb2=%s",
            (time.time() - stage_start) / 60, len(weights1), len(weights2),
)
    weighted_min_score = threshold if literal_idf else None
    if not literal_identity_only:
        # Dates
        stage_start = time.time()
        compareLiterals(mapScores, dateBucket1, dateBucket2, 'date', weights1, weights2, weighted_min_score)
        # Compare numbers
        compareLiterals(mapScores, quantityBucket1, quantityBucket2, 'quantity', weights1, weights2, weighted_min_score)
        compareLiterals(mapScores, digitBucket1, digitBucket2, 'digit', weights1, weights2, weighted_min_score)
        logging.info(
            "Literal date/number matching done | elapsed_min=%.3f | sources=%s | pairs=%s",
            (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
        )

        # Compare strings
        stage_start = time.time()
        compareLiterals_identity(mapScores, strBucket1, strBucket2, weights1, weights2, weighted_min_score) # first get exact match
        logging.info(
            "Literal exact string matching done | elapsed_min=%.3f | sources=%s | pairs=%s",
            (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
        )

        emb1_path = os.path.join(path_emb, 'kb1.pkl')
        emb2_path = os.path.join(path_emb, 'kb2.pkl')

        if os.path.exists(emb1_path) and os.path.exists(emb2_path):
            stage_start = time.time()
            literal2id_kb1, embedding_matrix_kb1 = load_emb(emb1_path)
            literal2id_kb2, embedding_matrix_kb2 = load_emb(emb2_path)
            logging.info(
                "Literal embeddings loaded | elapsed_min=%.3f | kb1_literals=%s | kb2_literals=%s",
                (time.time() - stage_start) / 60, len(literal2id_kb1), len(literal2id_kb2),
            )

            candidate_keys1 = []
            candidate_ids1 = []

            stage_start = time.time()
            for key1 in strBucket1:
                # Empty string
                if len(key1.strip('"')) == 0:
                    continue

                # Already handled by exact string matching.
                if key1 in strBucket2:
                    continue

                # Skip embedding search when this bucket key only contains source literals that already have a perfect match.
                if all_literals_have_perfect_match(strBucket1[key1], mapScores, sameAsScore):
                    continue

                # Keep only human-readable strings.
                if not is_faiss_literal_candidate(key1):
                    continue

                # Must have a precomputed embedding.
                if key1 not in literal2id_kb1:
                    continue

                candidate_keys1.append(key1)
                candidate_ids1.append(literal2id_kb1[key1])

            candidate_keys2 = []
            candidate_ids2 = []

            for key2 in strBucket2:
                if len(key2.strip('"')) == 0:
                    continue

                # Keep only human-readable strings.
                if not is_faiss_literal_candidate(key2):
                    continue

                if key2 not in literal2id_kb2:
                    continue

                candidate_keys2.append(key2)
                candidate_ids2.append(literal2id_kb2[key2])
            logging.info(
                "Literal FAISS candidates selected | elapsed_min=%.3f | kb1=%s | kb2=%s",
                (time.time() - stage_start) / 60, len(candidate_ids1), len(candidate_ids2),
            )

            if len(candidate_ids1) > 0 and len(candidate_ids2) > 0:
                candidate_ids1 = np.asarray(candidate_ids1, dtype=np.int64)
                candidate_ids2 = np.asarray(candidate_ids2, dtype=np.int64)

                # Normalize the embedding matrices.
                stage_start = time.time()
                emb1_valid = np.ascontiguousarray(normalize_embedding_matrix(embedding_matrix_kb1[candidate_ids1]), dtype=np.float32)
                emb2_valid = np.ascontiguousarray(normalize_embedding_matrix(embedding_matrix_kb2[candidate_ids2]), dtype=np.float32)
                logging.info(
                    "Literal FAISS matrices prepared | elapsed_min=%.3f | emb1_shape=%s | emb2_shape=%s",
                    (time.time() - stage_start) / 60, emb1_valid.shape, emb2_valid.shape,
                )

                # FAISS inner-product search
                # Since embeddings are L2-normalized, inner product = cosine similarity.
                dim = emb2_valid.shape[1]
                # Keep gpu_res alive for the lifetime of the GPU FAISS index.
                index, gpu_res = create_faiss_inner_product_index(
                    dim,
                    index_type=literal_faiss_index,
                    hnsw_m=literal_hnsw_m,
                    hnsw_ef_search=literal_hnsw_ef_search,
                    hnsw_ef_construction=literal_hnsw_ef_construction,
                )
                stage_start = time.time()
                index.add(emb2_valid)
                logging.info(
                    "Literal FAISS index populated | elapsed_min=%.3f | vectors=%s | dim=%s",
                    (time.time() - stage_start) / 60, len(candidate_keys2), dim,
                )

                # Query in batches to avoid allocating large arrays
                query_batch_size = chunk_size
                search_k = min(top_k, len(candidate_keys2))

                stage_start = time.time()
                for start1 in range(0, len(candidate_keys1), query_batch_size):
                    end1 = min(start1 + query_batch_size, len(candidate_keys1))

                    block_emb1 = np.ascontiguousarray(emb1_valid[start1:end1], dtype=np.float32)

                    local_best_scores, local_best_indices = index.search(block_emb1, search_k)

                    for row_idx, key1 in enumerate(candidate_keys1[start1:end1]):
                        for best_score, idx2 in zip(local_best_scores[row_idx], local_best_indices[row_idx]):
                            if idx2 < 0:
                                continue

                            best_score = float(best_score)
                            if best_score > 1.0:
                                best_score = 1.0
                            if not literal_idf and best_score < threshold:
                                continue

                            literal2 = candidate_keys2[int(idx2)]

                            for object1 in strBucket1[key1]:
                                if has_perfect_literal_match(object1, mapScores, sameAsScore):
                                    continue

                                if object1 not in mapScores:
                                    mapScores[object1] = {}

                                for object2 in strBucket2[literal2]:
                                    update_literal_score(mapScores, object1, object2, best_score, weights1, weights2, weighted_min_score)
                logging.info(
                    "Literal FAISS search done | elapsed_min=%.3f | sources=%s | pairs=%s",
                    (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
                )
        else: # no embeddings files available
            print("   Warning: No valid embedding files for literals, Using exact match only.")
    else:
        # Identity mapping only
        stage_start = time.time()
        compareLiterals_identity(mapScores, strBucket1, strBucket2, weights1, weights2, weighted_min_score)
        compareLiterals_identity(mapScores, dateBucket1, dateBucket2, weights1, weights2, weighted_min_score)
        compareLiterals_identity(mapScores, quantityBucket1, quantityBucket2, weights1, weights2, weighted_min_score)
        compareLiterals_identity(mapScores, digitBucket1, digitBucket2, weights1, weights2, weighted_min_score)
        logging.info(
            "Literal identity matching done | elapsed_min=%.3f | sources=%s | pairs=%s",
            (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
        )
    
    # Load to sameAsScore
    stage_start = time.time()
    for literal1 in mapScores:
        if hasattr(kb1, 'entity_id'):
            if kb1.entity_id(literal1) is None:
                continue
        elif not kb1.has_subject(literal1):
            continue
        # check empty mapping
        if literal1 not in sameAsScore:
            sameAsScore[literal1] = {}
        for literal2, score in mapScores[literal1].items():
            if score > sameAsScore[literal1].get(literal2, 0):
                sameAsScore[literal1][literal2] = score
    logging.info(
        "Literal matching loaded into sameAsScore | elapsed_min=%.3f | total_elapsed_min=%.3f | sources=%s | pairs=%s",
        (time.time() - stage_start) / 60, (time.time() - total_start) / 60, len(sameAsScore), sum(len(values) for values in sameAsScore.values()),
    )


#################################################################
#  Streaming TTL loaders for memory-efficient literal matching  #
#################################################################

class StreamingLiteralGraph:
    """A Graph-like adapter used for memory-efficient literal matching.
    It keeps only literal objects, and optionally literal -> subjects mappings for IDF calculation,
    so mapLiterals can run without loading the full RDF graph into memory.
    """

    def __init__(self, literal_objects, literal_subjects=None):
        self.literal_objects = literal_objects # literal objects
        self.literal_subjects = literal_subjects # literal -> subjects mappings

    def objects(self, subject=None, predicate=None):
        # mapLiterals only needs the full literal-object set.
        if subject is None and predicate is None:
            return self.literal_objects
        return set()

    def has_subject(self, subject):
        # used to check if a literal exists
        return subject in self.literal_objects

    def iter_subjects(self, predicate=None, object=None, limit=None):
        # Subject iteration is available only when collect_subjects=True.
        if object is None or self.literal_subjects is None:
            return
        subjects = self.literal_subjects.get(object, ())
        for index, subject in enumerate(subjects):
            if limit is not None and index >= limit:
                break
            yield subject

    def __iter__(self):
        # compute_literal_idf_weights expects (subject, predicate, object).
        # Predicate values are not needed for IDF, so yields None.
        if self.literal_subjects is None:
            return
        for obj, subjects in self.literal_subjects.items():
            for subject in subjects:
                yield subject, None, obj


def iter_ttl_literal_facts(path):
    """Yield (subject, literal_object) from one-triple-per-line TTL files."""
    with open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("@prefix") or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) != 3 or not parts[2].endswith("."):
                continue
            obj = parts[2][:-1].strip()
            if obj.startswith('"') and isLiteral(obj):
                yield parts[0], obj


def load_streaming_literal_graph(path, collect_subjects=False):
    """Load a TTL file and return a StreamingLiteralGraph containing only literal objects."""
    literal_objects = set()
    literal_subjects = defaultdict(set) if collect_subjects else None
    literal_fact_count = 0
    start_time = time.time()

    for subject, obj in iter_ttl_literal_facts(path):
        literal_fact_count += 1
        literal_objects.add(obj)
        if literal_subjects is not None:
            literal_subjects[obj].add(subject)

    logging.info(
        "Streaming literal scan done | file=%s | parsed_literal_facts=%s | unique_literals=%s | elapsed_min=%.3f",
        path, literal_fact_count, len(literal_objects), (time.time() - start_time) / 60,
    )
    return StreamingLiteralGraph(literal_objects, literal_subjects)


#################################################################
#     Command-line interface for standalone precomputation      #
#################################################################

def get_params():
    parser = argparse.ArgumentParser(description="Compute FLORA literal matching scores.")
    parser.add_argument("--kg1", required=True, help="KG1 Turtle file")
    parser.add_argument("--kg2", required=True, help="KG2 Turtle file")
    parser.add_argument("--embedding", required=True, help="Folder containing kb1.pkl/kb2.pkl literal embeddings")
    parser.add_argument("--output", required=True, help="Output pickle path for literal sameAs scores")
    parser.add_argument("--init", type=float, default=0.7, help="Initial literal similarity threshold")
    parser.add_argument("--string_identity", action="store_true", help="Use exact literal identity only")
    parser.add_argument("--chunk_size", type=int, default=32768, help="FAISS query batch size")
    parser.add_argument("--top_k", type=int, default=1, help="Number of target literals retrieved per source literal")
    parser.add_argument("--literal_english_filter", action="store_true")
    parser.add_argument("--literal_idf", action="store_true")
    parser.add_argument("--literal_faiss_index", choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--literal_hnsw_m", type=int, default=32)
    parser.add_argument("--literal_hnsw_ef_search", type=int, default=64)
    parser.add_argument("--literal_hnsw_ef_construction", type=int, default=200)
    return parser.parse_args()

def main():
    args = get_params()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    sameAsScore = {}
    needs_subjects = args.literal_idf
    kb1 = load_streaming_literal_graph(args.kg1, collect_subjects=needs_subjects)
    kb2 = load_streaming_literal_graph(args.kg2, collect_subjects=needs_subjects)
    mapLiterals(
        kb1,
        kb2,
        args.embedding,
        sameAsScore,
        literal_identity_only=args.string_identity,
        threshold=args.init,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        literal_english_filter=args.literal_english_filter,
        literal_idf=args.literal_idf,
        literal_faiss_index=args.literal_faiss_index,
        literal_hnsw_m=args.literal_hnsw_m,
        literal_hnsw_ef_search=args.literal_hnsw_ef_search,
        literal_hnsw_ef_construction=args.literal_hnsw_ef_construction,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as output_file:
        pickle.dump(sameAsScore, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(
        "Saved literal matching scores | output=%s | sources=%s | pairs=%s",
        args.output, len(sameAsScore), sum(len(values) for values in sameAsScore.values()),
    )

if __name__ == "__main__":
    main()