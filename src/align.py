from itertools import combinations
from collections import Counter, OrderedDict
import multiprocessing as mp
import os
import time
import logging
import shutil
from queue import Empty, Full
import log
import utils
import sides


# Constants for accessing the components of a triple
SUBJ=0
PRED=1
OBJ=2

ENTITY_TASK_CHUNK_SIZE = 100 # number of entities to feed workers in each chunk
SUBRELATION_FACT_CHUNK_SIZE = 5000 # number of subject-relation pairs to feed workers in each chunk
SUBRELATION_SUBJECT_OBJECT_CACHE_DEGREE_THRESHOLD = 512 # minimum target-subject degree to build a temporary object->predicate cache
SUBRELATION_SUBJECT_OBJECT_CACHE_MAX_SUBJECTS = 1024 # maximum cached target subjects per subrelation worker
MATCH_RESULT_FLUSH_PAIRS = 50000 # number of match results to accumulate before flushing to the output
RESULT_DRAIN_INTERVAL = 0.1 # time interval (in seconds) to drain results from the output queue
PROGRESS_LOG_INTERVAL = 300.0 # time interval (in seconds) to log progress information
PROGRESS_CHECK_FACT_INTERVAL = 100000 # number of facts to process before checking progress
MAX_ENT_SCORE_CACHE_SIZE = 262144 # maximum size of the entity max-score cache to avoid unbounded memory growth


#################################################################
#                       General Utilities                       #
#################################################################

def fast_hmean(values):
    """Harmonic mean for short positive score sequences."""
    count = 0
    reciprocal_sum = 0.0
    for value in values:
        if value <= 0:
            return 0.0
        reciprocal_sum += 1.0 / value
        count += 1
    return count / reciprocal_sum if count and reciprocal_sum else 0.0

def encode_pattern(seq):
    """
    Encode a sequence by first-occurrence pattern.

    Example:
    ("a", "b", "a") -> (0, 1, 0)
    ("x", "y", "x") -> (0, 1, 0)
    """
    value2code = {}
    pattern = []
    next_code = 0
    for value in seq:
        if value not in value2code:
            value2code[value] = next_code
            next_code += 1
        pattern.append(value2code[value])
    return tuple(pattern)


#################################################################
#                 Multiprocessing Queue Helpers                 #
#################################################################

def iter_entity_chunks(graph, chunk_size=ENTITY_TASK_CHUNK_SIZE, include_literals=False, only_literals=False):
    """
    Yield graph entities in bounded chunks to avoid materializing the full task list in memory before workers start consuming it.

    Parameters
    ----------
    graph : Graph
        Graph whose entities should be streamed into chunks.
    chunk_size : int, optional
        Maximum number of entities per chunk.
    include_literals : bool, optional
        When False, filter out literal entities.
    only_literals : bool, optional
        When True, emit only literal entities. Bootstrap uses this to propagate only literals.
    """
    chunk = []
    subject_iter = graph.iter_view_subjects()

    for subject in subject_iter:
        if isinstance(subject, int) and hasattr(graph, 'is_literal_id'):
            is_literal = graph.is_literal_id(subject)
        else:
            subject_value = graph.entity_for_id(subject) if isinstance(subject, int) else subject
            is_literal = bool(utils.isLiteral(subject_value))
        if only_literals and not is_literal:
            continue
        if not include_literals and is_literal:
            continue
        chunk.append(subject)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def iter_source_entity_chunks(graph, source_entities, chunk_size=ENTITY_TASK_CHUNK_SIZE,
                              include_literals=False, only_literals=False):
    """
    Yield bounded chunks from an explicit source set.
    """
    chunk = []
    graph_side = sides.graph_side(graph, sides.KB1)
    for subject in source_entities:
        if sides.is_side_key(subject):
            if subject[0] != graph_side:
                continue
            if hasattr(graph, 'is_literal_id'):
                is_literal = graph.is_literal_id(subject[1])
                subject_value = None
            else:
                try:
                    subject_value = graph.entity_for_id(subject[1])
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
        elif isinstance(subject, int) and hasattr(graph, 'entity_for_id'):
            if hasattr(graph, 'is_literal_id'):
                is_literal = graph.is_literal_id(subject)
                subject_value = None
            else:
                try:
                    subject_value = graph.entity_for_id(subject)
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
        else:
            subject_value = subject
        if subject_value is not None:
            is_literal = bool(utils.isLiteral(subject_value))
        if only_literals and not is_literal:
            continue
        if not include_literals and is_literal:
            continue
        chunk.append(subject)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def drain_queue_items(queue_obj, handler):
    """
    Drain all currently available items from a multiprocessing queue.
    """
    drained = 0
    while True:
        try:
            item = queue_obj.get_nowait()
        except Empty:
            break
        handler(item)
        drained += 1
    return drained

def drain_side_queue_items(side_queues):
    """
    Drain optional side queues such as worker profile metadata.
    """
    drained_counts = {}
    for label, queue_obj, handler in side_queues or []:
        drained_counts[label] = drained_counts.get(label, 0) + drain_queue_items(queue_obj, handler)
    return drained_counts

def _put_task_with_drain(task_queue, item, result_queue=None, merge_fn=None,
                         wait_interval=RESULT_DRAIN_INTERVAL,
                         progress_logger=None, progress_fields=None,
                         memory_tracker=None, side_queues=None,
                         worker_tasks=None, stage_label=None):
    """
    Try to enqueue one task without blocking indefinitely. If the task queue is
    full, drain worker results before retrying so the producer cannot deadlock
    while workers are blocked on the result queue.
    """
    drained_results = 0
    wait_loops = 0
    while True:
        try:
            task_queue.put_nowait(item)
            return drained_results, wait_loops
        except Full: # if the task queue is full, drain results and retry
            wait_loops += 1
            if result_queue is not None and merge_fn is not None:
                drained_results += drain_queue_items(result_queue, merge_fn)
            side_drained_counts = drain_side_queue_items(side_queues)
            if worker_tasks is not None and not any(task.is_alive() for task in worker_tasks):
                for task in worker_tasks:
                    task.join(timeout=0)
                exitcodes = [task.exitcode for task in worker_tasks]
                logging.error(
                    "%s task queue is full but no workers are alive | "
                    "exitcodes=%s | pending_item_is_sentinel=%s | wait_loops=%s | "
                    "drained_results=%s",
                    stage_label or "Worker stage",
                    exitcodes, item is None, wait_loops,
                    drained_results,
                )
                raise RuntimeError(
                    "%s task queue is full but no workers are alive; exitcodes=%s"
                    % (stage_label or "Worker stage", exitcodes)
                )
            if progress_logger is not None:
                fields = dict(progress_fields or {})
                fields['queue_wait_loops'] = wait_loops
                fields['merged_result_chunks'] = fields.get('merged_result_chunks', 0) + drained_results
                for label, count in side_drained_counts.items():
                    fields[f"drained_{label}"] = count
                if memory_tracker is not None:
                    fields.update(memory_tracker.progress_fields())
                progress_logger(**fields)
            time.sleep(wait_interval)

def feed_entity_chunks(task_queue, graph, num_workers, chunk_size=ENTITY_TASK_CHUNK_SIZE,
                       result_queue=None, merge_fn=None, include_literals=False,
                       stage_label=None, progress_interval=PROGRESS_LOG_INTERVAL,
                       memory_tracker=None, side_queues=None, only_literals=False,
                       source_entities=None, worker_tasks=None):
    """
    Feed entity chunks lazily so only a bounded number of tasks reside in the
    queue at a time. Optionally drain partial worker results while feeding.
    """
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)
    fed_chunks = 0  # number of fed chunks
    fed_entities = 0 # number of fed entities
    sentinels_sent = 0 # number of ended workers
    merged_result_chunks = 0 # number of merged chunks

    if source_entities is None:
        chunk_iter = iter_entity_chunks(
            graph,
            chunk_size=chunk_size,
            include_literals=include_literals,
            only_literals=only_literals,
        )
    else:
        chunk_iter = iter_source_entity_chunks(
            graph,
            source_entities,
            chunk_size=chunk_size,
            include_literals=include_literals,
            only_literals=only_literals,
        )

    for entity_chunk in chunk_iter:
        drained_results, wait_loops = _put_task_with_drain(
            task_queue,
            entity_chunk,
            result_queue=result_queue,
            merge_fn=merge_fn,
            progress_logger=progress_logger,
            progress_fields={
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
            },
            memory_tracker=memory_tracker,
            side_queues=side_queues,
            worker_tasks=worker_tasks,
            stage_label=stage_label,
        )
        merged_result_chunks += drained_results
        fed_chunks += 1
        fed_entities += len(entity_chunk)
        if memory_tracker is not None:
            memory_tracker.sample()
        if result_queue is not None and merge_fn is not None:
            merged_result_chunks += drain_queue_items(result_queue, merge_fn)
        side_drained_counts = drain_side_queue_items(side_queues)
        if progress_logger is not None:
            fields = {
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
                'queue_wait_loops': wait_loops,
            }
            for label, count in side_drained_counts.items():
                fields[f"drained_{label}"] = count
            if memory_tracker is not None:
                fields.update(memory_tracker.progress_fields())
            progress_logger(**fields)
    for _ in range(num_workers):
        drained_results, wait_loops = _put_task_with_drain(
            task_queue,
            None,
            result_queue=result_queue,
            merge_fn=merge_fn,
            progress_logger=progress_logger,
            progress_fields={
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
            },
            memory_tracker=memory_tracker,
            side_queues=side_queues,
            worker_tasks=worker_tasks,
            stage_label=stage_label,
        )
        merged_result_chunks += drained_results
        sentinels_sent += 1
        if memory_tracker is not None:
            memory_tracker.sample()
        if result_queue is not None and merge_fn is not None:
            merged_result_chunks += drain_queue_items(result_queue, merge_fn)
        side_drained_counts = drain_side_queue_items(side_queues)
        if progress_logger is not None:
            fields = {
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
                'queue_wait_loops': wait_loops,
            }
            for label, count in side_drained_counts.items():
                fields[f"drained_{label}"] = count
            if memory_tracker is not None:
                fields.update(memory_tracker.progress_fields())
            progress_logger(**fields)

    return {
        'fed_chunks': fed_chunks,
        'fed_entities': fed_entities,
        'sentinels_sent': sentinels_sent,
        'merged_result_chunks': merged_result_chunks,
    }

def wait_for_workers_and_drain(tasks, result_queue, merge_fn, poll_interval=RESULT_DRAIN_INTERVAL,
                               stage_label=None, progress_interval=PROGRESS_LOG_INTERVAL,
                               memory_tracker=None, side_queues=None):
    """
    Merge partial worker results while processes are still running so the
    result queue does not accumulate the full iteration output. Side queues are
    drained too, otherwise workers can block while publishing final metadata
    that the parent only reads after all workers exit.
    """
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)
    merged_result_chunks = 0
    while True:
        alive_workers = sum(task.is_alive() for task in tasks)
        merged_result_chunks += drain_queue_items(result_queue, merge_fn)
        side_drained_counts = drain_side_queue_items(side_queues)
        if memory_tracker is not None:
            memory_tracker.sample()
        if alive_workers == 0:
            break
        if progress_logger is not None:
            fields = {
                'alive_workers': alive_workers,
                'finished_workers': len(tasks) - alive_workers,
                'merged_result_chunks': merged_result_chunks,
            }
            for label, count in side_drained_counts.items():
                fields[f"drained_{label}"] = count
            if memory_tracker is not None:
                fields.update(memory_tracker.progress_fields())
            progress_logger(**fields)
        time.sleep(poll_interval)
    for task in tasks:
        task.join()
    merged_result_chunks += drain_queue_items(result_queue, merge_fn)
    drain_side_queue_items(side_queues)
    if memory_tracker is not None:
        memory_tracker.log_peak(stage_label or "Worker stage")

    nonzero_exitcodes = [task.exitcode for task in tasks if task.exitcode not in (None, 0)]
    if nonzero_exitcodes:
        logging.warning("%s worker exitcodes | exitcodes=%s", stage_label or "Worker stage", nonzero_exitcodes)

    return {'merged_result_chunks': merged_result_chunks, 'worker_count': len(tasks)}

#################################################################
#                Predicates and Functionalities                 #
#################################################################

def initializePredicateSubsumption(predicates1, predicates2, pred2superPred12={}, pred2superPred21={}, relinit=0.1):
    """ 
    Initializes identical relations to 1.0, all others as given or else to RELINC.

    Parameters
    ----------
    predicates1 : set
        set of predicates in KB1
    predicates2 : set
        set of predicates in KB2
    pred2superPred12 : dict, optional
        subsumption scores from predicates in KB1 to predicates in KB2
    pred2superPred21 : dict, optional
        subsumption scores from predicates in KB2 to predicates in KB1
    relinit : float, optional
        initial score for non-identical relations, by default 0.1
    
    Returns
    -------
    result : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    """
    result = {}
    for pred1 in predicates1:
        if pred1 not in result:
            result[pred1] = {}
        for pred2 in predicates2:
            if pred2 not in result:
                result[pred2] = {}
            if pred1 == pred2:
                result[pred1][pred2] = 1.0
            else:
                score1 = max(pred2superPred12.get(pred1,{}).get(pred2,relinit),
                             pred2superPred21.get(pred1,{}).get(pred2,relinit))
                score2 = max(pred2superPred21.get(pred2,{}).get(pred1,relinit),
                             pred2superPred12.get(pred2,{}).get(pred1,relinit))
                result[pred1][pred2] = score1
                result[pred2][pred1] = score2
    return result

class CompactDenseDefaultPredicateMapping:
    """ID-level equivalent of initializePredicateSubsumption."""
    def __init__(self, kb_src, kb_dst, relinit=0.1):
        self.relinit = relinit 
        self.target_predicate_ids = tuple(kb_dst.iter_predicate_ids()) # all target predicate IDs
        dst_predicate_ids = {}
        for pred2_id in kb_dst.iter_predicate_ids():
            pred2 = kb_dst.predicate_for_id(pred2_id)
            dst_predicate_ids.setdefault(pred2, []).append(pred2_id)

        equivalent_pairs = set()
        for pred1_id in kb_src.iter_predicate_ids():
            pred1 = kb_src.predicate_for_id(pred1_id)
            for pred2_id in dst_predicate_ids.get(pred1, ()):
                equivalent_pairs.add((pred1_id, pred2_id))
        self.equivalent_pairs = frozenset(equivalent_pairs) # exact same predicate ID pairs
        self.seed_scores = {}

    def set_score_ids(self, predicate1_id, predicate2_id, score):
        self.seed_scores[(int(predicate1_id), int(predicate2_id))] = float(score)

    def score_ids(self, predicate1_id, predicate2_id):
        pair = (int(predicate1_id), int(predicate2_id))
        if pair in self.seed_scores:
            return self.seed_scores[pair]
        if pair in self.equivalent_pairs:
            return 1.0
        return self.relinit

def is_dense_default_predicate_mapping(mapping):
    return isinstance(mapping, CompactDenseDefaultPredicateMapping)

def initializePredicateIdentityOnly(predicates1, predicates2):
    """
    Initialize predicate subsumption with exact predicate matches only.

    Parameters
    ----------
    predicates1 : set
        Set of predicates in KB1
    predicates2 : set
        Set of predicates in KB2
    
    Returns
    -------
    result : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    """
    result = {}
    for pred1 in predicates1:
        if pred1 not in result:
            result[pred1] = {}
        for pred2 in predicates2:
            if pred2 not in result:
                result[pred2] = {}
            if pred1 == pred2:
                result[pred1][pred2] = 1.0
    return result

def initializePredicateIdentityOnlyIds(kb_src, kb_dst):
    """ID-level equivalent of initializePredicateIdentityOnly."""
    src_side = sides.graph_predicate_side(kb_src, sides.PRED1)
    dst_side = sides.graph_predicate_side(kb_dst, sides.PRED2)
    dst_predicate_ids = {}
    for pred2_id in kb_dst.iter_predicate_ids():
        pred2 = kb_dst.predicate_for_id(pred2_id)
        dst_predicate_ids.setdefault(pred2, []).append(pred2_id)

    result = {}
    for pred1_id in kb_src.iter_predicate_ids():
        pred1 = kb_src.predicate_for_id(pred1_id)
        pred1_key = sides.side_predicate_key(src_side, pred1_id)
        equivalent_pred2_ids = set(dst_predicate_ids.get(pred1, ()))
        for pred2_id in equivalent_pred2_ids:
            pred2_key = sides.side_predicate_key(dst_side, pred2_id)
            result.setdefault(pred1_key, {})[pred2_key] = 1.0
    return result

def updatePredicateSubsumption(pred2superPred12, pred2superPred21, previousPredicate2superPredicate):
    """ 
    Updates the predicate subsumptions from two directions: kb1->kb2, kb2->kb1 

    Parameters
    ----------
    pred2superPred12 : dict
        current subsumption scores from predicates in KB1 to predicates in KB2
    pred2superPred21 : dict
        current subsumption scores from predicates in KB2 to predicates in KB1
    previousPredicate2superPredicate : dict
        previous subsumption scores to be updated
    """
    for pred1 in pred2superPred12:
        if previousPredicate2superPredicate.get(pred1) is None:
            previousPredicate2superPredicate[pred1] = {}
        for pred2 in pred2superPred12[pred1]:
            # Make relation subsumption monotonic
            previousPredicate2superPredicate[pred1][pred2] = max(previousPredicate2superPredicate[pred1].get(pred2, 0),
                                                                 pred2superPred12[pred1][pred2])
    for pred2 in pred2superPred21:
        if previousPredicate2superPredicate.get(pred2) is None:
            previousPredicate2superPredicate[pred2] = {}
        for pred1 in pred2superPred21[pred2]:
            # Make relation subsumption monotonic
            previousPredicate2superPredicate[pred2][pred1] = max(previousPredicate2superPredicate[pred2].get(pred1, 0),
                                                                 pred2superPred21[pred2][pred1])

def computeFunctionalities(kb, gram=[]):
    """ 
    Returns the functionalities of the predicates in the KB 

    Parameters
    ----------
    kb : Graph
        The input knowledge base
    gram : list
        List of integers indicating the n-grams to consider for functionality computation
    
    Returns
    -------
    dict
        A dictionary mapping predicates to their functionalities
    """
    predicate2numFacts={}
    predicate2subjects={}
    for subject in kb.subjects():
        facts = list(kb.triplesWithSubject(subject))
        for n in gram:
            if n == 1:
                for fact in facts:
                    predicate_ = fact[PRED]
                    if predicate_ not in predicate2numFacts:
                        predicate2numFacts[predicate_]=0
                        predicate2subjects[predicate_]=set()
                    predicate2numFacts[predicate_]+=1
                    predicate2subjects[predicate_].add(fact[SUBJ])
                continue
            # gram > 1
            cnt = 0
            for evs in combinations(facts, n):
                cnt += 1
                predicate_ = tuple(sorted([utils.invert(fact[PRED]) for fact in evs]))
                subjs_ = tuple(sorted([fact[OBJ] for fact in evs]))
                if predicate_ not in predicate2numFacts:
                    predicate2numFacts[predicate_]=0
                    predicate2subjects[predicate_]=set()
                predicate2numFacts[predicate_]+=1
                predicate2subjects[predicate_].add(subjs_)
                if cnt > 100000: # avoid memory overflow
                    break
    return { predicate : len(predicate2subjects[predicate])/predicate2numFacts[predicate] for predicate in predicate2numFacts }

def computeFunctionalitiesIds(kb, gram=[]):
    """
    ID-level functionality computation for CompactGraph.
    Returned keys are local predicate IDs, or tuples of local predicate IDs.
    """
    predicate2numFacts = {}
    predicate2subjects = {}
    for subject_id in range(kb.num_entities()):
        facts = list(kb.triplesWithSubjectIds(subject_id))
        for n in gram:
            if n == 1:
                for fact in facts:
                    predicate_key = fact[PRED]
                    if predicate_key not in predicate2numFacts:
                        predicate2numFacts[predicate_key] = 0
                        predicate2subjects[predicate_key] = set()
                    predicate2numFacts[predicate_key] += 1
                    predicate2subjects[predicate_key].add(fact[SUBJ])
                continue

            cnt = 0
            for evs in combinations(facts, n):
                cnt += 1
                predicate_ids = []
                skip = False
                for fact in evs:
                    inverse_predicate_id = kb._inverse_predicate_id(fact[PRED])
                    if inverse_predicate_id is None:
                        skip = True
                        break
                    predicate_ids.append(inverse_predicate_id)
                if skip:
                    continue
                predicate_key = tuple(sorted(predicate_ids))
                subject_key = tuple(sorted(fact[OBJ] for fact in evs))
                if predicate_key not in predicate2numFacts:
                    predicate2numFacts[predicate_key] = 0
                    predicate2subjects[predicate_key] = set()
                predicate2numFacts[predicate_key] += 1
                predicate2subjects[predicate_key].add(subject_key)
                if cnt > 100000: # avoid memory overflow
                    break
    return {predicate: len(predicate2subjects[predicate]) / predicate2numFacts[predicate] for predicate in predicate2numFacts}

def computeFunctionalitiesForPredicates(kb, predicates):
    """ 
    Returns the functionalities of the given predicates list in the KB 

    Parameters
    ----------
    kb : Graph
        The input knowledge base
    predicates : list
        Relation list to compute functionalities for
    
    Returns
    -------
    float
        The functionality of the given relation list in the KB
    """
    pred_numFacts = 0
    pred_subjects = set()
    counter = Counter(predicates)
    predicates_inv = sorted([utils.invert(pred) for pred in predicates])
    subKB = kb.headTriplesWithPredicateList({utils.invert(pred):counter[pred] for pred in set(predicates)})
    for obj in subKB:
        for evs in combinations(subKB[obj], len(predicates_inv)):
            _, predicate_, subjs_ = zip(*evs)
            if tuple(sorted(predicate_)) == tuple(predicates_inv):
                pred_numFacts += 1
                pred_subjects.add(tuple(sorted(subjs_)))
    return len(pred_subjects) / pred_numFacts if pred_numFacts > 0 else 0


#################################################################
#                     Implication Functions                     #
#################################################################

def updateScoreMin(mapping, key1, key2, *body, min_score=0.0):
    """ 
    Updates mapping[key1][key2] so that the rule body=>mapping[key1][key2] holds 
        using the minimum operator (Godel logic), as shown in equation (1) in the paper.
    
    Parameters
    ----------
    mapping : dict
        Nested dictionary to be updated with the entity alignment scores
    key1 : hashable
        The entity from KB1
    key2 : hashable
        The entity from KB2
    body : list of float
        The values in the body of the rule
    min_score : float, optional
        Ignore updates whose resulting score is not strictly above this threshold.
    """
    curScore = 0
    key2scores = mapping.get(key1)
    created_pair = key2scores is None or key2 not in key2scores
    if key2scores is not None and key2 in key2scores:
        curScore = key2scores[key2]
    tmp = max(curScore, min(min(body),1.0))

    if tmp >= min_score:
        if key1 not in mapping:
            mapping[key1]={}
        mapping[key1][key2] = tmp
        return created_pair
    return False

def updateScoreAdditiveMin(mapping, key1, key2, factor, *body):
    """ 
    Updates mapping[key1][key2] so that the rule body=>mapping[key1][key2] holds, but adds the values. 
    It is used for subrelation rules, as shown in equation (2) in the paper.

    Parameters
    ----------
    mapping : dict
        Nested dictionary to be updated with the subrelation scores
    key1 : hashable
        The predicate from KB1
    key2 : hashable
        The predicate from KB2
    factor : float
        Normalization factor (already multipled by benefit of the doubt paramter)
    body : list of float
        The values in the body of the rule
    """
    curScore = 0
    if key1 in mapping and key2 in mapping.get(key1, {}):
        curScore = mapping[key1][key2]
    value = curScore + min(body) * factor
    if value > 0:
        if key1 not in mapping:
            mapping[key1]={}
        mapping[key1][key2] = max(min(value, 1.0), 0)
    return

def updateMaxScoreMin(mapping, pred, fact, *body):
    """ 
    Parameters
    ----------
    mapping : dict
        Dictionary to be updated with the maximum aligned scoring fact for each predicate
    pred : hashable
        The predicate from KB2
    fact : tuple
        The fact from KB2
    body : list of float
        The values in the body of the rule
    """
    # subrelation rules
    score = min(body)
    if pred not in mapping:
        mapping[pred] = (fact, score)
    else:
        if score > mapping[pred][1]:
            mapping[pred] = (fact, score)
    return


#################################################################
#                      Score Merge Helpers                      #
#################################################################

def _nested_max_score(ent_max_assign, entity, ent_max_score_cache=None):
    """
    Return the maximum alignment score for an entity from a nested score mapping.

    Parameters
    ----------
    ent_max_assign : dict | None
        Nested dictionary of entity alignment scores.
    entity : hashable
        Entity whose current maximum score should be looked up.
    ent_max_score_cache : dict | None, optional
        Optional cache mapping entity -> maximum similarity score.

    Returns
    -------
    float
        The maximum known alignment score for the entity, or 0.0 if absent.
    """
    if ent_max_assign is None:
        return 0.0
    if ent_max_score_cache is not None and entity in ent_max_score_cache:
        return ent_max_score_cache[entity]
    score = max(ent_max_assign.get(entity, {None: 0.0}).values())
    if ent_max_score_cache is not None:
        ent_max_score_cache[entity] = score
    return score

def _indexed_max_score(max_scores, entity_id):
    """
    Return the maximum alignment score for an entity ID.

    Parameters
    ----------
    max_scores : list | dict | None
        Indexed or dictionary-backed entity maximum scores.
    entity_id : int
        Entity ID whose maximum score should be read.

    Returns
    -------
    float
        The maximum known alignment score for the entity ID, or 0.0 if absent.
    """
    if max_scores is None or entity_id is None:
        return 0.0
    if isinstance(max_scores, dict):
        return max_scores.get(entity_id, 0.0)
    if entity_id < 0 or entity_id >= len(max_scores):
        return 0.0
    return float(max_scores[entity_id])

def merge_same_as_score_chunk(target_scores, score_chunk, min_score=0.0,
                              ent_max_assign=None, ent_max_score_cache=None):
    """
    Merge a chunk of entity scores into the global sameAs mapping using max
    aggregation. Updates are kept only when they improve the existing score 
    for the same source-target pair.

    Parameters
    ----------
    target_scores : dict
        Nested dictionary of global entity alignment scores to update in place.
    score_chunk : dict
        Nested dictionary of worker-produced entity alignment scores.
    min_score : float, optional
        Drop candidate pairs whose score is not strictly above this threshold.
    ent_max_assign : dict | None, optional
        Current bilateral max assignment used to skip pairs that cannot improve
        either endpoint's best known score.
    ent_max_score_cache : dict | None, optional
        Optional cache for maximum scores read from ent_max_assign.

    Returns
    -------
    None
        The target_scores mapping is updated in place.
    """
    for subj1, subj2scores in score_chunk.items():
        target_map = target_scores.get(subj1)
        if target_map is None:
            target_map = {}
            target_scores[subj1] = target_map

        max_score1 = _nested_max_score(ent_max_assign, subj1, ent_max_score_cache)
        for subj2, score in subj2scores.items():
            if score <= min_score:
                continue
            max_score2 = _nested_max_score(ent_max_assign, subj2, ent_max_score_cache)
            if score <= max(max_score1, max_score2):
                continue
            if score > target_map.get(subj2, 0):
                target_map[subj2] = score

        if not target_map:
            target_scores.pop(subj1, None)

def merge_same_as_score_chunk_ids(target_scores, score_chunk, kb_src, kb_dst, min_score=0.0, 
                                  src_max_scores=None, dst_max_scores=None):
    """
    ID-level equivalent of merge_same_as_score_chunk.

    Parameters
    ----------
    target_scores : dict
        Side/id-keyed nested dictionary of global entity alignment scores to
        update in place.
    score_chunk : dict
        Nested dictionary keyed by source entity IDs and target entity IDs.
    kb_src : Graph
        Source knowledge base used to encode source-side entity keys.
    kb_dst : Graph
        Target knowledge base used to encode target-side entity keys.
    min_score : float, optional
        Drop candidate pairs whose score is not strictly above this threshold.
    src_max_scores : list | dict | None, optional
        Current maximum score per source entity ID.
    dst_max_scores : list | dict | None, optional
        Current maximum score per target entity ID.

    Returns
    -------
    None
        The target_scores mapping is updated in place.
    """

    if not sides.is_id_keyed_mapping(target_scores):
        raise TypeError("ID-level score chunks require side/id-keyed target_scores.")

    src_side = sides.graph_side(kb_src, sides.KB1)
    dst_side = sides.graph_side(kb_dst, sides.KB2)
    for subj1_id, subj2scores in score_chunk.items():
        subj1_key = sides.side_entity_key(src_side, subj1_id)
        target_map = target_scores.get(subj1_key)
        if target_map is None:
            target_map = {}
            target_scores[subj1_key] = target_map

        for subj2_id, score in subj2scores.items():
            if score <= min_score:
                continue
            if score <= max(_indexed_max_score(src_max_scores, subj1_id), _indexed_max_score(dst_max_scores, subj2_id)):
                continue
            subj2_key = sides.side_entity_key(dst_side, subj2_id)
            if score > target_map.get(subj2_key, 0):
                target_map[subj2_key] = score

        if not target_map:
            target_scores.pop(subj1_key, None)

def _flush_match_scores(result_queue, ent_match_scores):
    if not ent_match_scores:
        return {}, 0
    result_queue.put(ent_match_scores)
    return {}, 0


#################################################################
#                      Bootstrap Procedure                      #
#################################################################

def _1st_iteration(kb_src, kb_dst, pred2superPred, functionalities,
        queue, ent_match_tuple_queue, ent_max_assign, min_score=0.0):
    """ 
    The first iteration used for bootstrapping the algorithm using the initial literal alignments.
    
    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    pred2superPred : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    functionalities : dict
        A dictionary mapping predicates to their functionalities
    queue : mp.Queue
        A multiprocessing queue containing the entities to be aligned
    ent_match_tuple_queue : mp.Queue
        A multiprocessing queue to store the resulting entity alignment scores
    ent_max_assign : dict
        The bilateral max assignment computed from the initial literal alignments
    min_score : float, optional
        Ignore bootstrap matches whose score is not strictly above this threshold.
    """
    ent_match_scores = dict()
    pending_pairs = 0
    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break
        
        # keep only the subjects that are in the max assignment
        for subj_kb1 in entity_chunk:
            subject_scores = ent_max_assign.get(subj_kb1)
            if not subject_scores:
                continue

            for fact1 in kb_src.triplesWithSubject(subj_kb1):
                # We don't match literals
                if utils.isLiteral(fact1[OBJ]):
                    continue

                pred1 = fact1[PRED]
                pred_targets = pred2superPred.get(pred1)
                if not pred_targets:
                    continue

                localfunc1 = kb_src.localFunctionality(subj_kb1, pred1)
                globalfunc1 = functionalities[pred1]
                for subj_kb2, subject_score in subject_scores.items():
                    if not kb_dst.has_subject(subj_kb2):
                        continue
                    for fact2 in kb_dst.triplesWithSubject(subj_kb2, pred_targets):
                        # We don't match literals
                        if utils.isLiteral(fact2[OBJ]):
                            continue

                        pred2 = fact2[PRED]
                        reverse_pred_targets = pred2superPred.get(pred2, {})
                        # Update
                        if updateScoreMin(
                            # Objects are the same, ...
                            ent_match_scores, fact1[OBJ], fact2[OBJ],
                            # ... if the subjects are the same, ...
                            subject_score,
                            # ... and the predicate is locally functional, ...
                            localfunc1, kb_dst.localFunctionality(subj_kb2, pred2),
                            # ... and the predicate is globally functional,
                            globalfunc1, functionalities[pred2],
                            # ... and the target predicate is subsumed.
                            max(pred_targets[pred2], reverse_pred_targets.get(pred1, 0)),
                            # ... and the minimum score threshold is met.
                            min_score=min_score,
                        ):
                            pending_pairs += 1
                        if pending_pairs >= MATCH_RESULT_FLUSH_PAIRS:
                            ent_match_scores, pending_pairs = _flush_match_scores(ent_match_tuple_queue, ent_match_scores)
        ent_match_scores, pending_pairs = _flush_match_scores(ent_match_tuple_queue,ent_match_scores)
    ent_match_scores, pending_pairs = _flush_match_scores(ent_match_tuple_queue, ent_match_scores)
    exit(0)

def _1st_iteration_compact(kb_src, kb_dst, pred2superPred, functionalities,
        queue, ent_match_tuple_queue, ent_max_assign, min_score=0.0):
    """ID-level bootstrap worker for CompactGraph."""

    ent_match_scores = dict()
    pending_pairs = 0

    src_to_dst_scores, _, _ = sides.encode_compact_entity_assign(kb_src, kb_dst, ent_max_assign)
    if not is_dense_default_predicate_mapping(pred2superPred):
        raise TypeError("Compact bootstrap expects CompactDenseDefaultPredicateMapping.")
    compact_predicate_score = pred2superPred.score_ids
    src_functionalities, _, dst_functionalities, _ = sides.resolve_compact_worker_functionalities(
        kb_src,
        kb_dst,
        functionalities,
    )

    # Helper function to flush the current match scores to the parent process
    def flush_compact_match_scores():
        nonlocal ent_match_scores, pending_pairs
        if not ent_match_scores:
            ent_match_scores = {}
            pending_pairs = 0
            return
        ent_match_tuple_queue.put(ent_match_scores)
        ent_match_scores = {}
        pending_pairs = 0

    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break

        for subj_kb1 in entity_chunk:
            subj_kb1_id = sides.coerce_entity_id(kb_src, subj_kb1)
            if subj_kb1_id is None:
                continue

            subject_scores = src_to_dst_scores.get(subj_kb1_id)
            if not subject_scores:
                continue

            for _, pred1_id, obj1_id in kb_src.triplesWithSubjectIds(subj_kb1_id):
                if kb_src.is_literal_id(obj1_id):
                    continue

                localfunc1 = kb_src.localFunctionalityIds(subj_kb1_id, pred1_id)
                globalfunc1 = src_functionalities[pred1_id]

                for subj_kb2_id, subject_score in subject_scores.items():
                    if not kb_dst.has_subject_id(subj_kb2_id):
                        continue

                    for _, pred2_id, obj2_id in kb_dst.triplesWithSubjectIds(subj_kb2_id):
                        if kb_dst.is_literal_id(obj2_id):
                            continue

                        if updateScoreMin(
                            ent_match_scores, obj1_id, obj2_id,
                            subject_score,
                            localfunc1, kb_dst.localFunctionalityIds(subj_kb2_id, pred2_id),
                            globalfunc1, dst_functionalities[pred2_id],
                            compact_predicate_score(pred1_id, pred2_id),
                            min_score=min_score,
                        ):
                            pending_pairs += 1
                        if pending_pairs >= MATCH_RESULT_FLUSH_PAIRS:
                            flush_compact_match_scores()
        flush_compact_match_scores()
    flush_compact_match_scores()
    exit(0)

def bootstrap_algo(kb_src, kb_dst, sameAsScore, pred2superPred, functionalities,
                   min_score=0.0, num_workers=None):
    """ 
    Bootstrapping the algorithm using the initial literal alignments.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    sameAsScore : dict
        Nested dictionary of entity alignment scores (includes initial literal alignments)
    pred2superPred : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    functionalities : dict
        A dictionary mapping predicates to their functionalities
    min_score : float, optional
        Ignore bootstrap matches whose score is not strictly above this threshold.
    num_workers : int | None, optional
        Number of worker processes to use. When None, use the CPU count.

    Returns
    -------
    None
        The sameAsScore mapping is updated in place with bootstrap matches.
    """
    stage_name = "Bootstrap worker-stage"
    logging.info("%s worker config | workers=%s ", stage_name , num_workers)
    ent_max_assign = bilateral_max_assign(sameAsScore)
    tasks = []
    ent_queue_ = None
    ent_match_tuple_queue_ = None
    try:
        log.prepare_for_worker_fork(stage_name)
        num_workers = max(1, num_workers or mp.cpu_count())
        queue_size = max(4, num_workers * 2)
        ent_queue_ = mp.Queue(maxsize=queue_size)
        ent_match_tuple_queue_ = mp.Queue(maxsize=queue_size)

        use_compact_id_match = _can_use_compact_id_match(kb_src, kb_dst)
        worker_target = _1st_iteration_compact if use_compact_id_match else _1st_iteration
        if use_compact_id_match and not sides.is_id_keyed_mapping(sameAsScore):
            raise TypeError(
                "CompactGraph bootstrap requires side/id-keyed sameAsScore; "
                "encode inputs with sides.maybe_encode_same_as_scores first."
            )

        def merge_chunk(ent_match_score_dict):
            if use_compact_id_match:
                merge_same_as_score_chunk_ids(
                    sameAsScore,
                    ent_match_score_dict,
                    kb_src,
                    kb_dst,
                    min_score=min_score,
                )
            else:
                merge_same_as_score_chunk(sameAsScore, ent_match_score_dict, min_score=min_score)
        
        logging.info("%s worker implementation | mode=%s", stage_name, "compact-id" if worker_target is _1st_iteration_compact else "string")
        logging.info("%s active source candidates | sources=%s", stage_name, len(ent_max_assign))

        for _ in range(num_workers):
            args = (
                kb_src, kb_dst,
                pred2superPred,
                functionalities,
                ent_queue_,
                ent_match_tuple_queue_,
                ent_max_assign,
                min_score,
            )
            task = mp.Process(target=worker_target, args=args)
            task.start()
            tasks.append(task)
        memory_tracker = log.ProcessMemoryPeakTracker(stage_name, [task.pid for task in tasks])
        memory_tracker.log_current(f"{stage_name} memory start")
        feed_entity_chunks(
            ent_queue_,
            kb_src,
            num_workers,
            result_queue=ent_match_tuple_queue_,
            merge_fn=merge_chunk,
            include_literals=True,
            stage_label=stage_name,
            memory_tracker=memory_tracker,
            only_literals=True,
            source_entities=ent_max_assign.keys(),
        )
        wait_for_workers_and_drain(
            tasks,
            ent_match_tuple_queue_,
            merge_chunk,
            stage_label=stage_name,
            memory_tracker=memory_tracker,
        )
    finally:
        if ent_queue_ is not None:
            ent_queue_.close()
            ent_queue_.join_thread()
        if ent_match_tuple_queue_ is not None:
            ent_match_tuple_queue_.close()
            ent_match_tuple_queue_.join_thread()


#################################################################
#                      Subrelation Mapping                      #
#################################################################

def iter_fact_chunks(graph, chunk_size=SUBRELATION_FACT_CHUNK_SIZE):
    """Yield graph facts in bounded chunks."""
    chunk = []
    for fact in graph:
        chunk.append(fact)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def iter_fact_id_chunks(graph, chunk_size=SUBRELATION_FACT_CHUNK_SIZE):
    """Yield ID-level graph facts in bounded chunks."""
    chunk = []
    for fact in graph.iterFactIds():
        chunk.append(fact)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def iter_candidate_fact_id_chunks(graph, active_entity_ids, chunk_size=SUBRELATION_FACT_CHUNK_SIZE):
    """
    Yield ID fact chunks whose subject and object can both match.

    Parameters
    ----------
    graph : Graph
        Knowledge base exposing ID-level subject and fact iteration.
    active_entity_ids : set
        Entity IDs that currently have at least one candidate alignment.
    chunk_size : int, optional
        Maximum number of candidate ID facts per yielded chunk.

    Yields
    ------
    list
        A list of ID-level facts whose subject and object are active entities.
    """
    chunk = []
    for subject_id in sorted(active_entity_ids):
        for fact in graph.triplesWithSubjectIds(subject_id):
            if fact[OBJ] not in active_entity_ids:
                continue
            chunk.append(fact)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk

def merge_additive_score_mapping(target_mapping, score_mapping):
    """Merge a mapping of additive scores into a target mapping, clamping to [0, 1]"""
    for key1, key2scores in score_mapping.items():
        target_scores = target_mapping.setdefault(key1, {})
        for key2, score in key2scores.items():
            target_scores[key2] = max(min(target_scores.get(key2, 0) + score, 1.0), 0)

def _subrelation_update_additive(mapping, key1, key2, value):
    """Update a mapping of additive scores, clamping to [0, 1]"""
    if value <= 0:
        return
    key2scores = mapping.setdefault(key1, {})
    key2scores[key2] = max(min(key2scores.get(key2, 0) + value, 1.0), 0)

def _subject_object_predicate_map_for_subject(kb_lookup, subject_id, cache):
    """
    Return an object-to-predicates map for one target subject using cache.

    Parameters
    ----------
    kb_lookup : Graph
        Target knowledge base used for subject fact lookups.
    subject_id : int
        Target subject entity ID to inspect.
    cache : OrderedDict
        LRU cache from subject ID to object_id -> tuple(predicate_ids).

    Returns
    -------
    tuple
        (object_to_predicates, None) when the subject is cached or cacheable;
        otherwise (None, triples) for low-degree subjects that should be scanned directly.
    """
    cached = cache.get(subject_id)
    # if the subject_id is already cached, return the cached mapping and update the cache
    if cached is not None:
        cache.move_to_end(subject_id)
        return cached, None
    
    # if the subject_id is not cached, look up the triples and build the mapping
    triples = list(kb_lookup.triplesWithSubjectIds(subject_id))
    # if the number of triples is below the threshold, bypass caching 
    if len(triples) < SUBRELATION_SUBJECT_OBJECT_CACHE_DEGREE_THRESHOLD:
        return None, triples
    # if the number of triples is above the threshold, build the mapping and cache it
    object_to_predicates = {}
    for _, predicate_id, object_id in triples:
        object_to_predicates.setdefault(object_id, []).append(predicate_id)
    object_to_predicates = {object_id: tuple(predicate_ids) for object_id, predicate_ids in object_to_predicates.items()} # convert lists to tuples for memory efficiency
    cache[subject_id] = object_to_predicates

    # if the cache exceeds the maximum number of subjects, evict the oldest entry
    if len(cache) > SUBRELATION_SUBJECT_OBJECT_CACHE_MAX_SUBJECTS:
        cache.popitem(last=False)
    return object_to_predicates, None

def _compute_subrelation_chunk(fact_chunk, kb_lookup, ent_max_assign, source_predicate_counts, alpha):
    """
    Compute source-to-target predicate subsumption scores for a chunk of facts.

    Parameters
    ----------
    fact_chunk : list
        Source facts to process.
    kb_lookup : Graph
        Target knowledge base used to find aligned facts.
    ent_max_assign : dict
        Current entity alignment scores.
    source_predicate_counts : dict
        Number of source facts for each source predicate.
    alpha : float
        Benefit of the doubt parameter for additive subrelation scoring.

    Returns
    -------
    tuple
        A local predicate score mapping, number of processed facts, and number
        of facts whose subject and object both had alignment candidates.
    """
    local_mapping = {}
    processed_facts = 0
    matched_facts = 0

    for fact in fact_chunk:
        processed_facts += 1
        subject_scores = ent_max_assign.get(fact[SUBJ])
        if not subject_scores:
            continue
        object_scores = ent_max_assign.get(fact[OBJ])
        if not object_scores:
            continue
        matched_facts += 1

        rel_max_scores = {}
        for aligned_subject, subject_score in subject_scores.items():
            for aligned_predicate, aligned_objects in kb_lookup.subject_items(aligned_subject):
                for aligned_object in aligned_objects:
                    object_score = object_scores.get(aligned_object)
                    if object_score is None:
                        continue
                    score = min(subject_score, object_score)
                    if score > rel_max_scores.get(aligned_predicate, 0): # keep the max score for each aligned predicate
                        rel_max_scores[aligned_predicate] = score

        predicate_weight = alpha / source_predicate_counts.get(fact[PRED], 1)
        for aligned_predicate, score in rel_max_scores.items():
            _subrelation_update_additive(local_mapping, fact[PRED], aligned_predicate, score * predicate_weight)

    return local_mapping, processed_facts, matched_facts

def _compute_subrelation_chunk_compact(fact_chunk, kb_lookup, ent_max_assign_ids,
                                       source_predicate_counts_ids, alpha, subject_object_cache=None):
    """
    Compute ID-level source-to-target predicate subsumption scores for a fact chunk.

    Parameters
    ----------
    fact_chunk : list
        Source ID facts whose subject and object both have alignment candidates.
    kb_lookup : Graph
        Target knowledge base used to find aligned ID facts.
    ent_max_assign_ids : dict
        Current source-ID to target-ID entity alignment scores.
    source_predicate_counts_ids : dict
        Number of source facts for each source predicate ID.
    alpha : float
        Benefit of the doubt parameter for additive subrelation scoring.
    subject_object_cache : OrderedDict | None, optional
        Worker-local cache for high-degree target subject lookups.

    Returns
    -------
    tuple
        A local predicate-ID score mapping and number of processed candidate facts.
    """
    local_mapping = {}
    processed_facts = 0
    # use cache for subject-object-predicate mapping to avoid repeated lookups for high-degree subjects
    if subject_object_cache is None:
        subject_object_cache = OrderedDict()

    for subject_id, predicate_id, object_id in fact_chunk:
        processed_facts += 1
        subject_scores = ent_max_assign_ids[subject_id]
        object_scores = ent_max_assign_ids[object_id]

        rel_max_scores = {}
        for aligned_subject_id, subject_score in subject_scores.items():
            if not kb_lookup.has_subject_id(aligned_subject_id):
                continue
            object_to_predicates, fallback_triples = _subject_object_predicate_map_for_subject(
                kb_lookup,
                aligned_subject_id,
                subject_object_cache,
            )
            if object_to_predicates is None:
                for _, aligned_predicate_id, aligned_object_id in fallback_triples:
                    object_score = object_scores.get(aligned_object_id)
                    if object_score is None:
                        continue
                    score = min(subject_score, object_score)
                    if score > rel_max_scores.get(aligned_predicate_id, 0):
                        rel_max_scores[aligned_predicate_id] = score
            else:
                for aligned_object_id, object_score in object_scores.items():
                    for aligned_predicate_id in object_to_predicates.get(aligned_object_id, ()):
                        score = min(subject_score, object_score)
                        if score > rel_max_scores.get(aligned_predicate_id, 0):
                            rel_max_scores[aligned_predicate_id] = score

        predicate_weight = alpha / source_predicate_counts_ids.get(predicate_id, 1)
        for aligned_predicate_id, score in rel_max_scores.items():
            _subrelation_update_additive(
                local_mapping,
                predicate_id,
                aligned_predicate_id,
                score * predicate_weight,
            )

    return local_mapping, processed_facts

def _map_subrelations_worker(kb_lookup, ent_max_assign, source_predicate_counts, alpha, task_queue, result_queue):
    while True:
        item = task_queue.get()
        if item is None:
            result_queue.put(None)
            return
        chunk_index, fact_chunk = item
        mapping, processed_facts, matched_facts = _compute_subrelation_chunk(
            fact_chunk,
            kb_lookup,
            ent_max_assign,
            source_predicate_counts,
            alpha,
        )
        result_queue.put((chunk_index, mapping, processed_facts, matched_facts))

def _map_subrelations_worker_compact(kb_lookup, ent_max_assign_ids, source_predicate_counts_ids, alpha, task_queue, result_queue):
    subject_object_cache = OrderedDict()
    while True:
        item = task_queue.get()
        if item is None:
            result_queue.put(None)
            return
        chunk_index, fact_chunk = item
        mapping, processed_facts = _compute_subrelation_chunk_compact(
            fact_chunk,
            kb_lookup,
            ent_max_assign_ids,
            source_predicate_counts_ids,
            alpha,
            subject_object_cache=subject_object_cache,
        )
        result_queue.put((chunk_index, mapping, processed_facts))

def _map_subrelations_direction(alpha, facts_iterable, kb_lookup, ent_max_assign,
                                source_predicate_counts, source_total_facts,
                                direction, num_workers, stage_label=None,
                                progress_interval=PROGRESS_LOG_INTERVAL):
    """Map subrelations from a source graph to a target graph in the given direction."""
    task_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    result_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    tasks = []
    mapping = {}
    processed_facts = 0
    submitted_chunks = 0
    finished_workers = 0
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)

    try:
        for _ in range(num_workers):
            task = mp.Process(
                target=_map_subrelations_worker,
                args=(
                    kb_lookup,
                    ent_max_assign,
                    source_predicate_counts,
                    alpha,
                    task_queue,
                    result_queue,
                ),
            )
            task.start()
            tasks.append(task)

        def merge_ready_results():
            nonlocal finished_workers, processed_facts
            drained = 0
            while True:
                try:
                    result = result_queue.get_nowait()
                except Empty:
                    break
                drained += 1
                if result is None:
                    finished_workers += 1
                    continue
                _, chunk_mapping, chunk_processed, chunk_matched = result
                merge_additive_score_mapping(mapping, chunk_mapping)
                processed_facts += chunk_processed
                matched_facts += chunk_matched
            return drained

        for fact_chunk in iter_fact_chunks(facts_iterable):
            while True:
                try:
                    task_queue.put_nowait((submitted_chunks, fact_chunk))
                    submitted_chunks += 1
                    break
                except Full:
                    merge_ready_results()
                    if progress_logger is not None:
                        progress_logger(
                            direction=direction,
                            submitted_chunks=submitted_chunks,
                            processed_facts=f"{processed_facts}/{source_total_facts}",
                            finished_workers=finished_workers,
                        )
                    time.sleep(RESULT_DRAIN_INTERVAL)
            merge_ready_results()
            if progress_logger is not None:
                progress_logger(
                    direction=direction,
                    submitted_chunks=submitted_chunks,
                    processed_facts=f"{processed_facts}/{source_total_facts}",
                    finished_workers=finished_workers,
                )

        for _ in range(num_workers):
            while True:
                try:
                    task_queue.put_nowait(None)
                    break
                except Full:
                    merge_ready_results()
                    time.sleep(RESULT_DRAIN_INTERVAL)

        while finished_workers < num_workers:
            merge_ready_results()
            if progress_logger is not None:
                progress_logger(
                    direction=direction,
                    submitted_chunks=submitted_chunks,
                    processed_facts=f"{processed_facts}/{source_total_facts}",
                    finished_workers=finished_workers,
                )
            time.sleep(RESULT_DRAIN_INTERVAL)

        for task in tasks:
            task.join()

        nonzero_exitcodes = [task.exitcode for task in tasks if task.exitcode not in (None, 0)]
        if nonzero_exitcodes:
            logging.warning(
                "%s subrelation worker exitcodes | direction=%s | exitcodes=%s",
                stage_label or "Subrelation stage", direction, nonzero_exitcodes,
            )
        return mapping
    finally:
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

def _can_use_compact_subrelation(kb_src, kb_dst):
    return (
        hasattr(kb_src, 'iterFactIds')
        and hasattr(kb_src, 'predicate_id')
        and hasattr(kb_src, 'predicate_for_id')
        and hasattr(kb_dst, 'iterFactIds')
        and hasattr(kb_dst, 'predicate_for_id')
    )

def _encode_compact_predicate_counts(graph):
    if hasattr(graph, 'predicate_counts_ids'):
        return graph.predicate_counts_ids()
    encoded_counts = {}
    for predicate, count in graph.predicates().items():
        predicate_id = graph.predicate_id(predicate)
        if predicate_id is not None:
            encoded_counts[predicate_id] = count
    return encoded_counts

def _map_subrelations_direction_compact(alpha, kb_src, kb_dst, ent_max_assign_ids,
                                        source_predicate_counts_ids, source_total_facts,
                                        direction, num_workers, stage_label=None,
                                        progress_interval=PROGRESS_LOG_INTERVAL):
    """ID-level equivalent of _map_subrelations_direction."""
    task_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    result_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    tasks = []
    mapping = {}
    processed_facts = 0
    matched_facts = 0
    submitted_chunks = 0
    finished_workers = 0
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)

    try:
        for _ in range(num_workers):
            task = mp.Process(
                target=_map_subrelations_worker_compact,
                args=(
                    kb_dst,
                    ent_max_assign_ids,
                    source_predicate_counts_ids,
                    alpha,
                    task_queue,
                    result_queue,
                ),
            )
            task.start()
            tasks.append(task)

        def merge_ready_results():
            nonlocal finished_workers, processed_facts, matched_facts
            drained = 0
            while True:
                try:
                    result = result_queue.get_nowait()
                except Empty:
                    break
                drained += 1
                if result is None:
                    finished_workers += 1
                    continue
                _, chunk_mapping, chunk_processed = result
                merge_additive_score_mapping(mapping, chunk_mapping)
                processed_facts += chunk_processed
                logging.debug(
                    "%s compact subrelation direct worker result | direction=%s | "
                    "processed_facts=%s",
                    stage_label or "Subrelation stage",
                    direction,
                    chunk_processed,
                )
            return drained

        for fact_chunk in iter_candidate_fact_id_chunks(kb_src, ent_max_assign_ids.keys()):
            while True:
                try:
                    task_queue.put_nowait((submitted_chunks, fact_chunk))
                    submitted_chunks += 1
                    break
                except Full:
                    merge_ready_results()
                    if progress_logger is not None:
                        progress_logger(
                            direction=direction,
                            submitted_chunks=submitted_chunks,
                            processed_facts=f"{processed_facts}/{source_total_facts}",
                            matched_facts=matched_facts,
                            finished_workers=finished_workers,
                        )
                    time.sleep(RESULT_DRAIN_INTERVAL)
            merge_ready_results()
            if progress_logger is not None:
                progress_logger(
                    direction=direction,
                    submitted_chunks=submitted_chunks,
                    processed_facts=f"{processed_facts}/{source_total_facts}",
                    matched_facts=matched_facts,
                    finished_workers=finished_workers,
                )

        for _ in range(num_workers):
            while True:
                try:
                    task_queue.put_nowait(None)
                    break
                except Full:
                    merge_ready_results()
                    time.sleep(RESULT_DRAIN_INTERVAL)

        while finished_workers < num_workers:
            merge_ready_results()
            if progress_logger is not None:
                progress_logger(
                    direction=direction,
                    submitted_chunks=submitted_chunks,
                    processed_facts=f"{processed_facts}/{source_total_facts}",
                    matched_facts=matched_facts,
                    finished_workers=finished_workers,
                )
            time.sleep(RESULT_DRAIN_INTERVAL)

        for task in tasks:
            task.join()

        nonzero_exitcodes = [task.exitcode for task in tasks if task.exitcode not in (None, 0)]
        if nonzero_exitcodes:
            logging.warning(
                "%s compact subrelation worker exitcodes | direction=%s | exitcodes=%s",
                stage_label or "Subrelation stage", direction, nonzero_exitcodes,
            )
        logging.info(
            "%s compact subrelation direct stats | direction=%s | "
            "submitted_chunks=%s | processed_facts=%s/%s",
            stage_label or "Subrelation stage", direction,
            submitted_chunks, processed_facts, source_total_facts,
        )
        return mapping
    finally:
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

def map_subrelations(alpha, kb_src, kb_dst, ent_maxAssign, previouspredicate2superPredicate,
                     stage_label=None, progress_interval=PROGRESS_LOG_INTERVAL, num_workers=1):
    """ 
    Maps subrelations (both directions) using the current entity alignments.

    Parameters
    ----------
    alpha : float
        Benefit of the doubt parameter for subrelation mapping
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    ent_maxAssign : dict
        The bilateral max assignment computed from the current entity alignments
    previouspredicate2superPredicate : dict
        Previous subsumption scores to be updated
    stage_label : str | None, optional
        Label used for throttled progress logging during subrelation mapping.
    progress_interval : float, optional
        Minimum interval in seconds between progress log messages.
    num_workers : int, optional
        Number of worker processes to use.

    Returns
    -------
    None
        previouspredicate2superPredicate is updated in place.
    """
    src_predicate_counts = kb_src.predicates()
    dst_predicate_counts = kb_dst.predicates()
    num_workers = max(1, num_workers or mp.cpu_count())
    src_total_facts = sum(src_predicate_counts.values())
    dst_total_facts = sum(dst_predicate_counts.values())
    logging.info("%s worker config | workers=%s ", stage_label or "Subrelation stage", num_workers)

    if _can_use_compact_subrelation(kb_src, kb_dst):
        logging.info("%s implementation | mode=compact", stage_label or "Subrelation stage")
        src_to_dst_scores, _, _ = sides.encode_compact_entity_assign(kb_src, kb_dst, ent_maxAssign)
        dst_to_src_scores, _, _ = sides.encode_compact_entity_assign(kb_dst, kb_src, ent_maxAssign)
        src_predicate_counts_ids = _encode_compact_predicate_counts(kb_src)
        dst_predicate_counts_ids = _encode_compact_predicate_counts(kb_dst)

        log.prepare_for_worker_fork(stage_label or "Compact subrelation worker-stage")
        # kb1 -> kb2
        pred2superPred1 = _map_subrelations_direction_compact(
            alpha,
            kb_src,
            kb_dst,
            src_to_dst_scores,
            src_predicate_counts_ids,
            src_total_facts,
            'kb1_to_kb2',
            num_workers,
            stage_label=stage_label,
            progress_interval=progress_interval,
        )
        # kb2 -> kb1
        pred2superPred2 = _map_subrelations_direction_compact(
            alpha,
            kb_dst,
            kb_src,
            dst_to_src_scores,
            dst_predicate_counts_ids,
            dst_total_facts,
            'kb2_to_kb1',
            num_workers,
            stage_label=stage_label,
            progress_interval=progress_interval,
        )

        src_side = sides.graph_predicate_side(kb_src, sides.PRED1)
        dst_side = sides.graph_predicate_side(kb_dst, sides.PRED2)
        pred2superPred1 = sides.side_key_compact_predicate_mapping(
            pred2superPred1,
            src_side,
            dst_side,
        )
        pred2superPred2 = sides.side_key_compact_predicate_mapping(
            pred2superPred2,
            dst_side,
            src_side,
        )
        updatePredicateSubsumption(pred2superPred1, pred2superPred2, previouspredicate2superPredicate)
        return

    log.prepare_for_worker_fork(stage_label or "Subrelation worker-stage")
    # kb1 -> kb2
    pred2superPred1 = _map_subrelations_direction(
        alpha,
        kb_src,
        kb_dst,
        ent_maxAssign,
        src_predicate_counts,
        src_total_facts,
        'kb1_to_kb2',
        num_workers,
        stage_label=stage_label,
        progress_interval=progress_interval,
    )
    # kb2 -> kb1
    pred2superPred2 = _map_subrelations_direction(
        alpha,
        kb_dst,
        kb_src,
        ent_maxAssign,
        dst_predicate_counts,
        dst_total_facts,
        'kb2_to_kb1',
        num_workers,
        stage_label=stage_label,
        progress_interval=progress_interval,
    )

    # complete the subrelation mapping
    updatePredicateSubsumption(pred2superPred1, pred2superPred2, previouspredicate2superPredicate)

def computeQuasiEqrel(kb_src, kb_dst, pred2superPred):
    """ 
    Computes the quasi equivalence relations between the two KGs' predicates, 
    the quasi equivalence is represented as r\\cong r' in paper.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    pred2superPred : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    
    Returns
    -------
    quasiEqrel_ : dict
        Nested dictionary of quasi equivalence relations
    """
    # if the predicate mapping is side-keyed (compactGraph)
    if sides.is_id_keyed_mapping(pred2superPred):
        src_side = sides.graph_predicate_side(kb_src, sides.PRED1)
        dst_side = sides.graph_predicate_side(kb_dst, sides.PRED2)
        quasiEqrel_ = {} # from kb1 to kb2
        for pred1 in pred2superPred:
            if not sides.is_side_key(pred1, src_side):
                continue
            for pred2 in pred2superPred[pred1]:
                if not sides.is_side_key(pred2, dst_side):
                    continue
                value = max(pred2superPred[pred1][pred2], pred2superPred.get(pred2, {}).get(pred1, 0))
                if pred1 not in quasiEqrel_:
                    quasiEqrel_[pred1] = {}
                quasiEqrel_[pred1][pred2] = value
        return quasiEqrel_

    # if the predicate mapping is directly predicate-keyed
    quasiEqrel_ = {} # from kb1 to kb2
    for pred1 in pred2superPred:
        for pred2 in pred2superPred[pred1]:
            value = max(pred2superPred[pred1][pred2], 
                        pred2superPred.get(pred2, {}).get(pred1, 0))
            if pred1 in kb_src.predicates():
                if pred2 in kb_dst.predicates():
                    if pred1 not in quasiEqrel_:
                        quasiEqrel_[pred1] = {}
                    quasiEqrel_[pred1][pred2] = value
            elif pred2 in kb_src.predicates() and pred2 not in quasiEqrel_:
                quasiEqrel_[pred2] = {}
                quasiEqrel_[pred2][pred1] = value
    return quasiEqrel_


#################################################################
#                 Bilateral Entity Assignment                   #
#################################################################

def bilateral_max_assign(sameASscore):
    """ 
    Computes the bilateral max assignment from the similarity scores, refer to equation (3) in paper.

    Parameters
    ----------
    sameASscore : dict
        Nested dictionary of entity alignment scores

    Returns
    -------
    res_max_assign : dict
        The bilateral max assignment of entities
    """
    # get max match for kb1
    max_match_e = {}
    # get max match for kb2 
    max_match_e_prime = {}

    for e, e_prime_scores in sameASscore.items():
        if not e_prime_scores:
            continue

        max_score = max(e_prime_scores.values())
        max_targets_e = []
        for e_prime, score in e_prime_scores.items():
            if score == max_score:
                max_targets_e.append(e_prime)

            current_match = max_match_e_prime.get(e_prime)
            if current_match is None or score > current_match['score']:
                max_match_e_prime[e_prime] = {'score': score, 'targets': {e}}
            elif score == current_match['score']:
                current_match['targets'].add(e)

        if max_targets_e:
            max_match_e[e] = {'score': max_score, 'targets': set(max_targets_e)}
    
    # bilateral max assignment
    res_max_assign = {}
    for e, match_data in max_match_e.items():
        max_score = match_data['score']
        max_targets = match_data['targets']

        for e_prime in max_targets:
            if (e_prime in max_match_e_prime and \
                max_match_e_prime[e_prime]['score'] == max_score and \
                e in max_match_e_prime[e_prime]['targets']):

                # write (e, e')
                if e not in res_max_assign:
                    res_max_assign[e] = {}
                res_max_assign[e][e_prime] = max_score
                # write (e',e)
                if e_prime not in res_max_assign:
                    res_max_assign[e_prime] = {}
                res_max_assign[e_prime][e] = max_score
    return res_max_assign

def _has_compact_id_match_api(kb):
    return (hasattr(kb, 'triplesWithSubjectIds') and hasattr(kb, 'localFunctionalityIds'))

def _can_use_compact_id_match(kb_src, kb_dst):
    return _has_compact_id_match_api(kb_src) and _has_compact_id_match_api(kb_dst)

def _compact_entity_count(graph):
    literal_flags = getattr(graph, '_entity_is_literal', None) # compact graph
    if literal_flags is not None:
        return len(literal_flags)
    return graph.entity_count()

class CompactEntityAssignIndex:
    """
    Read-only CSR-like view over bilateral max assignment scores.
    The parent process builds the arrays once as .npy memmaps, 
    and each worker opens the same files read-only to save memory.
    """

    def __init__(self, metadata):
        self.metadata = metadata
        self.src_offsets = utils.np.load(metadata['src_offsets_path'], mmap_mode='r') # CSR row offsets: source ID -> offset into dst_ids.
        self.dst_ids = utils.np.load(metadata['dst_ids_path'], mmap_mode='r') # target entity IDs for all source-to-target assignments.
        self.scores = utils.np.load(metadata['scores_path'], mmap_mode='r') # assignment scores for all source-to-target assignments.
        self.src_max = utils.np.load(metadata['src_max_path'], mmap_mode='r') # best assignment score for each source entity ID.
        self.dst_max = utils.np.load(metadata['dst_max_path'], mmap_mode='r') # best assignment score for each target entity ID.

    @property
    def pair_count(self):
        """Return the total number of source->target assignment pairs in the index."""
        return int(self.metadata.get('pair_count', len(self.dst_ids)))

    def max_src(self, entity_id):
        """Return the best assignment score for a source entity ID."""
        if entity_id is None or entity_id < 0 or entity_id >= len(self.src_max):
            return 0.0
        return float(self.src_max[entity_id])

    def max_dst(self, entity_id):
        """Return the best assignment score for a target entity ID."""
        if entity_id is None or entity_id < 0 or entity_id >= len(self.dst_max):
            return 0.0
        return float(self.dst_max[entity_id])

    def has_src_score(self, entity_id):
        """Return True if the source entity ID has at least one assignment with a positive score."""
        return self.max_src(entity_id) > 0.0

    def has_dst_score(self, entity_id):
        """Return True if the target entity ID has at least one assignment with a positive score."""
        return self.max_dst(entity_id) > 0.0

    def target_items(self, src_id):
        """Return a tuple of (target_id, score) pairs for all target entities currently aligned to the given source entity ID."""
        if src_id is None or src_id < 0 or src_id + 1 >= len(self.src_offsets):
            return ()
        # CSR slice for all target entities currently aligned to this source.
        start = int(self.src_offsets[src_id])
        end = int(self.src_offsets[src_id + 1])
        if start == end:
            return ()
        return tuple((int(self.dst_ids[index]), float(self.scores[index])) for index in range(start, end))

    def score(self, src_id, dst_id):
        """Return the assignment score for a given source and target entity ID pair."""
        if src_id is None or dst_id is None or src_id < 0 or src_id + 1 >= len(self.src_offsets):
            return 0.0
        start = int(self.src_offsets[src_id])
        end = int(self.src_offsets[src_id + 1])
        for index in range(start, end):
            if int(self.dst_ids[index]) == dst_id:
                return float(self.scores[index])
        return 0.0

def build_compact_entity_assign_index(kb_src, kb_dst, ent_max_assign, mmap_dir):
    """
    Build shared array files for compact entity assignment lookups.

    Parameters
    ----------
    kb_src : Graph
        Source knowledge base.
    kb_dst : Graph
        Target knowledge base.
    ent_max_assign : dict
        Current bilateral max assignment of entities.
    mmap_dir : str
        Directory where the .npy memmap arrays should be written.

    Returns
    -------
    dict
        Metadata containing memmap paths, entity counts, pair count, and total
        array bytes. Workers pass this metadata to CompactEntityAssignIndex.
    """
    np = utils.np
    # Rebuild the index every iteration because ent_max_assign changes as newly inferred sameAs scores are merged.
    if os.path.isdir(mmap_dir):
        shutil.rmtree(mmap_dir)
    os.makedirs(mmap_dir, exist_ok=True)

    src_entity_count = _compact_entity_count(kb_src)
    dst_entity_count = _compact_entity_count(kb_dst)
    id_dtype = np.int64 if max(src_entity_count, dst_entity_count) > np.iinfo(np.int32).max else np.int32
    offset_dtype = np.int64
    score_dtype = np.float32

    src_offsets = np.zeros(src_entity_count + 1, dtype=offset_dtype)
    src_max = np.zeros(src_entity_count, dtype=score_dtype)
    dst_max = np.zeros(dst_entity_count, dtype=score_dtype)

    if not sides.is_id_keyed_mapping(ent_max_assign):
        raise TypeError("Compact entity assignment index requires side/id-keyed ent_max_assign.")
    src_side = sides.graph_side(kb_src, sides.KB1)
    dst_side = sides.graph_side(kb_dst, sides.KB2)

    def scan_assignments(write_arrays=False, cursor=None, dst_ids=None, scores=None):
        """Scan ent_max_assign."""
        for entity_key, target_scores in ent_max_assign.items():
            if not target_scores:
                continue
            side, entity_id = entity_key
            max_score = max(target_scores.values())
            if side == src_side:
                if 0 <= entity_id < src_entity_count:
                    src_max[entity_id] = max(src_max[entity_id], max_score)
                for target_key, score in target_scores.items():
                    if not sides.is_side_key(target_key, dst_side):
                        continue
                    target_id = target_key[1]
                    if not (0 <= entity_id < src_entity_count and 0 <= target_id < dst_entity_count):
                        continue
                    if write_arrays:
                        index = cursor[entity_id]
                        dst_ids[index] = target_id
                        scores[index] = score
                        cursor[entity_id] += 1
                    else:
                        src_offsets[entity_id + 1] += 1
            elif side == dst_side and 0 <= entity_id < dst_entity_count:
                dst_max[entity_id] = max(dst_max[entity_id], max_score)

    # First pass: count outgoing source->target pairs and record max-score arrays.
    scan_assignments(write_arrays=False)
    # Convert per-source pair counts into CSR offsets. The final offset is the
    # total number of source->target score pairs that need storage.
    np.cumsum(src_offsets, out=src_offsets)
    pair_count = int(src_offsets[-1])

    paths = {
        'src_offsets_path': os.path.join(mmap_dir, 'src_offsets.npy'),
        'dst_ids_path': os.path.join(mmap_dir, 'dst_ids.npy'),
        'scores_path': os.path.join(mmap_dir, 'scores.npy'),
        'src_max_path': os.path.join(mmap_dir, 'src_max.npy'),
        'dst_max_path': os.path.join(mmap_dir, 'dst_max.npy'),
    }
    array_bytes = 0

    def write_array(path, array):
        nonlocal array_bytes
        mmap_array = np.lib.format.open_memmap(path, mode='w+', dtype=array.dtype, shape=array.shape)
        mmap_array[:] = array
        mmap_array.flush()
        array_bytes += int(array.nbytes)

    write_array(paths['src_offsets_path'], src_offsets)
    write_array(paths['src_max_path'], src_max)
    write_array(paths['dst_max_path'], dst_max)

    dst_ids = np.lib.format.open_memmap(paths['dst_ids_path'], mode='w+', dtype=id_dtype, shape=(pair_count,))
    scores = np.lib.format.open_memmap(paths['scores_path'], mode='w+', dtype=score_dtype, shape=(pair_count,))
    cursor = src_offsets[:-1].copy()
    # Second pass: cursor advances within each source slice while filling sparse target IDs and scores.
    scan_assignments(write_arrays=True, cursor=cursor, dst_ids=dst_ids, scores=scores)
    dst_ids.flush()
    scores.flush()
    array_bytes += int(dst_ids.nbytes) + int(scores.nbytes)

    metadata = {
        **paths,
        'pair_count': pair_count,
        'src_entity_count': src_entity_count,
        'dst_entity_count': dst_entity_count,
        'array_bytes': array_bytes,
    }
    logging.info(
        "Compact entity assignment array index built | pairs=%s | src_entities=%s | "
        "dst_entities=%s | array_bytes=%.2fMB | dir=%s",
        pair_count, src_entity_count,
        dst_entity_count, array_bytes / (1024 * 1024), mmap_dir,
    )
    return metadata


#################################################################
#                    Entity Matching Workers                    #
#################################################################

def _match_entities_by_rules_compact(kb_src, kb_dst, quasiEqvirel, queue,
                                     ent_match_tuple_queue, ent_max_assign,
                                     functionalities, params, profile_queue=None):
    """
    Match entities using ID-level CompactGraph rules.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base.
    kb_dst : Graph
        The target knowledge base.
    quasiEqvirel : dict
        Nested dictionary of quasi equivalence scores between predicates.
    queue : mp.Queue
        Multiprocessing queue containing chunks of source entities to process.
    ent_match_tuple_queue : mp.Queue
        Multiprocessing queue receiving nested entity alignment score chunks.
    ent_max_assign : dict
        Current bilateral max assignment of entities.
    functionalities : dict
        Predicate and predicate-group functionality scores.
    params : dict
        Runtime parameters, including gramN, pruning thresholds, and the
        shared compact entity assignment index metadata.
    profile_queue : mp.Queue | None, optional
        Optional queue receiving worker timing and pruning statistics.

    Returns
    -------
    None
        Match scores and optional profile data are emitted through queues.
    """
    ent_match_scores = dict()
    pending_pairs = 0
    profile_stats = {
        'build_facts': 0.0, # seconds spent finding and ranking usable source facts
        'collect_pairs': 0.0, # seconds spent collecting target evidence pairs
        'select': 0.0, # seconds spent selecting target entity candidates
        'align': 0.0, # seconds spent applying entity alignment rules
        'entities_seen': 0, # source entities read from the task queue
        'entities_processed': 0, # non-literal, non-finalized source entities processed
        'skipped_matched': 0, # source entities skipped because they already match strongly
        'fact_upper_bound_pruned_facts': 0, # source facts skipped by best-case score bound
        'target_score_upper_bound_pruned_predicates': 0, # target predicates skipped by score bound
        'target_score_upper_bound_pruned_candidates': 0, # target object candidates losing all predicates to score bound
        'upper_bound_pruned_evidence': 0, # evidence pairs skipped by best-case score bound
        'upper_bound_pruned_rules': 0, # final rule applications skipped by best-case score bound
        'target_hub_functionality_pruned_predicates': 0, # target predicates skipped for high degree
        'target_hub_functionality_pruned_candidates': 0, # target object candidates losing all predicates to high degree
        'total_candidate_obj2': 0, # target object candidates considered across entities
        'total_scanned_evi2': 0, # target evidence triples scanned across entities
        'max_scanned_evi2_entity': 0, # largest scanned target evidence count for one entity
        'max_context_pairs_entity': 0, # largest retained evidence-pair count for one entity
        'max_align_time_entity_s': 0.0, # longest rule-application time for one entity
    }
    # bilateral max assignment
    entity_assign_metadata = params.get('_compact_entity_assign_index')
    if entity_assign_metadata is None:
        raise RuntimeError("Compact entity matching requires shared entity assignment arrays")
    entity_assign_index = CompactEntityAssignIndex(entity_assign_metadata)
    # quasi predicate scores
    quasi_scores, quasi_predicates, max_pred_scores = sides.encode_compact_predicate_scores(
        kb_src,
        kb_dst,
        quasiEqvirel,
    )
    quasi_predicate_sets = {
        predicate_id: frozenset(predicate_ids)
        for predicate_id, predicate_ids in quasi_predicates.items()
    }
    # predicate functionalities
    (   src_functionalities,
        src_group_functionalities,
        dst_functionalities,
        dst_group_functionalities,
    ) = sides.resolve_compact_worker_functionalities(
        kb_src,
        kb_dst,
        functionalities,
    )
    evidence_upper_bound_enabled = params.get('evidence_upper_bound', True)
    minimum_output_score = max(0.0, float(params.get('prune_min_score', 0.0) or 0.0))
    target_hub_degree_threshold = params.get('target_hub_degree_threshold', 10000)
    target_hub_degree_threshold = (target_hub_degree_threshold if target_hub_degree_threshold and target_hub_degree_threshold > 0 else None)

    def compact_group_functionality(group_cache, single_predicate_cache, predicate_ids):
        """Get the functionality score for a group of predicates using the group cache."""
        if len(predicate_ids) == 1:
            return single_predicate_cache.get(predicate_ids[0], 1.0)
        return group_cache.get(predicate_ids, 1.0)

    def max_ent_score_src(entity_id):
        """Get the maximum score of any target entity aligned to this source entity."""
        return entity_assign_index.max_src(entity_id)

    def max_ent_score_dst(entity_id):
        """Get the maximum score of any source entity aligned to this target entity."""
        return entity_assign_index.max_dst(entity_id)

    def has_ent_score_src(entity_id):
        """Check if this source entity has any aligned target entities with nonzero score."""
        return entity_assign_index.has_src_score(entity_id)

    def has_ent_score_dst(entity_id):
        """Check if this target entity has any aligned source entities with nonzero score."""
        return entity_assign_index.has_dst_score(entity_id)

    def active_pair_score(src_id, dst_id):
        """Get the current score of a source-target entity pair from the bilateral max assignment."""
        return entity_assign_index.score(src_id, dst_id)

    def active_target_scores(src_id):
        """Get all target entities and their scores currently aligned to this source entity."""
        return entity_assign_index.target_items(src_id)

    def max_pred_score(predicate_id):
        """Get the maximum quasi equivalence score for this predicate across all target predicates."""
        return max_pred_scores.get(predicate_id, 0)

    def current_pair_score(subj1_id, subj2_id):
        """Get the current best score of a source-target entity pair."""
        # alignment scores generated by this worker in current iteration
        generated_scores = ent_match_scores.get(subj1_id) 
        generated_score = 0.0 if generated_scores is None else generated_scores.get(subj2_id, 0.0)
        # alignment scores from the bilateral max assignment
        active_score = active_pair_score(subj1_id, subj2_id)
        return max(generated_score, active_score)

    def rule_score_floor(subj1_id, subj2_id):
        """Get the minimum score floor for a source-target entity pair based on current scores and max entity scores."""
        return max(
            minimum_output_score,
            current_pair_score(subj1_id, subj2_id),
            max_ent_score_src(subj1_id),
            max_ent_score_dst(subj2_id),
        )

    def rule_can_survive(subj1_id, subj2_id, upper_bound):
        """Check if a score can survive the upper bound score pruning."""
        return upper_bound > rule_score_floor(subj1_id, subj2_id)

    def best_case_hmean_with_score(score, max_count):
        """Compute the best-case harmonic mean of a score with a maximum count of evidence."""
        if score <= 0:
            return 0.0
        if score >= 1.0:
            return 1.0
        max_count = max(1, int(max_count))
        return max_count / ((1.0 / score) + max_count - 1) # all other evidence is perfect (score=1.0)

    def fact_can_survive_upper_bound(fact_kb1, quasi_score_map, obj_kb2_scores):
        """Check if a source fact can survive the upper bound score pruning considering the best object and predicate scores."""
        if not evidence_upper_bound_enabled:
            return True
        if not quasi_score_map or not obj_kb2_scores:
            return False

        best_obj_score = max((score for _, score in obj_kb2_scores), default=0.0)
        best_pred_score = max(quasi_score_map.values(), default=0.0)
        if best_obj_score <= 0 or best_pred_score <= 0:
            return False
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        fact_upper_bound = min(
            best_case_hmean_with_score(best_obj_score, max_rule_evidence),
            best_case_hmean_with_score(best_pred_score, max_rule_evidence),
        )
        return fact_upper_bound > max(minimum_output_score, max_ent_score_src(fact_kb1[OBJ]))

    def filter_target_predicates_by_functionality_bound(
            fact_kb1,
            obj_kb2_id,
            obj_score,
            quasi_score_map,
            predicate_ids):
        """Filter target predicates based on functionality and score upper bounds."""
        if not evidence_upper_bound_enabled:
            return tuple(predicate_ids)
        if not hasattr(kb_dst, '_objects_for_subject_predicate_id_count'):
            return tuple(predicate_ids)

        floor = max(minimum_output_score, max_ent_score_src(fact_kb1[OBJ]))
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        obj_bound = best_case_hmean_with_score(obj_score, max_rule_evidence)
        kept_predicates = []
        score_pruned_predicates = 0
        hub_pruned_predicates = 0

        for predicate_id in predicate_ids:
            object_count = kb_dst._objects_for_subject_predicate_id_count(obj_kb2_id, predicate_id)
            if object_count <= 0:
                continue

            pred_score = quasi_score_map[predicate_id]
            pred_bound = best_case_hmean_with_score(pred_score, max_rule_evidence)
            score_upper_bound = min(obj_bound, pred_bound)
            if score_upper_bound <= floor:
                score_pruned_predicates += 1
                continue

            if (target_hub_degree_threshold is not None and object_count >= target_hub_degree_threshold):
                hub_pruned_predicates += 1
                continue

            kept_predicates.append(predicate_id)

        if score_pruned_predicates:
            profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
        if hub_pruned_predicates:
            profile_stats['target_hub_functionality_pruned_predicates'] += hub_pruned_predicates
        if not kept_predicates:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            if hub_pruned_predicates:
                profile_stats['target_hub_functionality_pruned_candidates'] += 1
        return tuple(kept_predicates)

    def iter_dst_evidence(subject_id, predicate_id_set):
        return kb_dst.triplesWithSubjectIdsFiltered(subject_id, predicate_id_set)

    def flush_compact_match_scores():
        nonlocal ent_match_scores, pending_pairs
        if not ent_match_scores:
            ent_match_scores = {}
            pending_pairs = 0
            return
        ent_match_tuple_queue.put(ent_match_scores)
        ent_match_scores = {}
        pending_pairs = 0

    def add_entity_evidence_pair(context, subj2_id, evi2, fact_kb1, score):
        """Add a source-target entity pair with evidence to the context, keeping only the best score for each target entity."""
        subj2_pairs = context['subj2_pairs']
        pair_map = subj2_pairs.get(subj2_id)
        if pair_map is None:
            pair_map = {}
            subj2_pairs[subj2_id] = pair_map
        previous_pair = pair_map.get(evi2)
        if previous_pair is None or score > previous_pair[1]:
            pair_map[evi2] = (fact_kb1, score)

    def collect_fact_pairs_by_adjacency(
            context,
            fact_kb1,
            quasi_score_map,
            quasi_predicate_ids,
            quasi_predicate_id_set,
            obj_kb2_scores):
        """Collect target entity candidates and their evidence for a given source fact."""
        entity_memory_profile = context['entity_memory_profile']
        tmp_subj2_evi2 = {}
        subj2_maxsubrel_score = {}
        for obj_kb2_id, _obj_score in sorted(
            obj_kb2_scores,
            key=lambda item: (-item[1], item[0]),
        ):
            entity_memory_profile['candidate_obj2'] += 1
            aligned_evi2 = []
            maxsubrel_score = 0
            if not kb_dst.has_subject_id(obj_kb2_id):
                continue
            filtered_predicate_ids = filter_target_predicates_by_functionality_bound(
                fact_kb1,
                obj_kb2_id,
                _obj_score,
                quasi_score_map,
                quasi_predicate_ids,
            )
            if not filtered_predicate_ids:
                continue
            evidence_iter = iter_dst_evidence(
                obj_kb2_id,
                quasi_predicate_id_set
                if len(filtered_predicate_ids) == len(quasi_predicate_ids)
                else frozenset(filtered_predicate_ids),
            )
            for evi2_ in evidence_iter:
                if kb_dst.is_literal_id(evi2_[OBJ]):
                    continue
                entity_memory_profile['scanned_evi2'] += 1
                subrel_score = quasi_score_map[evi2_[PRED]]
                if subrel_score > maxsubrel_score:
                    maxsubrel_score = subrel_score
                    aligned_evi2 = [evi2_]
                if subrel_score == maxsubrel_score:
                    aligned_evi2.append(evi2_)

            if len(aligned_evi2) > entity_memory_profile['aligned_evi2_peak']:
                entity_memory_profile['aligned_evi2_peak'] = len(aligned_evi2)
            for evi2 in aligned_evi2:
                subj2_id = evi2[OBJ]
                subrel_score = quasi_score_map[evi2[PRED]]
                if subrel_score > subj2_maxsubrel_score.get(subj2_id, 0):
                    subj2_maxsubrel_score[subj2_id] = subrel_score
                    tmp_subj2_evi2[subj2_id] = evi2

        if len(tmp_subj2_evi2) > entity_memory_profile['tmp_subj2_peak']:
            entity_memory_profile['tmp_subj2_peak'] = len(tmp_subj2_evi2)
        for subj2_id, single_evi2 in tmp_subj2_evi2.items():
            obj_score = active_pair_score(fact_kb1[SUBJ], single_evi2[SUBJ])
            pred_score = quasi_scores[fact_kb1[PRED]][single_evi2[PRED]]
            score = min(obj_score, pred_score)
            max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
            if evidence_upper_bound_enabled:
                evidence_upper_bound = min(
                    best_case_hmean_with_score(obj_score, max_rule_evidence),
                    best_case_hmean_with_score(pred_score, max_rule_evidence),
                )
                if not rule_can_survive(fact_kb1[OBJ], subj2_id, evidence_upper_bound):
                    profile_stats['upper_bound_pruned_evidence'] += 1
                    continue
            add_entity_evidence_pair(context, subj2_id, single_evi2, fact_kb1, score)

    def context_pair_count(context):
        """Count the total number of source-target entity pairs in the context."""
        return sum(len(pair_map) for pair_map in context['subj2_pairs'].values())

    def align_context(context):
        nonlocal pending_pairs
        subj2_pairs = context['subj2_pairs']
        entity_memory_profile = context['entity_memory_profile']

        # Select the target entity candidates with the most evidence pairs, skipping any that already have a strong alignment score.
        stage_start = time.perf_counter()
        subj2_count = dict()
        maxCount = 0
        for subj2, pair_map in subj2_pairs.items():
            if has_ent_score_dst(subj2) and round(max_ent_score_dst(subj2), 1) >= 1.0:
                continue
            cur_count = len(pair_map)
            if cur_count > maxCount: # keep only the candidates with the most evidence pairs
                subj2_count = dict()
                maxCount = cur_count
                subj2_count[subj2] = cur_count
            elif cur_count == maxCount:
                subj2_count[subj2] = cur_count
        select_time = time.perf_counter() - stage_start
        profile_stats['select'] += select_time
        context['timings']['select'] += select_time
        context['selected_candidates'] = len(subj2_count)

        # Align the source entity with each selected target entity candidate using the collected evidence pairs.
        stage_start = time.perf_counter()
        gramN = min(20, maxCount)
        entity_memory_profile['maxCount'] = maxCount
        for subj_kb2_id in subj2_count:
            sorted_pairs = sorted(subj2_pairs[subj_kb2_id].items(), reverse=True, key=lambda item: item[1][1])
            if len(sorted_pairs) > entity_memory_profile['sorted_pairs_peak']:
                entity_memory_profile['sorted_pairs_peak'] = len(sorted_pairs)
            ev2s = [pair[0] for pair in sorted_pairs]
            ev1s = [pair[1][0] for pair in sorted_pairs]

            visited_facts = set()
            for n in range(1, gramN + 1):
                ev1, ev2 = ev1s[:n], ev2s[:n]
                if (tuple(ev1), tuple(ev2)) in visited_facts:
                    continue
                visited_facts.add((tuple(ev1), tuple(ev2)))
                if len(visited_facts) > entity_memory_profile['visited_facts_peak']:
                    entity_memory_profile['visited_facts_peak'] = len(visited_facts)
                obj1_combo, pred1_combo, subj1_combo = zip(*ev1)
                obj2_combo, pred2_combo, subj2_combo = zip(*ev2)
                assert len(set(subj1_combo)) == 1
                assert len(set(subj2_combo)) == 1
                if encode_pattern(obj1_combo) != encode_pattern(obj2_combo):
                    continue
                localfunc1 = kb_src.localFunctionalityIds(obj1_combo, pred1_combo)
                localfunc2 = kb_dst.localFunctionalityIds(obj2_combo, pred2_combo)
                pred1_sort = tuple(sorted(pred1_combo))
                pred2_sort = tuple(sorted(pred2_combo))
                globalfunc1 = compact_group_functionality(
                    src_group_functionalities,
                    src_functionalities,
                    pred1_sort,
                )
                globalfunc2 = compact_group_functionality(
                    dst_group_functionalities,
                    dst_functionalities,
                    pred2_sort,
                )

                obj_eq = fast_hmean(
                    active_pair_score(obj1_combo[i], obj2_combo[i])
                    for i in range(len(obj1_combo))
                )
                pred_eq = fast_hmean(
                    quasi_scores[pred1_combo[i]][pred2_combo[i]]
                    for i in range(len(pred1_combo))
                )

                if n == 1:
                    body_values = (
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        src_functionalities[pred1_combo[0]],
                        dst_functionalities[pred2_combo[0]],
                    )
                else:
                    body_values = (
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        globalfunc1, globalfunc2,
                    )
                rule_score = min(min(body_values), 1.0)
                if not rule_can_survive(subj1_combo[0], subj2_combo[0], rule_score):
                    profile_stats['upper_bound_pruned_rules'] += 1
                    continue
                created_pair = updateScoreMin(
                    ent_match_scores, subj1_combo[0], subj2_combo[0],
                    *body_values,
                )
                if created_pair:
                    pending_pairs += 1
                if pending_pairs >= MATCH_RESULT_FLUSH_PAIRS:
                    flush_compact_match_scores()
        align_time = time.perf_counter() - stage_start
        profile_stats['align'] += align_time
        context['timings']['align'] += align_time

    def finish_context(context):
        log._record_entity_expansion_profile(
            profile_stats,
            context['timings'],
            context['entity_memory_profile'],
            context_pairs=context.get('context_pairs', 0),
        )

    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break

        for subj_kb1 in entity_chunk:
            profile_stats['entities_seen'] += 1
            subj_kb1_id = sides.coerce_entity_id(kb_src, subj_kb1)
            if subj_kb1_id is None:
                continue

            if kb_src.is_literal_id(subj_kb1_id):
                continue

            if has_ent_score_src(subj_kb1_id) and round(max_ent_score_src(subj_kb1_id), 1) >= 1.0:
                profile_stats['skipped_matched'] += 1
                continue
            profile_stats['entities_processed'] += 1
            context = {
                'subj_kb1_id': subj_kb1_id,
                'subj2_pairs': {},
                'timings': {
                    'build_facts': 0.0,
                    'collect_pairs': 0.0,
                    'select': 0.0,
                    'align': 0.0,
                },
                'facts_kept': 0,
                'context_pairs': 0,
                'selected_candidates': 0,
                'entity_memory_profile': {
                    'candidate_obj2': 0,
                    'scanned_evi2': 0,
                    'aligned_evi2_peak': 0,
                    'tmp_subj2_peak': 0,
                    'maxCount': 0,
                    'sorted_pairs_peak': 0,
                    'visited_facts_peak': 0,
                },
            }

            stage_start = time.perf_counter()
            kb1_facts_ordered = []
            for _, predicate_id, object_id in kb_src.triplesWithSubjectIds(subj_kb1_id):
                obj_match_score = max_ent_score_src(object_id)
                inverse_predicate_id = kb_src._inverse_predicate_id(predicate_id)
                if inverse_predicate_id is None:
                    continue
                pred_match_score = max_pred_score(predicate_id)
                if obj_match_score <= 0:
                    continue
                if pred_match_score <= 0:
                    continue
                kb1_facts_ordered.append((object_id, inverse_predicate_id, subj_kb1_id))
            kb1_facts_ordered.sort(
                reverse=True,
                key=lambda x: min(max_ent_score_src(x[SUBJ]), max_pred_score(x[PRED])),
            )
            build_time = time.perf_counter() - stage_start
            profile_stats['build_facts'] += build_time
            context['timings']['build_facts'] += build_time
            context['facts_kept'] = len(kb1_facts_ordered)

            stage_start = time.perf_counter()
            candidate_facts = kb1_facts_ordered[:params['gramN']]
            for fact_kb1 in candidate_facts:
                pred_kb1, obj_kb1 = fact_kb1[PRED], fact_kb1[SUBJ]
                quasi_score_map = quasi_scores.get(pred_kb1)
                if quasi_score_map:
                    quasi_predicate_ids = quasi_predicates[pred_kb1]
                    quasi_predicate_id_set = quasi_predicate_sets[pred_kb1]
                    obj_kb2_scores = tuple(active_target_scores(obj_kb1))
                    if obj_kb2_scores:
                        if not fact_can_survive_upper_bound(
                            fact_kb1,
                            quasi_score_map,
                            obj_kb2_scores,
                        ):
                            profile_stats['fact_upper_bound_pruned_facts'] += 1
                            continue
                        collect_fact_pairs_by_adjacency(
                            context,
                            fact_kb1,
                            quasi_score_map,
                            quasi_predicate_ids,
                            quasi_predicate_id_set,
                            obj_kb2_scores,
                        )
            collect_time = time.perf_counter() - stage_start
            profile_stats['collect_pairs'] += collect_time
            context['timings']['collect_pairs'] += collect_time
            context_pairs = context_pair_count(context)
            context['context_pairs'] = context_pairs
            align_context(context)
            finish_context(context)
            del context

    flush_compact_match_scores()
    flush_compact_match_scores()
    if profile_queue is not None:
        profile_queue.put(profile_stats)
    exit(0)

def _match_entities_by_rules(kb_src, kb_dst, quasiEqvirel, queue, ent_match_tuple_queue, ent_max_assign, functionalities, params, profile_queue=None):
    """ 
    Match entities in parallel using the rules, corresponding to equation (1) in the paper.
    The function consists of two parts: candidate search and entity alignment.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    quasiEqvirel : dict
        Nested dictionary of quasi equivalence relations
    queue : mp.Queue
        A multiprocessing queue containing the entities to be aligned
    ent_match_tuple_queue : mp.Queue
        A multiprocessing queue to store the resulting entity alignment scores
    ent_max_assign : dict
        The bilateral max assignment of entities
    functionalities : dict
        A dictionary mapping predicates to their functionalities
    params : dict
        A dictionary of parameters, including 'gramN' (the maximum n-gram size to consider)
    profile_queue : mp.Queue | None, optional
        Optional queue receiving worker timing and pruning statistics.

    Returns
    -------
    None
        Match scores and optional profile data are emitted through queues.
    """
    # Check if we can use the compact graph for efficient matching
    if _can_use_compact_id_match(kb_src, kb_dst):
        return _match_entities_by_rules_compact(
            kb_src,
            kb_dst,
            quasiEqvirel,
            queue,
            ent_match_tuple_queue,
            ent_max_assign,
            functionalities,
            params,
            profile_queue=profile_queue,
        )

    ent_match_scores = dict()
    pending_pairs = 0
    profile_stats = {
        'build_facts': 0.0, # seconds spent finding and ranking usable source facts
        'collect_pairs': 0.0, # seconds spent collecting target evidence pairs
        'select': 0.0, # seconds spent selecting target entity candidates
        'align': 0.0, # seconds spent applying entity alignment rules
        'entities_seen': 0, # source entities read from the task queue
        'entities_processed': 0, # non-literal, non-finalized source entities processed
        'skipped_matched': 0, # source entities skipped because they already match strongly
        'fact_upper_bound_pruned_facts': 0, # source facts skipped by best-case score bound
        'target_score_upper_bound_pruned_predicates': 0, # target predicates skipped by score bound
        'target_score_upper_bound_pruned_candidates': 0, # target object candidates losing all predicates to score bound
        'upper_bound_pruned_evidence': 0, # evidence pairs skipped by best-case score bound
        'upper_bound_pruned_rules': 0, # final rule applications skipped by best-case score bound
        'target_hub_functionality_pruned_predicates': 0, # target predicates skipped for high degree
        'target_hub_functionality_pruned_candidates': 0, # target object candidates losing all predicates to high degree
        'total_candidate_obj2': 0, # target object candidates considered across entities
        'total_scanned_evi2': 0, # target evidence triples scanned across entities
        'max_scanned_evi2_entity': 0, # largest scanned target evidence count for one entity
        'max_context_pairs_entity': 0, # largest retained evidence-pair count for one entity
        'max_align_time_entity_s': 0.0, # longest rule-application time for one entity
    }
    # quasi predicate scores
    positive_quasi_scores = {
        predicate: {
            target_predicate: score
            for target_predicate, score in target_scores.items()
            if score > 0
        }
        for predicate, target_scores in quasiEqvirel.items()
    }
    max_pred_score_cache = {
        predicate: max(target_scores.values())
        for predicate, target_scores in positive_quasi_scores.items()
        if target_scores
    }
    evidence_upper_bound_enabled = params.get('evidence_upper_bound', True)
    minimum_output_score = max(0.0, float(params.get('prune_min_score', 0.0) or 0.0))
    target_hub_degree_threshold = params.get('target_hub_degree_threshold', 10000)
    target_hub_degree_threshold = (
        target_hub_degree_threshold
        if target_hub_degree_threshold and target_hub_degree_threshold > 0
        else None
    )
    def group_functionality(predicates):
        return functionalities.get(predicates, 1.0)

    max_ent_score_cache = OrderedDict()

    # LRU cache for max entity scores
    def lru_get(cache, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    def lru_put(cache, key, value, maxsize):
        cache[key] = value
        if len(cache) > maxsize:
            cache.popitem(last=False)

    def max_ent_score(entity):
        cached_score = lru_get(max_ent_score_cache, entity)
        if cached_score is not None:
            return cached_score
        score = max(ent_max_assign.get(entity, {None: 0}).values())
        lru_put(max_ent_score_cache, entity, score, MAX_ENT_SCORE_CACHE_SIZE)
        return score

    def max_pred_score(predicate):
        return max_pred_score_cache.get(predicate, 0)

    def current_pair_score(subj1, subj2):
        """Get the current best score of a source-target entity pair."""
        # alignment scores generated by this worker in current iteration
        generated_scores = ent_match_scores.get(subj1)
        generated_score = 0.0 if generated_scores is None else generated_scores.get(subj2, 0.0)
        # alignment scores from the bilateral max assignment
        active_scores = ent_max_assign.get(subj1)
        active_score = 0.0 if active_scores is None else active_scores.get(subj2, 0.0)
        return max(generated_score, active_score)

    def rule_score_floor(subj1, subj2):
        """Get the minimum score floor for a source-target entity pair based on current scores and max entity scores."""
        return max(
            minimum_output_score,
            current_pair_score(subj1, subj2),
            max_ent_score(subj1),
            max_ent_score(subj2),
        )

    def rule_can_survive(subj1, subj2, upper_bound):
        """Check if a score can survive the upper bound score pruning."""
        return upper_bound > rule_score_floor(subj1, subj2)

    def best_case_hmean_with_score(score, max_count):
        """Compute the best-case harmonic mean of a score with a maximum count of evidence."""
        if score <= 0:
            return 0.0
        if score >= 1.0:
            return 1.0
        max_count = max(1, int(max_count))
        return max_count / ((1.0 / score) + max_count - 1) # all other evidence is perfect (score=1.0)

    def fact_can_survive_upper_bound(fact_kb1, quasi_score_map, obj_kb2_scores):
        """Check if a source fact can survive the upper bound score pruning considering the best object and predicate scores."""
        if not evidence_upper_bound_enabled:
            return True
        if not quasi_score_map or not obj_kb2_scores:
            return False

        best_obj_score = max((score for _, score in obj_kb2_scores), default=0.0)
        best_pred_score = max(quasi_score_map.values(), default=0.0)
        if best_obj_score <= 0 or best_pred_score <= 0:
            return False
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        fact_upper_bound = min(
            best_case_hmean_with_score(best_obj_score, max_rule_evidence),
            best_case_hmean_with_score(best_pred_score, max_rule_evidence),
        )
        return fact_upper_bound > max(minimum_output_score, max_ent_score(fact_kb1[OBJ]))

    def target_predicate_object_count(subject, predicate):
        """Get the number of objects for a given subject and predicate in the target knowledge base."""
        objects = kb_dst._objects_for_subject_predicate(subject, predicate)
        return len(objects) if objects else 0

    def filter_target_predicates_by_functionality_bound(fact_kb1, obj_kb2, obj_score, quasi_scores):
        """Filter target predicates based on functionality and score upper bounds."""
        predicates = tuple(quasi_scores)
        if not evidence_upper_bound_enabled:
            return predicates

        floor = max(minimum_output_score, max_ent_score(fact_kb1[OBJ]))
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        obj_bound = best_case_hmean_with_score(obj_score, max_rule_evidence)
        kept_predicates = []
        score_pruned_predicates = 0
        hub_pruned_predicates = 0

        for predicate in predicates:
            object_count = target_predicate_object_count(obj_kb2, predicate)
            if object_count <= 0:
                continue

            pred_score = quasi_scores[predicate]
            pred_bound = best_case_hmean_with_score(pred_score, max_rule_evidence)
            score_upper_bound = min(obj_bound, pred_bound)
            if score_upper_bound <= floor:
                score_pruned_predicates += 1
                continue

            if (
                target_hub_degree_threshold is not None
                and object_count >= target_hub_degree_threshold
            ):
                hub_pruned_predicates += 1
                continue

            kept_predicates.append(predicate)

        if score_pruned_predicates:
            profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
        if hub_pruned_predicates:
            profile_stats['target_hub_functionality_pruned_predicates'] += hub_pruned_predicates
        if not kept_predicates:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            if hub_pruned_predicates:
                profile_stats['target_hub_functionality_pruned_candidates'] += 1
        return tuple(kept_predicates)

    def add_entity_evidence_pair(subj2_pairs, subj2, evi2, fact_kb1, score):
        """Add a source-target entity pair with evidence to the subj2_pairs, keeping only the best score for each target entity."""
        pair_map = subj2_pairs.get(subj2)
        if pair_map is None:
            pair_map = {}
            subj2_pairs[subj2] = pair_map
        previous_pair = pair_map.get(evi2)
        if previous_pair is None or score > previous_pair[1]:
            pair_map[evi2] = (fact_kb1, score)

    def collect_fact_pairs_by_adjacency(
            subj2_pairs,
            fact_kb1,
            quasi_score_map,
            obj_kb2_scores,
            entity_memory_profile):
        """Collect target entity candidates and their evidence for a given source fact."""
        tmp_subj2_evi2 = {}
        subj2_maxsubrel_score = {}

        for obj_kb2, obj_score in sorted(
            obj_kb2_scores,
            key=lambda item: (-item[1], item[0]),
        ):
            entity_memory_profile['candidate_obj2'] += 1
            if not kb_dst.has_subject(obj_kb2):
                continue

            filtered_quasi_predicates = filter_target_predicates_by_functionality_bound(
                fact_kb1,
                obj_kb2,
                obj_score,
                quasi_score_map,
            )
            if not filtered_quasi_predicates:
                continue

            aligned_evi2 = []
            maxsubrel_score = 0
            for evi2_ in kb_dst.triplesWithSubject(obj_kb2, filtered_quasi_predicates):
                entity_memory_profile['scanned_evi2'] += 1
                if utils.isLiteral(evi2_[OBJ]):
                    continue
                subrel_score = quasi_score_map[evi2_[PRED]]
                if subrel_score > maxsubrel_score:
                    maxsubrel_score = subrel_score
                    aligned_evi2 = [evi2_]
                elif subrel_score == maxsubrel_score:
                    aligned_evi2.append(evi2_)

            if len(aligned_evi2) > entity_memory_profile['aligned_evi2_peak']:
                entity_memory_profile['aligned_evi2_peak'] = len(aligned_evi2)

            for evi2 in aligned_evi2:
                subj2 = evi2[OBJ]
                subrel_score = quasi_score_map[evi2[PRED]]
                if subrel_score > subj2_maxsubrel_score.get(subj2, 0):
                    subj2_maxsubrel_score[subj2] = subrel_score
                    tmp_subj2_evi2[subj2] = evi2

        if len(tmp_subj2_evi2) > entity_memory_profile['tmp_subj2_peak']:
            entity_memory_profile['tmp_subj2_peak'] = len(tmp_subj2_evi2)

        for subj2, single_evi2 in tmp_subj2_evi2.items():
            obj_score = ent_max_assign[fact_kb1[SUBJ]][single_evi2[SUBJ]]
            pred_score = positive_quasi_scores[fact_kb1[PRED]][single_evi2[PRED]]
            score = min(obj_score, pred_score)
            max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
            if evidence_upper_bound_enabled:
                evidence_upper_bound = min(
                    best_case_hmean_with_score(obj_score, max_rule_evidence),
                    best_case_hmean_with_score(pred_score, max_rule_evidence),
                )
                if not rule_can_survive(fact_kb1[OBJ], subj2, evidence_upper_bound):
                    profile_stats['upper_bound_pruned_evidence'] += 1
                    continue
            add_entity_evidence_pair(subj2_pairs, subj2, single_evi2, fact_kb1, score)

    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break

        for subj_kb1 in entity_chunk:
            profile_stats['entities_seen'] += 1

            # We don't need to match literals
            if utils.isLiteral(subj_kb1):
                continue

            # Skip if the entity is already matched
            if subj_kb1 in ent_max_assign and round(max_ent_score(subj_kb1), 1) >= 1.0:
                profile_stats['skipped_matched'] += 1
                continue
            profile_stats['entities_processed'] += 1
            entity_memory_profile = {
                'candidate_obj2': 0,
                'scanned_evi2': 0,
                'aligned_evi2_peak': 0,
                'tmp_subj2_peak': 0,
                'maxCount': 0,
                'sorted_pairs_peak': 0,
                'visited_facts_peak': 0,
            }
            entity_timings = {'build_facts': 0.0, 'collect_pairs': 0.0, 'select': 0.0, 'align': 0.0}

            # Candidate search: rank usable source facts by current alignment strength.
            stage_start = time.perf_counter()
            kb1_facts_ordered = []
            for fact1 in kb_src.triplesWithSubject(subj_kb1):
                obj_match_score = max_ent_score(fact1[OBJ])
                pred_match_score = max_pred_score(fact1[PRED])
                if obj_match_score <= 0:
                    continue
                if pred_match_score <= 0:
                    continue
                kb1_facts_ordered.append((fact1[OBJ], utils.invert(fact1[PRED]), subj_kb1))
            kb1_facts_ordered.sort(
                reverse=True,
                key=lambda x: min(max_ent_score(x[SUBJ]), max_pred_score(x[PRED])),
            )
            build_time = time.perf_counter() - stage_start
            profile_stats['build_facts'] += build_time
            entity_timings['build_facts'] += build_time

            # {subj2: {evi2: (best_evi1, pair_score)}}.
            # Using dictionaries avoids repeated linear scans over evidence lists.
            stage_start = time.perf_counter()
            subj2_pairs = dict()
            candidate_facts = kb1_facts_ordered[:params['gramN']]
            for fact_kb1 in candidate_facts:
                pred_kb1, obj_kb1 = fact_kb1[PRED], fact_kb1[SUBJ]
                quasi_score_map = positive_quasi_scores.get(pred_kb1)
                if not quasi_score_map:
                    continue
                obj_kb2_scores = tuple(ent_max_assign[obj_kb1].items())
                if not fact_can_survive_upper_bound(
                    fact_kb1,
                    quasi_score_map,
                    obj_kb2_scores,
                ):
                    profile_stats['fact_upper_bound_pruned_facts'] += 1
                    continue
                collect_fact_pairs_by_adjacency(
                    subj2_pairs,
                    fact_kb1,
                    quasi_score_map,
                    obj_kb2_scores,
                    entity_memory_profile,
                )
            collect_time = time.perf_counter() - stage_start
            profile_stats['collect_pairs'] += collect_time
            entity_timings['collect_pairs'] += collect_time
            context_pairs = sum(len(pair_map) for pair_map in subj2_pairs.values())

            # Selection Algorithm
            # select the entities with the most evidences
            stage_start = time.perf_counter()
            subj2_count = dict()
            maxCount = 0
            for subj2, pair_map in subj2_pairs.items():
                if subj2 in ent_max_assign and round(max_ent_score(subj2), 1) >= 1.0:
                    continue
                cur_count = len(pair_map)
                if cur_count > maxCount:
                    subj2_count = dict()
                    maxCount = cur_count
                    subj2_count[subj2] = cur_count
                elif cur_count == maxCount:
                    subj2_count[subj2] = cur_count
            select_time = time.perf_counter() - stage_start
            profile_stats['select'] += select_time
            entity_timings['select'] += select_time

            # Alignment Algorithm
            # Apply rules in order to update the scores
            stage_start = time.perf_counter()
            gramN = min(20, maxCount)
            entity_memory_profile['maxCount'] = maxCount
            for subj_kb2 in subj2_count:
                sorted_pairs = sorted(
                    subj2_pairs[subj_kb2].items(),
                    reverse=True,
                    key=lambda item: item[1][1],
                )
                if len(sorted_pairs) > entity_memory_profile['sorted_pairs_peak']:
                    entity_memory_profile['sorted_pairs_peak'] = len(sorted_pairs)
                ev2s = [pair[0] for pair in sorted_pairs]
                ev1s = [pair[1][0] for pair in sorted_pairs]

                # find the common patterns
                visited_facts = set()
                # Try all possible sets
                for n in range(1, gramN+1):
                    ev1, ev2 = ev1s[:n], ev2s[:n]
                    if (tuple(ev1), tuple(ev2)) in visited_facts:
                            continue
                    visited_facts.add((tuple(ev1), tuple(ev2)))
                    if len(visited_facts) > entity_memory_profile['visited_facts_peak']:
                        entity_memory_profile['visited_facts_peak'] = len(visited_facts)
                    obj1_combo, pred1_combo, subj1_combo = zip(*ev1)
                    obj2_combo, pred2_combo, subj2_combo = zip(*ev2)
                    # check if subjects itself are the same
                    assert len(set(subj1_combo)) == 1
                    assert len(set(subj2_combo)) == 1
                    # check same pattern
                    if encode_pattern(obj1_combo) != encode_pattern(obj2_combo):
                        continue
                    localfunc1 = kb_src.localFunctionality(obj1_combo, pred1_combo)
                    localfunc2 = kb_dst.localFunctionality(obj2_combo, pred2_combo)
                    pred1_sort = tuple(sorted(list(pred1_combo)))
                    pred2_sort = tuple(sorted(list(pred2_combo)))
                    globalfunc1 = group_functionality(pred1_sort)
                    globalfunc2 = group_functionality(pred2_sort)

                    obj_eq = fast_hmean(
                        ent_max_assign[obj1_combo[i]][obj2_combo[i]]
                        for i in range(len(obj1_combo))
                    )
                    pred_eq = fast_hmean(
                        quasiEqvirel[pred1_combo[i]][pred2_combo[i]]
                        for i in range(len(pred1_combo))
                    )
                    # update
                    if n == 1:
                        body_values = (
                            obj_eq, pred_eq, localfunc1, localfunc2,
                            functionalities[pred1_combo[0]],
                            functionalities[pred2_combo[0]],
                        )
                    else:
                        body_values = (
                            obj_eq, pred_eq, localfunc1, localfunc2,
                            globalfunc1, globalfunc2,
                        )
                    rule_score = min(min(body_values), 1.0)
                    if not rule_can_survive(subj1_combo[0], subj2_combo[0], rule_score):
                        profile_stats['upper_bound_pruned_rules'] += 1
                        continue
                    created_pair = updateScoreMin(
                        ent_match_scores, subj1_combo[0], subj2_combo[0],
                        *body_values,
                    )
                    if created_pair:
                        pending_pairs += 1
                    if pending_pairs >= MATCH_RESULT_FLUSH_PAIRS:
                        ent_match_scores, pending_pairs = _flush_match_scores(
                            ent_match_tuple_queue,
                            ent_match_scores,
                        )
            align_time = time.perf_counter() - stage_start
            profile_stats['align'] += align_time
            entity_timings['align'] += align_time
            log._record_entity_expansion_profile(
                profile_stats,
                entity_timings,
                entity_memory_profile,
                context_pairs=context_pairs,
            )
        ent_match_scores, pending_pairs = _flush_match_scores(
            ent_match_tuple_queue,
            ent_match_scores,
        )
    ent_match_scores, pending_pairs = _flush_match_scores(ent_match_tuple_queue, ent_match_scores)
    if profile_queue is not None:
        profile_queue.put(profile_stats)
    exit(0)