import Prefixes
import Announce
import time
import gc
import multiprocessing as mp
import argparse
import logging
import os
import sys
import shutil
import pickle
import align
import cache
import log
import sides


# hyperparameters
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass

def get_params():
    parser = argparse.ArgumentParser(
        usage=argparse.SUPPRESS,
        description="""\
        
        FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic

        Usage: 
        There are two different ways of calling FLORA.

        1) For Custom KGs, please provide the input KGs explicitly through --kg1, --kg2, for example:
            python main.py --kg1 ../data/kg1.ttl --kg2 ../data/kg2.ttl --embedding ../data/emb/ --output results.ttl

        2) For benchmark datasets, please use --dataset parameter, for example:
            python main.py --dataset OpenEA/D_W_15K_V2/ --embedding emb/D_W_15K_V2/ --alpha 3.0 --init 0.7 --output dw-v2.ttl

        To quickly test the code, you can use the small-test dataset:
            python main.py --dataset small-test/mini/ --embedding emb/mini/ --output mini-test.ttl
        """,
        formatter_class=CustomFormatter
    )

    # Data source and run artifacts: choose either a benchmark dataset or custom
    # KG files, then configure where auxiliary inputs and final output live.
    io_group = parser.add_argument_group('Input and output')
    io_group.add_argument('--dataset', type=str, default=None, help='Benchmark dataset under ../data, e.g., OpenEA/D_W_15K_V2/')
    io_group.add_argument('--kg1', type=str, default='../data/source.ttl', help='Custom source Turtle file used as KG1')
    io_group.add_argument('--kg2', type=str, default='../data/target.ttl', help='Custom target Turtle file used as KG2')
    io_group.add_argument('--embedding', type=str, default=None, help='Literal embedding folder for the two KGs, e.g., emb/D_W_15K_V2/')
    io_group.add_argument('--literal_scores', type=str, default=None, help='Precomputed literal sameAs score pickle from init.py, e.g., emb/D_W_15K_V2/literal_scores.pkl')
    io_group.add_argument('--trainingdata', type=str, default=None, help='Optional seed alignment file under ../data')
    io_group.add_argument('--output', type=str, default='results.ttl', help='Output file name under ../save')
    # Literal initialization
    literal_group = parser.add_argument_group('Literal initialization')
    literal_group.add_argument('--init', type=float, default=0.7, help='Initial literal similarity threshold')
    literal_group.add_argument('--string_identity', action='store_true', help='Use only exact string literal identity for initialization; otherwise use literal embedding similarity')
    literal_group.add_argument('--literal_embedding_model', type=str, default='Lihuchen/pearl_small', help='HuggingFace model used when FLORA needs to pre-compute literal embeddings. For multilingual embeddings, use sentence-transformers/LaBSE')
    literal_group.add_argument('--literal_english_filter', action='store_true', help='Keep only English literals during literal initialization')
    literal_group.add_argument('--literal_idf', action='store_true', help='Reweight literal initialization scores with literal IDF to filter out common literals')
    literal_group.add_argument('--literal_faiss_index', choices=['flat', 'hnsw'], default='flat', help='FAISS index for literal embedding search: flat is exact search and can use GPU; hnsw is approximate CPU search')
    literal_group.add_argument('--literal_hnsw_m', type=int, default=32, help='HNSW graph degree for --literal_faiss_index hnsw')
    literal_group.add_argument('--literal_hnsw_ef_search', type=int, default=64, help='HNSW efSearch for --literal_faiss_index hnsw')
    literal_group.add_argument('--literal_hnsw_ef_construction', type=int, default=200, help='HNSW efConstruction for --literal_faiss_index hnsw')
    # Core alignment algorithm
    alignment_group = parser.add_argument_group('Alignment algorithm')
    alignment_group.add_argument('--alpha', type=float, default=3.0, help='Benefit-of-doubt factor for calculating subrelation scores')
    alignment_group.add_argument('--gramN', type=int, default=100, help='Maximum number of evidences to consider for each entity during alignment')
    alignment_group.add_argument('--epsilon', type=float, default=0.01, help='Convergence threshold for stopping the main loop')
    alignment_group.add_argument('--max_iterations', type=int, default=50, help='Maximum number of main-loop iterations to run')
    alignment_group.add_argument('--prune_min_score', type=float, default=0.01, help='Drop sameAs candidates below this score')
    # Performance and memory controls
    performance_group = parser.add_argument_group('Performance and memory')
    performance_group.add_argument('--evidence_upper_bound', type=bool, default=True, help='Enable evidence upper-bound pruning before candidate rule scoring')
    performance_group.add_argument('--target_hub_degree_threshold', type=int, default=10000, help='Apply target-side hub local-functionality pruning when a candidate subject/predicate has at least this many objects; 0 disables hub pruning')
    performance_group.add_argument('--bootstrap_workers', type=int, default=None, help='Worker processes for bootstrap alignment; defaults to --workers when unset')
    performance_group.add_argument('--workers', type=int, default=None, help='Worker processes for iteration-time entity alignment')
    performance_group.add_argument('--subrelation_workers', type=int, default=None, help='Worker processes for predicate subrelation mapping')
    performance_group.add_argument('--compact_kg', action='store_true', help='Store KGs as read-only mmap arrays instead of nested Python dict/set to reduce memory usage for large KGs')
    # Cache and checkpointing
    state_group = parser.add_argument_group('Cache and checkpointing')
    state_group.add_argument('--disable_preprocessing_cache', action='store_true', help='Disable reusable preprocessing caches for KG loading, functionalities, and literal matching')
    state_group.add_argument('--enable_checkpoint', action='store_true', help='Save resumable FLORA checkpoints; disabled by default because checkpoints can be large')
    state_group.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for FLORA checkpoints; defaults to ../save/checkpoints/<output-stem>')
    state_group.add_argument('--checkpoint_interval', type=int, default=1, help='When --enable_checkpoint is set, save every N completed iterations; 0 disables periodic checkpoints')
    state_group.add_argument('--resume_checkpoint', action='store_true', help='Resume from the latest compatible checkpoint in --checkpoint_dir')

    # Show help if no args
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        parser.error("unrecognized arguments: %s" % ' '.join(unknown_args))
    params_ = vars(args)
    return params_



if __name__ == '__main__':
    Announce.doing("Running FLORA...")

    params = get_params()
    Announce.set_logger(params)
    procedure_start = time.time()

    # File paths
    dataset_path = '../data/{a}'.format(a=params['dataset']) if params['dataset'] else None
    training_data_file = '../data/{a}'.format(a=params['trainingdata']) if params['trainingdata'] else None
    if dataset_path is not None:
        emb_path = '../data/{a}'.format(a=params['embedding']) if params['embedding'] else '../data/emb/' # default path
    else:
        emb_path = params['embedding'] if params['embedding'] else '../data/emb/' # default path
    output_path = '../save/{a}'.format(a=params['output'])
    checkpoint_dir = (
        os.path.abspath(params['checkpoint_dir'])
        if params['checkpoint_dir']
        else cache.default_checkpoint_dir(output_path)
    )


    #################################################################
    #                    Loading data                               #
    #################################################################

    # Load knowledge bases
    Announce.doing("Loading Knowledge Bases")
    loading_start = time.time()
    if params['dataset'] is None:
        assert os.path.exists(params['kg1']), "File %s does not exist!" % params['kg1']
        assert os.path.exists(params['kg2']), "File %s does not exist!" % params['kg2']

    kb1, kb2, _ = cache.load_knowledge_bases(
        params,
        dataset_path,
        use_cache=not params['disable_preprocessing_cache'],
    )
    sides.tag_graph_sides(kb1, kb2)
    logging.info("Time used for loading KGs: %s minutes"%(round((time.time() - loading_start)/60, 5)))
    Announce.done()
    # cache and checkpoint information
    kg_cache_info = cache.dataset_cache_info(params, dataset_path)
    checkpoint_signature = cache.checkpoint_signature(params, kg_cache_info)
    # unique tag for the current run
    run_tag = "%s-pid%s-%s" % (
        os.path.splitext(os.path.basename(params['output']))[0],
        os.getpid(),
        time.strftime("%Y%m%d%H%M%S"),
    )
    compact_entity_assign_run_dir = os.path.join(
        kg_cache_info['compact_mmap_dir'],
        'entity_assign_runs',
        run_tag,
    )
    logging.info(
        "Compact entity assignment runtime dir | dir=%s",
        compact_entity_assign_run_dir,
    )

    # Load training data (if any)
    sameAsScores={}
    if training_data_file is not None:
        Announce.doing("Loading training data from %s" % training_data_file)
        with open(training_data_file, "rt", encoding="utf-8") as trainingDataFile:
            for line in trainingDataFile:
                split=line.strip().split("\t")
                if split[0] not in sameAsScores:
                    sameAsScores[split[0]]={}
                sameAsScores[split[0]][split[1]]=1.0 if len(split)<3 else float(split[2])
        sameAsScores = sides.maybe_encode_same_as_scores(sameAsScores, kb1, kb2)
        Announce.done()


    #################################################################
    #            Initialization + Bootstrapping                     #
    #################################################################

    Announce.doing("Initializing Subrelations")
    predicates1 = kb1.predicates()
    predicates2 = kb2.predicates()
    if align._can_use_compact_id_match(kb1, kb2):
        predicate2superPredicate = align.CompactDenseDefaultPredicateMapping(kb1, kb2, relinit=0.1)
    else:
        predicate2superPredicate = align.initializePredicateSubsumption(predicates1, predicates2, relinit=0.1)
    Announce.done()

    compute_functionalities_start = time.time()
    Announce.doing("Computing functionalities")
    functionalities1 = cache.load_or_compute_functionalities(
        kb1,
        kg_cache_info['signature'],
        'kb1',
        [1, 2],
        use_cache=not params['disable_preprocessing_cache'],
    )
    functionalities2 = cache.load_or_compute_functionalities(
        kb2,
        kg_cache_info['signature'],
        'kb2',
        [1, 2],
        use_cache=not params['disable_preprocessing_cache'],
    )
    if sides.can_use_id_keyed_state(kb1, kb2):
        functionalities = sides.preencode_compact_worker_functionalities(functionalities1, functionalities2)
    else:
        functionalities = {}
        for pred in functionalities1:
            functionalities[pred] = functionalities1[pred]
        for pred in functionalities2:
            if pred not in functionalities:
                functionalities[pred] = functionalities2[pred]
                continue
            functionalities[pred] = min(functionalities[pred], functionalities2[pred])
    # release memory
    del functionalities1
    del functionalities2
    gc.collect()
    logging.info("Time used for computing functionalities: %s minutes"%(round((time.time() - compute_functionalities_start)/60, 5)))
    Announce.done()

    BOD = params['alpha']
    iterations = 0
    checkpoint_payload = None
    if params['resume_checkpoint']:
        checkpoint_payload = cache.load_latest_checkpoint(checkpoint_dir, checkpoint_signature)

    if checkpoint_payload is not None: # load checkpoint
        sameAsScores = checkpoint_payload['sameAsScores']
        predicate2superPredicate = checkpoint_payload['predicate2superPredicate']
        quasiEqvirel = checkpoint_payload['quasiEqvirel']
        sameAsScores = sides.maybe_encode_same_as_scores(sameAsScores, kb1, kb2)
        predicate2superPredicate = sides.maybe_encode_predicate_mapping(
            predicate2superPredicate,
            kb1,
            kb2,
            preserve_mapping=align.is_dense_default_predicate_mapping,
        )
        quasiEqvirel = sides.maybe_encode_predicate_mapping(
            quasiEqvirel,
            kb1,
            kb2,
            preserve_mapping=align.is_dense_default_predicate_mapping,
        )
        iterations = checkpoint_payload['iterations']
        logging.info(
            "Resuming main loop from checkpoint | iteration=%s | checkpoint_dir=%s",
            iterations, checkpoint_dir,
        )
        log.log_nested_mapping_stats("Checkpoint sameAsScores", sameAsScores)
        log.log_predicate_mapping_stats("Checkpoint predicate2superPredicate stats", predicate2superPredicate)
        log.log_predicate_mapping_stats("Checkpoint quasiEqvirel stats", quasiEqvirel)
    else:
        # Literal Matching
        Announce.doing(
            "Computing initialization scores | literal_mode=%s | threshold=%s"
            % (
                'identity' if params['string_identity'] else 'embedding similarity',
                params['init'],
            )
        )
        literal_matching_start = time.time()
        if params['literal_scores']: # have precomputed literal scores
            literal_scores_path = os.path.abspath(params['literal_scores'])
            Announce.doing("Loading precomputed literal scores from %s" % literal_scores_path)
            with open(literal_scores_path, "rb") as literal_scores_file:
                literal_scores = pickle.load(literal_scores_file)
            logging.info(
                "Loaded precomputed literal scores | path=%s | sources=%s | pairs=%s",
                literal_scores_path, len(literal_scores), sum(len(values) for values in literal_scores.values()),
            )
            Announce.done()
        else:
            # Precompute the literal embeddings if necessary
            if params['string_identity']:
                logging.info("Skipping literal embedding precomputation because literal identity-only matching is enabled")
            elif os.path.exists(os.path.join(emb_path, 'kb1.pkl')) and os.path.exists(os.path.join(emb_path, 'kb2.pkl')):
                Announce.doing("Loading precomputed literal embeddings from %s" % emb_path)
                Announce.done()
            else:
                import literals
                Announce.doing("PreComputing literal embeddings with %s..." % params['literal_embedding_model'])
                literals.compute_literal_embeddings(
                    kb1,
                    kb2,
                    emb_path,
                    embedding_model=params['literal_embedding_model'],
                )
                Announce.done()
            literal_scores = cache.load_or_compute_literal_scores(
                kb1,
                kb2,
                emb_path,
                params,
                kg_cache_info['signature'],
                use_cache=not params['disable_preprocessing_cache'],
            )
        literal_scores = sides.maybe_encode_same_as_scores(literal_scores, kb1, kb2)
        if sameAsScores:
            for entity1, entity2scores in literal_scores.items():
                if entity1 not in sameAsScores:
                    sameAsScores[entity1] = {}
                for entity2, score in entity2scores.items():
                    if score > sameAsScores[entity1].get(entity2, 0):
                        sameAsScores[entity1][entity2] = score
        else:
            sameAsScores = literal_scores
        del literal_scores # release memory
        gc.collect()
        logging.info("Time used for initialization matching: %s minutes"%(round((time.time() - literal_matching_start)/60, 5)))
        Announce.done()


        # Bootstrap entity alignment from the configured initialization scores.
        starttime = time.time()
        Announce.doing("Bootstrapping")
        log.log_nested_mapping_stats("Bootstrap pre-state sameAsScores", sameAsScores)
        bootstrap_workers = max(1, params['bootstrap_workers'] or mp.cpu_count())
        round_prefix = "Bootstrap"
        ent_maxAssign = None
        quasiEqvirel = None
        if sameAsScores:
            align.bootstrap_algo(
                kb1,
                kb2,
                sameAsScores,
                predicate2superPredicate,
                functionalities,
                min_score=params['prune_min_score'],
                num_workers=bootstrap_workers,
            )
        else:
            logging.info("Skipping %s worker-stage because literal bootstrap sameAsScores is empty", round_prefix)
        log.log_nested_mapping_stats("%s post-bootstrap sameAsScores" % round_prefix, sameAsScores)
        log.log_memory_snapshot("%s post-bootstrap memory" % round_prefix)
        log.log_nested_mapping_stats("%s post-prune sameAsScores" % round_prefix, sameAsScores)
        log.log_memory_snapshot("%s post-prune memory" % round_prefix)
        ent_maxAssign = align.bilateral_max_assign(sameAsScores)
        log.log_nested_mapping_stats("%s bilateral_max_assign stats" % round_prefix, ent_maxAssign)
        log.log_alignment_fanout("%s bilateral_max_assign" % round_prefix, ent_maxAssign, kb1, kb2)
        log.log_memory_snapshot("%s post-bilateral memory" % round_prefix)
        logging.info("%s bilateral max assign successfully computed.", round_prefix)

        if align._can_use_compact_id_match(kb1, kb2):
            predicate2superPredicate = align.initializePredicateIdentityOnlyIds(kb1, kb2)
        else:
            predicate2superPredicate = align.initializePredicateIdentityOnly(predicates1, predicates2)
        align.map_subrelations(
            BOD,
            kb1,
            kb2,
            ent_maxAssign,
            predicate2superPredicate,
            stage_label="%s subrelation-stage" % round_prefix,
            num_workers=max(1, params['subrelation_workers'] or mp.cpu_count()),
        )
        log.log_predicate_mapping_stats("%s predicate2superPredicate stats" % round_prefix, predicate2superPredicate)
        log.log_memory_snapshot("%s post-map_subrelations memory" % round_prefix)
        logging.info("%s subrelation mapping successfully computed.", round_prefix)
        quasiEqvirel = align.computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
        log.log_predicate_mapping_stats("%s quasiEqvirel stats" % round_prefix, quasiEqvirel)
        log.log_memory_snapshot("%s post-quasiEqvirel memory" % round_prefix)
        logging.info("%s quasi-equivalence relation successfully computed.", round_prefix)
        Announce.done()
        logging.info("Time used for bootstrapping: %s minutes"%(round((time.time() - starttime)/60, 5)))
        if params['enable_checkpoint'] and params['checkpoint_interval'] != 0:
            cache.save_checkpoint(
                checkpoint_dir,
                checkpoint_signature,
                iterations,
                sameAsScores,
                predicate2superPredicate,
                quasiEqvirel,
            )
    logging.info("---------------Main Loop---------------")


    #################################################################
    #                         Main Loop                             #
    #################################################################


    max_iterations = max(0, params.get('max_iterations', 100))
    logging.info("Main loop max_iterations=%s | epsilon=%s", max_iterations, params['epsilon'])
    while iterations < max_iterations:
        Announce.doing("Iteration", iterations+1)
        logging.info("----Iteration %s----" % (iterations + 1))
        
        sameAsSum=sum(val for dict_ in sameAsScores.values() for val in dict_.values())

        # Entity Alignment
        Announce.doing("Applying the Entity Alignment rules")
        starttime1 = time.time()
        worker_stage_start = time.perf_counter()
        log.log_nested_mapping_stats("Iteration pre-worker sameAsScores", sameAsScores)
        log.log_alignment_fanout(
            f"Iteration {iterations + 1} pre-worker sameAsScores",
            sameAsScores,
            kb1,
            kb2,
            min_targets=20,
        )
        log.log_memory_snapshot("Iteration pre-worker memory")
        ent_maxAssign = align.bilateral_max_assign(sameAsScores)
        log.log_nested_mapping_stats("Iteration pre-worker ent_maxAssign", ent_maxAssign)
        log.log_alignment_fanout(f"Iteration {iterations + 1} pre-worker ent_maxAssign", ent_maxAssign, kb1, kb2)
        log.log_memory_snapshot("Iteration post-bilateral memory")
        use_compact_id_score_chunks = align._can_use_compact_id_match(kb1, kb2)
        worker_params = params
        worker_ent_max_assign = ent_maxAssign
        merge_guard_ent_max_assign = ent_maxAssign
        merge_guard_src_max_scores = None
        merge_guard_dst_max_scores = None
        if use_compact_id_score_chunks:
            # Compact KG workers share the current max assignment through mmap
            # arrays instead of receiving a large nested Python dict.
            worker_params = dict(params)
            assign_index_dir = os.path.join(
                compact_entity_assign_run_dir,
                f"iter_{iterations + 1:04d}",
            )
            entity_assign_metadata = align.build_compact_entity_assign_index(
                kb1,
                kb2,
                ent_maxAssign,
                assign_index_dir,
            )
            worker_params['_compact_entity_assign_index'] = entity_assign_metadata
            # Keep only the compact max-score guards needed by the parent merge.
            merge_guard_index = align.CompactEntityAssignIndex(entity_assign_metadata)
            merge_guard_src_max_scores = merge_guard_index.src_max
            merge_guard_dst_max_scores = merge_guard_index.dst_max
            worker_ent_max_assign = None
            ent_maxAssign = None
            merge_guard_ent_max_assign = None
            gc.collect()
            logging.info(
                "Iteration %s worker score transport | mode=compact-id-chunks | "
                "entity_assign=shared-array",
                iterations + 1,
            )
        tasks = []
        ent_queue = None
        ent_match_tuple_queue = None
        profile_queue = None
        try:
            log.prepare_for_worker_fork(f"Iteration {iterations + 1} worker-stage")
            num_workers = max(1, params['workers'] or mp.cpu_count())
            logging.info("Iteration %s worker config | workers=%s", iterations + 1, num_workers)
            queue_size = max(4, num_workers * 2)
            ent_queue = mp.Queue(maxsize=queue_size)
            ent_match_tuple_queue = mp.Queue(maxsize=queue_size)
            profile_queue = mp.Queue()
            merge_guard_score_cache = {}

            def merge_worker_chunk(ent_match_score_dict):
                if use_compact_id_score_chunks:
                    align.merge_same_as_score_chunk_ids(
                        sameAsScores,
                        ent_match_score_dict,
                        kb1,
                        kb2,
                        min_score=params['prune_min_score'],
                        src_max_scores=merge_guard_src_max_scores,
                        dst_max_scores=merge_guard_dst_max_scores,
                    )
                else:
                    align.merge_same_as_score_chunk(
                        sameAsScores,
                        ent_match_score_dict,
                        min_score=params['prune_min_score'],
                        ent_max_assign=merge_guard_ent_max_assign,
                        ent_max_score_cache=merge_guard_score_cache,
                    )

            for _ in range(num_workers):
                worker_args = (
                    kb1, kb2,
                    quasiEqvirel,
                    ent_queue,
                    ent_match_tuple_queue,
                    worker_ent_max_assign,
                    functionalities,
                    worker_params,
                    profile_queue,
                    )
                task = mp.Process(
                    target=log.run_worker_with_crash_logging,
                    args=(
                        align._match_entities_by_rules,
                        worker_args,
                        profile_queue,
                        f"Iteration {iterations + 1} worker-stage",
                    ),
                )
                task.start()
                tasks.append(task)
            memory_tracker = log.ProcessMemoryPeakTracker(
                f"Iteration {iterations + 1} worker-stage",
                [task.pid for task in tasks],
            )
            memory_tracker.log_current(f"Iteration {iterations + 1} worker-stage memory start")
            worker_profiles = []

            align.feed_entity_chunks(
                ent_queue,
                kb1,
                num_workers,
                result_queue=ent_match_tuple_queue,
                merge_fn=merge_worker_chunk,
                stage_label=f"Iteration {iterations + 1} worker-stage",
                memory_tracker=memory_tracker,
                side_queues=[
                    ('profile_chunks', profile_queue, worker_profiles.append),
                ],
                worker_tasks=tasks,
            )
            align.wait_for_workers_and_drain(
                tasks,
                ent_match_tuple_queue,
                merge_worker_chunk,
                stage_label=f"Iteration {iterations + 1} worker-stage",
                memory_tracker=memory_tracker,
                side_queues=[
                    ('profile_chunks', profile_queue, worker_profiles.append),
                ],
            )

            logging.info("All worker processes finished. Merging results...")
            merge_stage_start = time.perf_counter()
            align.drain_queue_items(ent_match_tuple_queue, merge_worker_chunk)
            merge_stage_time = time.perf_counter() - merge_stage_start
            logging.info("Merging worker results successfully completed.")

            align.drain_queue_items(profile_queue, worker_profiles.append)
        finally:
            if ent_queue is not None:
                ent_queue.close()
                ent_queue.join_thread()
            if ent_match_tuple_queue is not None:
                ent_match_tuple_queue.close()
                ent_match_tuple_queue.join_thread()
            if profile_queue is not None:
                profile_queue.close()
                profile_queue.join_thread()
        worker_stage_time = time.perf_counter() - worker_stage_start
        Announce.done()
        logging.info("Aligning entities: %s minutes"%(round((time.time() - starttime1)/60, 5)))
        log.log_nested_mapping_stats("Iteration post-worker sameAsScores", sameAsScores)
        log.log_memory_snapshot("Iteration post-worker memory")
        worker_crashes = [
            item for item in worker_profiles
            if isinstance(item, dict) and item.get('_worker_crash')
        ]
        for crash in worker_crashes:
            logging.error(
                "Worker crash report | stage=%s | pid=%s | exception=%s: %s\n%s",
                crash.get('stage_label'),
                crash.get('pid'),
                crash.get('exception_type'),
                crash.get('exception'),
                crash.get('traceback'),
            )
        worker_profiles = [
            item for item in worker_profiles
            if isinstance(item, dict) and not item.get('_worker_crash')
        ]
        if worker_profiles:
            total_build_facts = sum(item['build_facts'] for item in worker_profiles)
            total_collect_pairs = sum(item['collect_pairs'] for item in worker_profiles)
            total_select = sum(item['select'] for item in worker_profiles)
            total_align = sum(item['align'] for item in worker_profiles)
            total_seen = sum(item['entities_seen'] for item in worker_profiles)
            total_processed = sum(item['entities_processed'] for item in worker_profiles)
            total_skipped_matched = sum(item['skipped_matched'] for item in worker_profiles)
            total_candidate_obj2 = sum(item.get('total_candidate_obj2', 0) for item in worker_profiles)
            total_scanned_evi2 = sum(item.get('total_scanned_evi2', 0) for item in worker_profiles)
            total_fact_upper_bound_pruned_facts = sum(
                item.get('fact_upper_bound_pruned_facts', 0)
                for item in worker_profiles
            )
            total_upper_bound_pruned_evidence = sum(
                item.get('upper_bound_pruned_evidence', 0)
                for item in worker_profiles
            )
            total_upper_bound_pruned_rules = sum(
                item.get('upper_bound_pruned_rules', 0)
                for item in worker_profiles
            )
            total_target_score_upper_bound_pruned_predicates = sum(
                item.get('target_score_upper_bound_pruned_predicates', 0)
                for item in worker_profiles
            )
            total_target_score_upper_bound_pruned_candidates = sum(
                item.get('target_score_upper_bound_pruned_candidates', 0)
                for item in worker_profiles
            )
            total_target_hub_functionality_pruned_predicates = sum(
                item.get('target_hub_functionality_pruned_predicates', 0)
                for item in worker_profiles
            )
            total_target_hub_functionality_pruned_candidates = sum(
                item.get('target_hub_functionality_pruned_candidates', 0)
                for item in worker_profiles
            )
            max_scanned_evi2_entity = max(
                (item.get('max_scanned_evi2_entity', 0) for item in worker_profiles),
                default=0,
            )
            max_context_pairs_entity = max(
                (item.get('max_context_pairs_entity', 0) for item in worker_profiles),
                default=0,
            )
            max_align_time_entity_s = max(
                (item.get('max_align_time_entity_s', 0.0) for item in worker_profiles),
                default=0.0,
            )
            logging.info(
                "Worker profile | seen=%s | processed=%s | skipped_matched=%s | build_facts=%.3fs | collect_pairs=%.3fs | select=%.3fs | align=%.3fs",
                total_seen,
                total_processed,
                total_skipped_matched,
                total_build_facts,
                total_collect_pairs,
                total_select,
                total_align,
            )
            logging.info(
                "Worker expansion profile | candidate_obj2=%s | scanned_evi2=%s | "
                "max_scanned_evi2_entity=%s | max_context_pairs_entity=%s | "
                "max_align_time_entity_s=%.3fs | "
                "fact_upper_bound_pruned_facts=%s | "
                "upper_bound_pruned_evidence=%s | "
                "upper_bound_pruned_rules=%s | "
                "target_score_upper_bound_pruned_predicates=%s | "
                "target_score_upper_bound_pruned_candidates=%s | "
                "target_hub_functionality_pruned_predicates=%s | "
                "target_hub_functionality_pruned_candidates=%s",
                total_candidate_obj2,
                total_scanned_evi2,
                max_scanned_evi2_entity,
                max_context_pairs_entity,
                max_align_time_entity_s,
                total_fact_upper_bound_pruned_facts,
                total_upper_bound_pruned_evidence,
                total_upper_bound_pruned_rules,
                total_target_score_upper_bound_pruned_predicates,
                total_target_score_upper_bound_pruned_candidates,
                total_target_hub_functionality_pruned_predicates,
                total_target_hub_functionality_pruned_candidates,
            )
        # Predicate Alignment
        Announce.doing("Recomputing predicate inclusions")
        starttime1 = time.time()
        bilateral_stage_start = time.perf_counter()
        ent_maxAssign = align.bilateral_max_assign(sameAsScores)
        bilateral_stage_time = time.perf_counter() - bilateral_stage_start
        log.log_nested_mapping_stats("Iteration predicate-stage ent_maxAssign", ent_maxAssign)
        log.log_alignment_fanout(f"Iteration {iterations + 1} predicate-stage ent_maxAssign", ent_maxAssign, kb1, kb2)
        log.log_memory_snapshot("Iteration predicate-stage post-bilateral memory")
        subrelation_stage_start = time.perf_counter()
        align.map_subrelations(
            BOD,
            kb1,
            kb2,
            ent_maxAssign,
            predicate2superPredicate,
            stage_label=f"Iteration {iterations + 1} subrelation-stage",
            num_workers=max(1, params['subrelation_workers'] or mp.cpu_count()),
        )
        subrelation_stage_time = time.perf_counter() - subrelation_stage_start
        log.log_predicate_mapping_stats("Iteration predicate2superPredicate stats", predicate2superPredicate)
        log.log_memory_snapshot("Iteration predicate-stage post-map_subrelations memory")
        quasi_stage_start = time.perf_counter()
        quasiEqvirel = align.computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
        quasi_stage_time = time.perf_counter() - quasi_stage_start
        log.log_predicate_mapping_stats("Iteration quasiEqvirel stats", quasiEqvirel)
        log.log_memory_snapshot("Iteration predicate-stage post-quasiEqvirel memory")
        Announce.done()
        logging.info("Aligning predicates: %s minutes"%(round((time.time() - starttime1)/60, 5)))
        logging.info(
            "Iteration timing | worker=%.3fs | merge=%.3fs | bilateral=%.3fs | subrelations=%.3fs | quasi=%.3fs",
            worker_stage_time, merge_stage_time, bilateral_stage_time, subrelation_stage_time, quasi_stage_time,
        )
        # Check convergence
        Announce.doing("Checking convergence")
        newSameAsSum=sum(val for dict_ in sameAsScores.values() for val in dict_.values())   
        Announce.done(sameAsSum,newSameAsSum)
        logging.info("SameAs sum: %s -> %s"%(sameAsSum, newSameAsSum))
        
        Announce.done() 
        iterations+=1
        if (
            params['enable_checkpoint']
            and params['checkpoint_interval'] > 0
            and iterations % params['checkpoint_interval'] == 0
        ):
            cache.save_checkpoint(
                checkpoint_dir,
                checkpoint_signature,
                iterations,
                sameAsScores,
                predicate2superPredicate,
                quasiEqvirel,
            )
        if abs(newSameAsSum - sameAsSum) < params['epsilon']:
            logging.info(
                "Stopping after iteration %s due to convergence: delta=%s < epsilon=%s",
                iterations, abs(newSameAsSum - sameAsSum), params['epsilon'],
            )
            break
    else:
        logging.info("Stopping after reaching max_iterations=%s", max_iterations)

    #################################################################
    #                       Write out results                       #
    #################################################################
    Announce.doing("Writing out results")
    with open(output_path, "wt", encoding="utf-8") as out:
        for p in Prefixes.prefixes:
            out.write("@prefix "+p+": <"+Prefixes.prefixes[p]+"> .\n")
        # Predicates
        kb1_predicates=kb1.predicates()
        kb2_predicates=kb2.predicates()
        predicates = kb1_predicates | kb2_predicates
        for predicate1 in predicates:
            predicate1_keys = [predicate1]
            if sides.can_use_id_keyed_state(kb1, kb2):
                predicate1_keys = sides.predicate_side_keys(predicate1, kb1, kb2)
            for predicate1_key in predicate1_keys:
                if predicate1_key not in predicate2superPredicate:
                    continue
                for predicate2_key, score in predicate2superPredicate[predicate1_key].items():
                    if score > 0.1:
                        predicate2 = sides.decode_predicate_key(predicate2_key, kb1, kb2)
                        out.write(predicate1+"\trdfs:subPropertyOf\t"+predicate2+"\t.#\t"+str(score)+"\n")
        # Literals and instances
        for entity1 in sameAsScores:
            for entity2 in sameAsScores[entity1]:
                if sameAsScores[entity1][entity2] > 0: # first report all possible scores
                    entity1_out = sides.decode_entity_key(entity1, kb1, kb2)
                    entity2_out = sides.decode_entity_key(entity2, kb1, kb2)
                    out.write(entity1_out+"\towl:sameAs\t"+entity2_out+"\t.#\t"+str(sameAsScores[entity1][entity2])+"\n")
    Announce.done()
    logging.info("Time used for the whole procedure: %s minutes"%(round((time.time() - procedure_start)/60, 5)))
    if os.path.isdir(compact_entity_assign_run_dir):
        try:
            shutil.rmtree(compact_entity_assign_run_dir)
            logging.info("Removed compact entity assignment runtime dir | dir=%s", compact_entity_assign_run_dir)
        except OSError as exc:
            logging.warning("Failed to remove compact entity assignment runtime dir | dir=%s | error=%s", compact_entity_assign_run_dir, exc)
