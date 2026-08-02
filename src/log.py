from queue import Full
import gc
import logging
import os
import time
import traceback
import sides


DEFAULT_PROGRESS_LOG_INTERVAL = 300.0


def current_pss_mb():
    """Return the current proportional set size (PSS) in MB when available."""
    return pss_mb_for_pid(os.getpid())


def pss_mb_for_pid(pid):
    """
    Return proportional set size (PSS) in MB for a process id when available.
    PSS divides shared pages across processes.
    """
    try:
        with open(f'/proc/{pid}/smaps_rollup', 'rt', encoding='utf-8') as smaps_file:
            for line in smaps_file:
                if line.startswith('Pss:'):
                    pss_kb = int(line.split()[1])
                    return pss_kb / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def process_tree_memory_snapshot(worker_pids=None, parent_pid=None):
    """Measure parent plus worker PSS. Missing/exited workers are ignored."""
    if parent_pid is None:
        parent_pid = os.getpid()
    worker_pids = [pid for pid in (worker_pids or []) if pid is not None]

    parent_pss_mb = pss_mb_for_pid(parent_pid)
    worker_pss = []
    missing_worker_pids = []
    for pid in worker_pids:
        pss_mb = pss_mb_for_pid(pid)
        if pss_mb is None:
            missing_worker_pids.append(pid)
            continue
        worker_pss.append((pid, pss_mb))

    worker_pss_mb = sum(pss_mb for _, pss_mb in worker_pss)
    total_pss_mb = (parent_pss_mb or 0.0) + worker_pss_mb
    return {
        'parent_pid': parent_pid,
        'parent_pss_mb': parent_pss_mb,
        'worker_pids': worker_pids,
        'alive_worker_pids': [pid for pid, _ in worker_pss],
        'missing_worker_pids': missing_worker_pids,
        'worker_pss_mb': worker_pss_mb,
        'total_pss_mb': total_pss_mb,
    }


class ProcessMemoryPeakTracker(object):
    """Track parent + worker PSS peak during a worker stage."""

    def __init__(self, stage_label, worker_pids=None, sample_interval=1.0):
        self.stage_label = stage_label
        self.worker_pids = list(worker_pids or [])
        self.sample_interval = sample_interval
        self.last_sample_time = 0.0
        self.current = None
        self.peak = None

    def set_worker_pids(self, worker_pids):
        self.worker_pids = list(worker_pids or [])
        self.sample(force=True)

    def sample(self, force=False):
        now = time.monotonic()
        if not force and self.current is not None and now - self.last_sample_time < self.sample_interval:
            return self.current
        snapshot = process_tree_memory_snapshot(self.worker_pids)
        self.current = snapshot
        self.last_sample_time = now
        if self.peak is None or snapshot['total_pss_mb'] > self.peak['total_pss_mb']:
            self.peak = dict(snapshot)
        return snapshot

    def progress_fields(self):
        current = self.sample()
        fields = {
            'total_pss_mb': f"{current['total_pss_mb']:.2f}",
            'worker_pss_mb': f"{current['worker_pss_mb']:.2f}",
        }
        if self.peak is not None:
            fields['peak_total_pss_mb'] = f"{self.peak['total_pss_mb']:.2f}"
        return fields

    def log_current(self, label=None):
        snapshot = self.sample(force=True)
        parent_pss = snapshot['parent_pss_mb']
        parent_pss_text = "unavailable" if parent_pss is None else f"{parent_pss:.2f}"
        logging.info(
            "%s memory snapshot | total_pss_mb=%.2f | parent_pss_mb=%s | "
            "worker_pss_mb=%.2f | alive_workers=%s | missing_workers=%s | worker_pids=%s",
            label or self.stage_label,
            snapshot['total_pss_mb'],
            parent_pss_text,
            snapshot['worker_pss_mb'],
            len(snapshot['alive_worker_pids']),
            len(snapshot['missing_worker_pids']),
            snapshot['worker_pids'],
        )

    def log_peak(self, label=None):
        self.sample(force=True)
        snapshot = self.peak or self.current
        if snapshot is None:
            logging.info("%s memory peak | unavailable", label or self.stage_label)
            return
        parent_pss = snapshot['parent_pss_mb']
        parent_pss_text = "unavailable" if parent_pss is None else f"{parent_pss:.2f}"
        logging.info(
            "%s memory peak | total_pss_mb=%.2f | parent_pss_mb=%s | "
            "worker_pss_mb=%.2f | alive_workers=%s | missing_workers=%s | worker_pids=%s",
            label or self.stage_label,
            snapshot['total_pss_mb'],
            parent_pss_text,
            snapshot['worker_pss_mb'],
            len(snapshot['alive_worker_pids']),
            len(snapshot['missing_worker_pids']),
            snapshot['worker_pids'],
        )


def prepare_for_worker_fork(stage_label=None):
    """Reduce avoidable copy-on-write pressure before spawning worker processes."""
    gc.collect()
    if hasattr(gc, 'freeze'):
        gc.freeze()
        logging.info("%s | gc=collected+freeze", stage_label or "Worker fork prep")
    else:
        logging.info("%s | gc=collected", stage_label or "Worker fork prep")


def log_memory_snapshot(stage_label):
    pss_mb = current_pss_mb()
    if pss_mb is None:
        logging.info("%s | pss_mb=unavailable | pid=%s", stage_label, os.getpid())
        return
    logging.info("%s | pss_mb=%.2f | pid=%s", stage_label, pss_mb, os.getpid())


def _record_entity_expansion_profile(profile_stats, timings, entity_memory_profile, context_pairs=0):
    """Update aggregate worker expansion counters for one processed entity."""
    total_time = sum(timings.values())
    if total_time <= 0:
        return

    profile_stats['total_candidate_obj2'] = (
        profile_stats.get('total_candidate_obj2', 0)
        + entity_memory_profile.get('candidate_obj2', 0)
    )
    profile_stats['total_scanned_evi2'] = (
        profile_stats.get('total_scanned_evi2', 0)
        + entity_memory_profile.get('scanned_evi2', 0)
    )
    profile_stats['max_scanned_evi2_entity'] = max(
        profile_stats.get('max_scanned_evi2_entity', 0),
        entity_memory_profile.get('scanned_evi2', 0),
    )
    profile_stats['max_context_pairs_entity'] = max(
        profile_stats.get('max_context_pairs_entity', 0),
        context_pairs,
    )
    profile_stats['max_align_time_entity_s'] = max(
        profile_stats.get('max_align_time_entity_s', 0.0),
        timings.get('align', 0.0),
    )


def nested_mapping_stats(mapping):
    pair_count = 0
    max_targets = 0
    non_empty_sources = 0
    for targets in mapping.values():
        target_count = len(targets)
        pair_count += target_count
        if target_count > 0:
            non_empty_sources += 1
        if target_count > max_targets:
            max_targets = target_count
    return {
        'sources': len(mapping),
        'non_empty_sources': non_empty_sources,
        'pairs': pair_count,
        'max_targets_per_source': max_targets,
    }


def log_nested_mapping_stats(stage_label, mapping):
    stats = nested_mapping_stats(mapping)
    logging.info(
        "%s | sources=%s | non_empty=%s | pairs=%s | max_targets_per_source=%s",
        stage_label,
        stats['sources'],
        stats['non_empty_sources'],
        stats['pairs'],
        stats['max_targets_per_source'],
    )


def _short_term(term, max_length=180):
    text = str(term)
    return text if len(text) <= max_length else text[:max_length - 3] + '...'


def _short_entity_term(term, kb1=None, kb2=None, max_length=180):
    return _short_term(sides.decode_entity_key(term, kb1, kb2), max_length=max_length)


def log_alignment_fanout(stage_label, mapping, kb1=None, kb2=None, top_n=10, min_targets=50, target_sample=8):
    """Log high fan-out sources in nested alignment mappings."""
    thresholds = (10, 20, 50, 100, 500, 1000)
    threshold_counts = {threshold: 0 for threshold in thresholds}
    top_items = []

    for entity, target_scores in mapping.items():
        target_count = len(target_scores)
        if target_count == 0:
            continue
        for threshold in thresholds:
            if target_count >= threshold:
                threshold_counts[threshold] += 1
        if target_count >= min_targets:
            top_items.append((target_count, max(target_scores.values()), entity, target_scores))

    logging.info(
        "%s fanout summary | >=10=%s | >=20=%s | >=50=%s | >=100=%s | "
        ">=500=%s | >=1000=%s | detail_min_targets=%s",
        stage_label,
        threshold_counts[10],
        threshold_counts[20],
        threshold_counts[50],
        threshold_counts[100],
        threshold_counts[500],
        threshold_counts[1000],
        min_targets,
    )

    top_items.sort(key=lambda item: (-item[0], -item[1], str(item[2])))
    for rank, (target_count, max_score, entity, target_scores) in enumerate(top_items[:top_n], start=1):
        scores = list(target_scores.values())
        sample_items = sorted(target_scores.items(), key=lambda item: (-item[1], str(item[0])))[:target_sample]
        sample = [
            "%s|%s|%.6g" % (
                _short_entity_term(target, kb1, kb2),
                sides.entity_side_label(target, kb1, kb2),
                score,
            )
            for target, score in sample_items
        ]
        logging.info(
            "%s fanout detail | rank=%s | entity=%s | side=%s | targets=%s | "
            "min_score=%.6g | max_score=%.6g | distinct_scores=%s | "
            "max_score_targets=%s | target_sample=%s",
            stage_label,
            rank,
            _short_entity_term(entity, kb1, kb2),
            sides.entity_side_label(entity, kb1, kb2),
            target_count,
            min(scores),
            max_score,
            len(set(scores)),
            sum(1 for score in scores if score == max_score),
            sample,
        )


def predicate_mapping_stats(predicate_mapping):
    relation_pairs = 0
    max_matches = 0
    for matches in predicate_mapping.values():
        match_count = len(matches)
        relation_pairs += match_count
        max_matches = max(max_matches, match_count)
    return {
        'predicates': len(predicate_mapping),
        'relation_pairs': relation_pairs,
        'max_matches_per_predicate': max_matches,
    }


def log_predicate_mapping_stats(stage_label, predicate_mapping):
    stats = predicate_mapping_stats(predicate_mapping)
    logging.info(
        "%s | predicates=%s | relation_pairs=%s | max_matches_per_predicate=%s",
        stage_label,
        stats['predicates'],
        stats['relation_pairs'],
        stats['max_matches_per_predicate'],
    )


def _build_progress_logger(stage_label, interval=DEFAULT_PROGRESS_LOG_INTERVAL):
    """Create a throttled progress logger for long-running stages."""
    if not stage_label:
        return None

    start_time = time.monotonic()
    last_log_time = start_time

    def maybe_log(**fields):
        nonlocal last_log_time
        now = time.monotonic()
        if now - last_log_time < interval:
            return False
        parts = [f"elapsed_min={(now - start_time) / 60.0:.2f}"]
        parts.extend(f"{key}={value}" for key, value in fields.items())
        logging.info("%s progress | %s", stage_label, " | ".join(parts))
        last_log_time = now
        return True

    return maybe_log


def run_worker_with_crash_logging(worker_fn, worker_args, profile_queue=None, stage_label=None):
    """Run a worker target and make child-process failures visible in the main log."""
    try:
        worker_fn(*worker_args)
    except SystemExit as exc:
        if exc.code in (None, 0):
            raise
        logging.exception(
            "%s worker exited with non-zero SystemExit | pid=%s | code=%s",
            stage_label or "Worker stage",
            os.getpid(),
            exc.code,
        )
        raise
    except BaseException as exc:
        crash_info = {
            '_worker_crash': True,
            'pid': os.getpid(),
            'stage_label': stage_label,
            'exception_type': type(exc).__name__,
            'exception': str(exc),
            'traceback': traceback.format_exc(),
        }
        logging.exception(
            "%s worker crashed | pid=%s | exception=%s: %s",
            stage_label or "Worker stage",
            os.getpid(),
            type(exc).__name__,
            exc,
        )
        if profile_queue is not None:
            try:
                profile_queue.put_nowait(crash_info)
            except Full:
                pass
        raise