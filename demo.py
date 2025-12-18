# XXX: This implementation is slow because it creates a separate process for every single trial
# (it needs an ability to properly kill a busy process). Options to optimize:
# 1. Reimplement the function under test to handle its own iteration limit. Rejected for odd trust
#    reasons.
# 2. Run a pool of worker processes using joblib. Joblib treats `timeout` in a very weird way.
# 3. Run a pool of worker processes using `multiprocessing`. Complicated (might be just impossible
#    with progress reporting).
# 4. Run many trials in a subprocess until it stops responding within timeout. Complicated
#    (especially so if we want to keep progress reporting).


import argparse
from collections.abc import Sized, Iterable
import importlib
import math
import pathlib
import sys
import time

import numpy

import keybert._maxsum


class LazyModuleProvider:

    def __init__(self, module, package=None):
        self._module_obj = None
        self._module = module
        self._package = package

    @property
    def module(self):
        if self._module_obj is None:
            self._module_obj = importlib.import_module(self._module, self._package)
        return self._module_obj


multiprocessing_provider = LazyModuleProvider('multiprocessing')

matplotlib_colors_provider = LazyModuleProvider('matplotlib.colors')
matplotlib_pyplot_provider = LazyModuleProvider('matplotlib.pyplot')
tqdm_auto_provider = LazyModuleProvider('tqdm.auto')


EMBEDDING_LEN = 25

NGRAMS_NUM = 64
# always: results_num <= candidates_num
CANDIDATES_NUM_LIMITS = (1, 20)
RESULTS_NUM_LIMITS = (1, 10)

assert CANDIDATES_NUM_LIMITS[0] <= CANDIDATES_NUM_LIMITS[1]
assert RESULTS_NUM_LIMITS[0] <= RESULTS_NUM_LIMITS[1]


def measure_avg_time(func, args=(), kwargs=None, *, max_iter=int(1e6), time_limit=1.0):
    if kwargs is None:
        kwargs = {}
    time_limit_ns = (int(time_limit * 1e+9) if time_limit is not None else time_limit)
    iterations = 0
    start_ns = time.perf_counter_ns()
    while (
        (iterations < max_iter)
        and (
            (iterations < 1)
            or (
                (time_limit_ns is not None)
                and (((end_ns := time.perf_counter_ns()) - start_ns) < time_limit_ns)
            )
        )
    ):
        func(*args, **kwargs)
        iterations += 1
    total_time = (end_ns - start_ns) * 1e-9
    assert iterations >= 1
    return ((total_time / iterations), iterations)


def random_embeddings(emb_len, shape_ext, generator):
    return generator.normal(0, 1., size=(*shape_ext, emb_len))


class TrialsInfoIterable(Iterable, Sized):

    def __init__(self, candidates_num_limits, results_num_limits):
        assert (
            (len(candidates_num_limits) == 2)
            and (candidates_num_limits[0] <= candidates_num_limits[1])
        )
        assert (len(results_num_limits) == 2) and (results_num_limits[0] <= results_num_limits[1])
        self._candidates_num_limits = candidates_num_limits
        self._results_num_limits = results_num_limits

    def __len__(self):
        candidates_num_range_len = (
            self._candidates_num_limits[1] - self._candidates_num_limits[0] + 1
        )
        results_num_range_len = self._results_num_limits[1] - self._results_num_limits[0] + 1
        triangle_side_len = min(candidates_num_range_len, results_num_range_len)
        return (
            candidates_num_range_len * results_num_range_len
            - (((triangle_side_len - 1) * triangle_side_len) // 2)
        )

    def __iter__(self):
        for candidates_num in range(
            self._candidates_num_limits[0], (self._candidates_num_limits[1] + 1),
        ):
            results_num_limits_loc = (
                self._results_num_limits[0], min(self._results_num_limits[1], candidates_num),
            )
            for results_num in range(results_num_limits_loc[0], (results_num_limits_loc[1] + 1)):
                yield (candidates_num, results_num)


class TimedTrialer:

    def __init__(
        self, doc_embedding, ngrams, ngram_embeddings, *, max_iter=int(1e6), trial_timeout=1.0,
    ):
        self.doc_embedding = doc_embedding
        self.ngrams = ngrams
        self.ngram_embeddings = ngram_embeddings
        self.max_iter = max_iter
        self.trial_timeout = trial_timeout

    @classmethod
    def _worker(cls, connection, func, args=(), kwargs=None, *, max_iter=int(1e6), time_limit=1.0):
        connection.send(None)
        try:
            avg_time, _ = measure_avg_time(
                func, args=args, kwargs=kwargs, max_iter=max_iter, time_limit=time_limit,
            )
        except BaseException as exc:
            connection.send((None, exc))
        else:
            connection.send((avg_time, None))
        connection.close()

    def __call__(self, candidates_num, results_num):
        multiprocessing = multiprocessing_provider.module
        connection_parent, connection_child = multiprocessing.Pipe(duplex=False)
        process = multiprocessing.Process(
            target=self._worker,
            args=(
                connection_child,
                keybert._maxsum.max_sum_distance,   # HERE: a worker calls max_sum_distance
                (),
                dict(
                    doc_embedding=self.doc_embedding[numpy.newaxis, ...],
                    word_embeddings=self.ngram_embeddings,
                    words=self.ngrams,
                    top_n=results_num,
                    nr_candidates=candidates_num,
                ),
            ),
            kwargs=dict(
                max_iter=self.max_iter,
                time_limit=self.trial_timeout,
            ),
        )
        try:
            process.start()
            connection_child.close()    # not needed in the parent
            # to avoid including the process startup time into the timeout:
            _ = connection_parent.recv()
            if connection_parent.poll(self.trial_timeout):
                result, exc = connection_parent.recv()
                process.join()
                if exc is not None:
                    raise RuntimeError('Error in the worker process') from exc
                return result
            process.terminate()
            process.join()
        except KeyboardInterrupt:
            process.terminate()
            process.join()
            raise
        return numpy.nan


def parse_args(argv_trunc, prog_name=None):
    argument_parser_kwargs_extra = {}
    if prog_name is not None:
        argument_parser_kwargs_extra['prog'] = prog_name
    parser = argparse.ArgumentParser(
        allow_abbrev=False, add_help=True, exit_on_error=True, **argument_parser_kwargs_extra,
    )
    parser.add_argument('--trial-timeout', type=float, default=1.0, dest='trial_timeout')
    parser.add_argument('--input', type=pathlib.Path, dest='input_file_path')
    parser.add_argument('-o', '--output', type=pathlib.Path, dest='output_file_path')
    parser.add_argument(
        '--display', action=argparse.BooleanOptionalAction, default=True, dest='do_display',
    )
    parser.add_argument('--seed', type=int, dest='random_seed')
    args = parser.parse_args(argv_trunc)    # may exit
    if math.isinf(args.trial_timeout):
        args.trial_timeout = None
    return args


def run_trials(*, trial_timeout=1.0, random_seed=None):
    embeddings_generator = numpy.random.default_rng(random_seed)
    ngrams_num = NGRAMS_NUM
    doc_embedding = random_embeddings(EMBEDDING_LEN, (), embeddings_generator)
    ngrams = list(map(str, range(ngrams_num)))
    ngram_embeddings = random_embeddings(EMBEDDING_LEN, (ngrams_num,), embeddings_generator)
    candidates_num_limits = CANDIDATES_NUM_LIMITS
    results_num_limits = RESULTS_NUM_LIMITS
    trialer = TimedTrialer(
        doc_embedding=doc_embedding, ngrams=ngrams, ngram_embeddings=ngram_embeddings,
        max_iter=int(1e3),
        trial_timeout=trial_timeout,
    )
    trials_time = numpy.full(
        (
            (candidates_num_limits[1] - candidates_num_limits[0] + 1),
            (results_num_limits[1] - results_num_limits[0] + 1),
        ),
        numpy.nan,
    )
    for candidates_num, results_num in tqdm_auto_provider.module.tqdm(
        TrialsInfoIterable(candidates_num_limits, results_num_limits), unit='trial',
    ):
        trials_time[
            (candidates_num - candidates_num_limits[0]),
            (results_num - results_num_limits[0]),
        ] = trialer(candidates_num, results_num)
    return trials_time


def display_trials_time(
    trials_time, *, candidates_num_limits=None, results_num_limits=None, trial_timeout=None,
):
    if candidates_num_limits is None:
        candidates_num_limits = (1, trials_time.shape[0])
    if results_num_limits is None:
        results_num_limits = (1, trials_time.shape[1])
    assert len(candidates_num_limits) == 2
    assert len(results_num_limits) == 2
    trials_time_log = numpy.log10(trials_time)
    matplotlib_pyplot = matplotlib_pyplot_provider.module
    cmap = matplotlib_pyplot.get_cmap('inferno').copy()
    cmap.set_bad('tab:blue')
    fig, ax = matplotlib_pyplot.subplots()
    _ = ax.set_xlabel('Number of candidates')
    _ = ax.set_ylabel('Number of results ("top N")')
    im = ax.imshow(
        trials_time_log.T,
        cmap=cmap,
        vmin=-5,
        vmax=(numpy.log10(trial_timeout) if trial_timeout is not None else 0),
        origin='lower',
        extent=(
            (candidates_num_limits[0] - 0.5), (candidates_num_limits[1] + 0.5),
            (results_num_limits[0] - 0.5), (results_num_limits[1] + 0.5),
        ),
    )
    _ = ax.axline((1, 1), slope=0.5, color='black', linestyle=':', linewidth=1)
    _ = fig.colorbar(im, label='Avg. run time (log s)')
    return im


def main(argv):
    args = parse_args(argv[1:], argv[0])    # may exit
    trial_timeout = args.trial_timeout
    if args.input_file_path is not None:
        trials_time = numpy.genfromtxt(args.input_file_path, delimiter=',')
    else:
        trials_time = run_trials(trial_timeout=trial_timeout, random_seed=args.random_seed)
    if args.output_file_path is not None:
        numpy.savetxt(args.output_file_path, trials_time, delimiter=',')
    if args.do_display:
        _ = display_trials_time(
            trials_time,
            candidates_num_limits=CANDIDATES_NUM_LIMITS,
            results_num_limits=RESULTS_NUM_LIMITS,
            trial_timeout=trial_timeout,
        )
        matplotlib_pyplot_provider.module.show()
    return 0


if __name__ == '__main__':
    sys.exit(int(main(sys.argv)))
