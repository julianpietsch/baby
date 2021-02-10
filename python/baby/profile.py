from cProfile import Profile
import functools
import logging
from pstats import Stats
from pathlib import Path
import sys
import timeit
import time

import click

def dummy(s):
    time.sleep(1)
    return s + " dummy"

def profile(f, *args, profiler=None, **kwargs):
    if profiler is None: 
        profiler = Profile()
    retval = profiler.runcall(f, *args, **kwargs)
    profiler.print_stats()
    return retval

class Profiler:
    def __init__(self, profile, outfile=None):
        if profile:
            self.outdir = Path(outfile)
            if not self.outdir.exists():
                self.outdir.mkdir()
            self.profiler = Profile()
            self.prof_fn = self._profile_args
        else:
            self.outfile = outfile
            self.prof_fn = self._time_args
    
    def _profile_args(self, f, *args, **kwargs):
        ret_val = self.profiler.runcall(f, *args, **kwargs)
        if self.outdir is not None:
            self.profiler.dump_stats(self.outdir / f.__name__)
        else:
            self.profiler.print_stats()
        return ret_val


    def _time_args(self, f, *args, **kwargs):
        start = time.perf_counter()
        ret_val = f(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print("{} : {}".format(f.__name__, elapsed), file=self.outfile)
        return ret_val


    def __call__(self, f, *args, **kwargs):
        retval = self.prof_fn(f, *args, **kwargs)
        return retval

@click.command()
@click.option('--profile', is_flag=True, flag_value=True)
@click.option('-o', '--out', default=None)
def run(profile, out):
    profiler = Profiler(profile, out)
    blah = profiler(dummy, 'hello')
    print(blah)

@click.command()
@click.argument('stats_files', type=str, nargs=-1)
def stats(stats_files):
    if isinstance(stats_files, str):
        stats_files = [stats_files]
    stats = Stats(*stats_files)
    stats.print_stats()


@click.group()
def cli():
    pass

cli.add_command(run)
cli.add_command(stats)

if __name__ == "__main__":
    cli()
