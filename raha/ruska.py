from pprint import pprint
import datetime
import itertools
from pathlib import Path, PosixPath
from typing import Dict, List, Callable


def format_delta(delta: datetime.timedelta):
    """Transforms timedelta to a human-readable format."""
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    h = f'{int(hours)}h ' if hours > 0 else ''
    m = f'{int(minutes)}m ' if minutes > 0 else ''
    s = f'{int(seconds)}s'

    return h + m + s


def estimate_time_to_finish(times: list, total_runs: int):
    deltas = []
    i = 1

    while i + 1 <= len(times):
        deltas.append(times[i] - times[i - 1])
        i += 1

    avg = sum(deltas, datetime.timedelta()) / len(deltas)
    return f'Run {len(times) - 1}/{total_runs}. {format_delta(avg)} per run, estimate {format_delta(avg * (total_runs - len(times)))} to finish'


class Ruska:
    """
    When running data-cleaning experiments, one needs to be extremely diligent to not
    mess up the measurements. This reminds me of my time studying Physics. Measuring
    experiments, I needed to be very careful and diligent, too.

    This is why there is Ruska: Ernst Ruska was a experimental phsicist, and the first
    to construct an electron microscope. He needed to be ridiculously careful building
    that thing in 1933.

    I need a similar level of diligence keeping track of my measurements. Ruska keeps
    track of parameters and measurements.
    """

    def __init__(self,
                 name: str,
                 description: str,
                 commit: str,
                 config: dict,
                 ranges: Dict[str, list],
                 runs: int,
                 save_path: str):
        """Pass all parameters for raha as kwargs."""
        self.name = name
        self.description = description
        self.commit = commit
        self.config = config
        self.ranges = {**ranges, 'run': list(range(runs))}
        self.save_path = Path(save_path) / f'{name}.txt'
        self.times = []

        for range_key in ranges:
            if range_key not in config.keys():
                raise ValueError('Ranges müssen im Config dict enthalten sein.')

        self.range_combinations: List[dict] = []

    @property
    def start_time(self):
        return self.times[0]

    @property
    def end_time(self):
        return self.times[-1]

    def _combine_ranges(self) -> None:
        """
        Calculate all possible combinations of ranges. Assigns them to
        self.range_combinations.
        @return: None
        """
        value_combinations = itertools.product(*list(self.ranges.values()))
        for combination in value_combinations:
            range_combinations = {}
            for i, key_range in enumerate(self.ranges.keys()):
                range_combinations[key_range] = combination[i]
            self.range_combinations.append(range_combinations)

    def run(self, experiment: Callable):
        self._combine_ranges()

        results = []
        self.times.append(datetime.datetime.now())

        for parameter_combination in self.range_combinations:
            config = {**self.config, **parameter_combination}
            result = experiment(**config)
            results.append({'result': result, 'config': config})
            self.times.append(datetime.datetime.now())
            print(estimate_time_to_finish(self.times, len(self.range_combinations)))

        with open(self.save_path, 'w') as f:
            print('Experiment using Ruska finished.', file=f)
            print(f'Start time: {self.times[0]} -- End time: {self.times[-1]}', file=f)
            print('Ruska was configured as follows:', file=f)
            print('[BEGIN CONFIG]', file=f)
            pprint(vars(self), f)
            print('[END CONFIG]', file=f)
            print('Ruska measured the following results:', file=f)
            print('[BEGIN RESULTS]', file=f)
            pprint(results, f)
            print('[END RESULTS]', file=f)
        print('Measurement finished')

    @staticmethod
    def load_result(path_to_result: str):
        path = Path(path_to_result)
        config_flag = False
        result_flag = False

        result = ''
        result_config = ''

        with open(path, 'rt') as f:
            for line in f:
                if line.strip() == '[BEGIN CONFIG]':
                    config_flag = True
                elif line.strip() == '[END CONFIG]':
                    config_flag = False
                elif line.strip() == '[BEGIN RESULTS]':
                    result_flag = True
                elif line.strip() == '[END RESULTS]':
                    result_flag = False
                else:
                    if config_flag:
                        result_config = result_config + line
                    elif result_flag:
                        result = result + line
        result_dict = eval(result)
        result_config_dict = eval(result_config)
        return result_dict, result_config_dict

if __name__ == '__main__':
    config = {'dataset': 'letter', 'strategy': 'paint', 'model': 'dtc'}
    ranges = {'dataset': ['paper', 'letter', 'restaurant']}
    ruska = Ruska('test', 'Eine Test-Messung, die ich ausführe, um zu testen.', 'f12fadsdfads', config, ranges, 3, save_path='/Users/philipp/code/raha/raha')
    print(ruska.load_result('/Users/philipp/code/raha/raha/test.txt'))
    #ruska.run(lambda **kwargs: True)