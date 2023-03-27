import time

from pathlib import Path

ROOT_PATH = Path(__file__).parent
PROJECT_PATH = ROOT_PATH.parent
DATA_PATH = PROJECT_PATH / 'data'
RESULT_PATH = PROJECT_PATH / 'results'
FIGURE_PATH = RESULT_PATH / 'figures'

states_to_remove = [3, 4, 10, 17, 18]   # Berlin, Brandenburg, Nordrhein, Westfalen-Lippe, unidentified

state_to_int = {
    "None": 0,
    "Baden-Württemberg": 1,
    "Bayern": 2,
    "Berlin": 3,
    "Brandenburg": 4,
    "Bremen": 5,
    "Hamburg": 6,
    "Hessen": 7,
    "Mecklenburg-Vorpommern": 8,
    "Niedersachsen": 9,
    "Nordrhein": 10,
    "Rheinland-Pfalz": 11,
    "Saarland": 12,
    "Sachsen": 13,
    "Sachsen-Anhalt": 14,
    "Schleswig-Holstein": 15,
    "Thüringen": 16,
    "Westfalen-Lippe": 17,
    "undefined": 18,
    "Nordrhein-Westfalen": 19,
    "Berlin-Brandenburg": 20,
    "Total": 21
}

int_to_state = {
    0: "None",
    1: "Baden-Württemberg",
    2: "Bayern",
    3: "Berlin",
    4: "Brandenburg",
    5: "Bremen",
    6: "Hamburg",
    7: "Hessen",
    8: "Mecklenburg-Vorpommern",
    9: "Niedersachsen",
    10: "Nordrhein",
    11: "Rheinland-Pfalz",
    12: "Saarland",
    13: "Sachsen",
    14: "Sachsen-Anhalt",
    15: "Schleswig-Holstein",
    16: "Thüringen",
    17: "Westfalen-Lippe",
    18: "undefined",
    19: "Nordrhein-Westfalen",
    20: "Berlin-Brandenburg",
    21: "Total"
}

def timeit(method):
    """Timing decorator"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{:20}  {:8.4f} [s]'.format(method.__name__, (te - ts)))
        return result

    return timed