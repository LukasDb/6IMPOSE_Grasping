import os
import json
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def write_settings():
    with open('settings.json', 'w') as F:
        json.dump(App().settings, F, indent=2, cls=NumpyArrayEncoder)


def read_settings():
    with open('settings.json', 'r') as F:
        data = json.load(F)
    App().settings = data


def register_settings(name, new_set):
    set = App().settings
    set[name] = new_set


class App:
    """ App Singleton """
    _instance = None
    window = None
    scene = None
    pov_scene = None
    settings = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(App, cls).__new__(cls, *args, **kwargs)
            read_settings()
        return cls._instance
