from typing import Dict, Any, Union
from collections import defaultdict

import argparse
import re


class DotDefaultDict(defaultdict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError
        return self.get(item)

    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__


class DotDict(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError
        return self.get(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_recursive_dot_dict(data: Dict[str, Any], cls=DotDict) -> Union[DotDict, DotDefaultDict]:
    """
    Takes a dict of string keys and arbitrary values, and creates a tree of DotDicts.

    The keys might contain . in which case child DotDicts are created.

    :param data: Input dict with string keys potentially containing .s.
    :param cls: Either DotDict or DotDefaultDict
    :return: tree DotDict or DotDefaultDict where the keys are split by .
    """
    res = cls()
    for k, v in data.items():
        k = k.split(".")
        target = res
        for i in range(0, len(k)-1):
            t2 = target.get(k[i])
            if t2 is None:
                t2 = cls()
                target[k[i]] = t2

            assert isinstance(t2, cls), f"Trying to overwrite key {'.'.join(k[:i+1])}"
            target = t2

        assert isinstance(target, cls), f"Trying to overwrite key {'.'.join(k)}"
        target[k[-1]] = v
    return res


def none_parser(other_parser):
    def fn(x):
        if x.lower() == "none":
            return None
        return other_parser(x)
    return fn


class ArgumentParser:
    _type = type

    @staticmethod
    @none_parser
    def int_list_parser(x):
        return [int(a) for a in re.split("[,_ ;]", x) if a]

    @staticmethod
    @none_parser
    def str_list_parser(x):
        return x.split(",")

    @staticmethod
    @none_parser
    def int_or_none_parser(x):
        return int(x)

    @staticmethod
    @none_parser
    def float_or_none_parser(x):
        return float(x)

    @staticmethod
    @none_parser
    def float_list_parser(x):
        return [float(a) for a in re.split("[,_ ;]", x) if a]

    @staticmethod
    def _merge_args(args, new_args, arg_schemas):
        for name, val in new_args.items():
            old = args.get(name)
            if old is None:
                args[name] = val
            else:
                args[name] = arg_schemas[name]["updater"](old, val)

    class Profile:
        def __init__(self, name, args=None, include=[]):
            assert not (args is None and not include), "One of args or include must be defined"
            self.name = name
            self.args = args
            if not isinstance(include, list):
                include = [include]
            self.include = include

        def get_args(self, arg_schemas, profile_by_name):
            res = {}

            for n in self.include:
                p = profile_by_name.get(n)
                assert p is not None, "Included profile %s doesn't exists" % n

                ArgumentParser._merge_args(res, p.get_args(arg_schemas, profile_by_name), arg_schemas)

            ArgumentParser._merge_args(res, self.args, arg_schemas)
            return res

    def __init__(self, description=None):
        self.parser = argparse.ArgumentParser(description=description)
        self.profiles = {}
        self.args = {}
        self.raw = None
        self.parsed = None
        self.parser.add_argument("-profile", "--profile", type=str, help="Pre-defined profiles.")

    def add_argument(self, name, type=None, default=None, help="", save=True, parser=lambda x: x,
                     updater=lambda old, new: new, choices=[]):
        assert name not in ["profile"], "Argument name %s is reserved" % name
        assert not (type is None and default is None), "Either type or default must be given"

        if type is None:
            type = ArgumentParser._type(default)

        self.parser.add_argument(name, "-" + name, type=int if type == bool else type, default=None, help=help)
        if name[0] == '-':
            name = name[1:]

        self.args[name] = {
            "type": type,
            "default": int(default) if type == bool else default,
            "save": save,
            "parser": parser,
            "updater": updater,
            "choices": choices
        }

    def add_profile(self, prof):
        if isinstance(prof, list):
            for p in prof:
                self.add_profile(p)
        else:
            self.profiles[prof.name] = prof

    def parse_args(self, loaded={}):
        self.raw = self.parser.parse_args()

        profile = {}
        if self.raw.profile:
            if loaded:
                if self.raw.profile != loaded.get("profile"):
                    assert False, "Loading arguments from file, but a different profile is given."
            else:
                for pr in self.raw.profile.split(","):
                    p = self.profiles.get(pr)
                    assert p is not None, "Invalid profile: %s. Valid profiles: %s" % (pr, self.profiles.keys())
                    p = p.get_args(self.args, self.profiles)
                    p['profile_name'] = pr
                    self._merge_args(profile, p, self.args)

        for k, v in self.raw.__dict__.items():
            if k in ["profile"]:
                continue

            if v is None:
                if k in loaded and self.args[k]["save"]:
                    self.raw.__dict__[k] = loaded[k]
                else:
                    self.raw.__dict__[k] = profile.get(k, self.args[k]["default"])

        for k, v in self.raw.__dict__.items():
            if k not in self.args:
                continue
            c = self.args[k]["choices"]
            if c and not v in c:
                assert False, f"Invalid value {v}. Allowed: {c}"

        self.parsed = create_recursive_dot_dict({k: self.args[k]["parser"](self.args[k]["type"](v)) if v is not None
                                                 else None for k, v in self.raw.__dict__.items() if k in self.args})

        return self.parsed
