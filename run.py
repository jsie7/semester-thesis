#!/usr/bin/env python

"""Pipeline runner module

This module defines a pipeline runner and a related data pipeline. The runner
can parse and run pipelines defined in a YAML configuration file.
The related data pipeline allows to pass data between pipeline steps, the
ordinary pipeline does not.

Todo:
    * Check if _run() is really needed in the DataPipeline, maybe this can be
      omitted.
    * Check if _reparse_methods() is needed or can be replaced by a better
      parser.
    * On a module level, revisit the structure of both classes and check if
      there is a more logical and easier implementation.

"""

# built-in modules
import argparse
import copy
import datetime as dt
import importlib
import logging
import os
import pickle
from pprint import pprint
import shutil
import sys

# third party modules
import yaml

# set logging configuration
logger = logging.getLogger("PipelineRunner")
logging.basicConfig(format='%(name)s - %(levelname)s: %(message)s',
    level=logging.INFO)


class PipelineRunner(object):
    """Pipeline runner class.

    This class parses and runs pipelines defined in YAML configuration files.
    The pipeline runner should be called by executing run.py from the
    command line.

    Example:
        python run.py --config_file example_config.yaml

    """
    def __init__(self, config_fname):
        """Initialize pipeline runner.

        This constructor will call the parser and parse the configuration
        file given as argument. Subsequently, the pipeline can be run with
        a call to the 'run()' member function.
        The constructor also prints the parsed pipeline to the console.

        Args:
            config_fname (str): Path to the YAML configuration file.

        Note:
            The parser returns a hirarchical dictionary containing the
            pipeline steps. This dictionary - called config_dict - will
            be passed between several functions inside this module.

        """
        self.timestamp = int(dt.datetime.now().timestamp())
        self.datapipeline = False
        self.config_fname = config_fname
        self.config_dict = self.parse(config_fname)
        self.print_config()

    def _replace_wildcards(self, config_dict, key, **kwargs):
        """Internal function replacing wildcards during parsing.

        This function replaces the wildcards '[fun]', 'timestamp', 'files'
        and 'base_path' with their dynamic equivalent.

        Args:
            config_dict (dict): One step of the pipeline key in the dictionary
                returned by the YAML parser. _recursive_parse() passes the
                pipeline stepwise to this function.
            key (str): Key in config_dict to replace wildcards in. This key
                should be a parameter key, e.g. params or init_params.
            kwargs (str): Optional argument containing the string which defines
                the module containing the external functions.

        Returns:
            dictionary: The passed dictionary step with replaced wildcards.

        """
        for kw in config_dict[key]:
            # single string
            if isinstance(config_dict[key][kw], str):
                if config_dict[key][kw].startswith('[fun]'):
                    function = config_dict[key][kw].strip('[fun]').strip(' ')
                    if function.startswith('lambda'):
                        config_dict[key][kw] = eval(function)
                    else:
                        config_dict[key][kw] = getattr(
                            kwargs['external_module'], function)
                if config_dict[key][kw] == 'timestamp':
                    config_dict[key][kw] = self.timestamp
                if config_dict[key][kw] == 'files':
                    config_dict[key][kw] = self.fnames
                if config_dict[key][kw] == 'base_path':
                    config_dict[key][kw] = self.base_path

            # list of strings
            if isinstance(config_dict[key][kw], list):
                entries = []
                for entry in config_dict[key][kw]:
                    if isinstance(entry, str):
                        if entry.startswith('[fun]'):
                            function = entry.strip('[fun]').strip(' ')
                            if function.startswith('lambda'):
                                entries.append(eval(function))
                            else:
                                entries.append(getattr(
                                    kwargs['external_module'], function))
                        elif entry == 'timestamp':
                            entries.append(self.timestamp)
                        elif entry == 'files':
                            entries.append(self.fnames)
                        elif entry == 'base_path':
                            entries.append(self.base_path)
                        else:
                            entries.append(entry)
                    else:
                        entries.append(entry)
                config_dict[key][kw] = entries

        return config_dict

    def _recursive_run(self, config_dict, arg):
        """Internal function running pipeline steps recursively.

        This function actually runs the pipeline. It takes the parsed pipeline
        as an argument and steps through the dictionary recursively.
        If it finds a 'module' key in the pipeline it will call itself again,
        otherwise it will execute the executable code.

        Args:
            config_dict (dict): Dictionary containing the parsed pipeline.
            arg (str): Argument passed to class constructor or module-level
                methods. In this case, this will be the path to the data file.

        """
        output = None
        if key_in_dict('class', config_dict):
            if key_in_dict('init_params', config_dict):
                init_params = config_dict['init_params']
                class_instance = config_dict['class'](arg, **init_params)
            else:
                class_instance = config_dict['class'](arg)

            for step in config_dict['pipeline']:
                if 'module' in step:
                    output = self._recursive_run(step, arg)
                if key_in_dict('method', step):
                    method = getattr(class_instance, step['method'])
                    if key_in_dict('params', step):
                        params = step['params']
                        output = method(**params)
                    else:
                        output = method()
        elif key_in_dict('method', config_dict):
            if key_in_dict('params', config_dict):
                params = config_dict['params']
                output = config_dict['method'](arg, **params)
            else:
                output = config_dict['method'](arg)
        else:
            for step in config_dict['pipeline']:
                if 'module' in step:
                    output = self._recursive_run(step, arg)
                if key_in_dict('method', step):
                    if key_in_dict('params', step):
                        params = step['params']
                        output = step['method'](arg, **params)
                    else:
                        output = step['method'](arg)
                if self.datapipeline:
                    arg = output

        return output

    def _recursive_parse(self, config_dict):
        """Internal function parsing config file recursively.

        This function recursively parses the configuration file. It steps
        through the list linked to the 'pipeline' key word and recursivley
        calls itself if another module is found inside the pipeline.

        Args:
            config_dict (dict): Dictionary containtaining the hirarchical
                pipeline as returned by the YAML parser.

        Returns:
            dictionary: The same dictionary passed to the function where
                wildcards and modules/classes/methods are replaced. Runnable
                parts are replaced by bound methods.

        Raises:
            AssertionError: If config_dict is not a dictionary.
            AssertionError: If there is no module or pipeline keyword in
                config_dict.

        """
        assert isinstance(config_dict, dict)
        assert key_in_dict(['module', 'pipeline'], config_dict)

        # parse header
        config_dict['module'] = importlib.import_module(config_dict['module'])
        if 'external_functions' in config_dict:
                config_dict['external_functions'] = importlib.import_module(
                                            config_dict['external_functions'])
        if key_in_dict('class', config_dict):
            if config_dict['class'] == "DataPipeline":
                self.datapipeline = True
                config_dict['class'] = True
            else:
                config_dict['class'] = getattr(config_dict['module'],
                    config_dict['class'])
            if key_in_dict('init_params', config_dict):
                if key_in_dict('external_functions', config_dict):
                    config_dict = self._replace_wildcards(config_dict,
                        'init_params',
                        external_module=config_dict['external_functions'])
                else:
                    config_dict = self._replace_wildcards(config_dict,
                        'init_params')

        # parse pipeline
        for step in config_dict['pipeline']:
            if 'module' in step:
                step = self._recursive_parse(step)
            if key_in_dict('params', step):
                if key_in_dict('external_functions', config_dict):
                    step = self._replace_wildcards(step, 'params',
                        external_module=config_dict['external_functions'])
                else:
                    step = self._replace_wildcards(step, 'params')
                if 'timestamp' in step['params']:
                    step['params']['timestamp'] = self.timestamp
            if key_in_dict('method', step):
                if not key_in_dict('class', config_dict):
                    step['method'] = getattr(config_dict['module'],
                        step['method'])

        return config_dict

    def run(self):
        """Run the pipeline.

        This function runs the pipeline which was previously parsed from the
        config file by the parser. It defines if a data pipeline is called or
        not and subsequently calls the recursive runner.

        Args:
            None.

        Raises:
            Warning: If at least one of the files is not found.

        """
        if self.datapipeline:
            DataPipeline(self.config_fname, self.config_dict).run()
            return
        else:
            timestamp = str(self.timestamp)
            for fname in self.fnames:
                fpath = self.base_path + fname
                directory = os.path.dirname(os.path.abspath(fpath))
                try:
                    logger.info('Processing {}.'.format(fname))
                    self._recursive_run(self.config_dict, fpath)
                    shutil.copy(src=self.config_fname,
                                dst=directory + '/' + 'config-' + timestamp +
                                '.yaml')
                except FileNotFoundError:
                    logger.warning('{} not found.\n'.format(fpath))
            return

    def parse(self, config_fname):
        """Parser.

        This function is called by the constructor, parses the YAML
        configuration file and builds a pipeline from it.

        Args:
            config_fname (str): Path to the YAML configuration file.

        Returns:
            dictionary: The dictionary returned by the recursive parser.
                All executable parts (modules/classes/methods) have been
                loaded and replaced by executable code.

        """
        logger.info('Parsing {}.'.format(config_fname))

        with open(config_fname) as config_file:
            config_dict = yaml.load(config_file)
        assert isinstance(config_dict, dict)
        assert key_in_dict(['base_path', 'files', 'pipeline'], config_dict)
        self.base_path = config_dict['base_path'] if \
            config_dict['base_path'].endswith('/') else \
            config_dict['base_path'] + '/'
        self.fnames = config_dict['files']

        # recursively parse the file
        config_dict = self._recursive_parse(config_dict)

        return config_dict

    def print_config(self):
        """Print the parsed pipeline.

        Prints the parsed configuration file - which defines the pipeline -
        to the console.

        Args:
            None.

        """
        logger.info("=========== Config ===========")
        pprint(self.config_dict)
        logger.info("==============================")
        sys.stdout.flush()


class DataPipeline(PipelineRunner):
    """Data pipeline class; Inherits from pipeline runner class.

    The data pipeline is a special form of pipeline and has to be
    initiated by the parent pipeline runner class. The class
    constructor will initialize the class instance based on the state
    of the pipeline runner, at the point in time the constructor was
    called.

    Note:
        This class can be called on its own, but this is discouraged
        since the PipelineRunner class is designed to initialize this
        class.

    """
    def __init__(self, config_fname, config_dict):
        """Initialize data pipeline.

        This constructor takes the state of the pipeline runner instance
        calling it and initializes the data pipeline on the basis of this
        state.

        Args:
            config_fname (str): Path to the YAML configuration file.
            config_dict (dict): Parsed pipeline containing the state of the
                pipeline runner. This was parsed by the PipelineRunner class.

        Note:
            The data which will be passed between steps is saved in 'data_dict'
            where the keys are the file names and the values the data. At
            initialization the data are the complete file paths.

        """
        logger.info("Running as a data pipeline!")
        logger.info("Make sure the first step in the pipeline reads the files"
            " and yields an output compatible to the input of the next step!")
        self.config_fname = config_fname
        self.config_dict = config_dict
        self.base_path = config_dict['base_path'] if \
            config_dict['base_path'].endswith('/') else \
            config_dict['base_path'] + '/'
        self.fnames = config_dict['files']
        self.datapipeline = True
        self.timestamp = int(dt.datetime.now().timestamp())
        self.data_dict = dict(zip(self.fnames, [self.base_path + f for f in
            self.fnames]))
        self.once = False

    def _run(self, config_dict):
        """Internal run function.

        This is the equivalent to the 'run' method in the PipelineRunner class.
        This internal function is necessary because not the whole pipeline is
        run recursively and the data needs to be passed between states.

        The 'arg' argument of recursive_run, will be the data to pass between
        steps.

        Args:
            config_dict (dict): one step of the parsed pipeline.

        """
        config_dict = self._reparse_methods(config_dict)
        if key_in_dict('class', config_dict) and \
        config_dict['class'] == self.datapipeline:
            for step in config_dict['pipeline']:
                output = self._run(step)
        else:
            if key_in_dict('class', config_dict):
                logger.info('Processing {}.'.format(config_dict['class']))
            elif key_in_dict('module', config_dict):
                logger.info('Processing {}.'.format(config_dict['module']))
            else:
                logger.info('Processing {}.'.format(config_dict['method']))
            if self.once:
                logger.info('Only called once with aggregated data.')
                arg = self.data_dict[self.fnames[0]]
                output = self._recursive_run(config_dict, arg)
                self.data_dict = output
                self.once = False
                if hasattr(self, "aggregated_data"):
                    self.aggregated_data = None
            else:
                for fname in self.fnames:
                    arg = self.data_dict[fname]
                    output = self._recursive_run(config_dict, arg)
                    self.data_dict[fname] = output
                if hasattr(self, "aggregated_data"):
                    self.aggregated_data = None
        return

    def _print_data(self):
        """Print pipeline output.

        Prints the data pipeline output to the console and calls the
        write_to_file function before returning.

        Args:
            None.

        """
        logger.info("----- Data Pipeline Output -----")
        pprint(self.data_dict)
        logger.info("--------------------------------")
        self._write_to_file()
        return

    def _write_to_file(self):
        """Write pipeline output to file.

        Writes the data pipeline output to a pickle file. The location is
        defined by the base_path and file names in the configuration file.

        The usage of pickle files allows the user to load the output data
        in the form the last pipeline step returned the data and no processing
        is required.

        Args:
            None.

        """
        logger.info("Saving output to pickle file. This preserves the output"
            "of your methods.")
        logger.info("Load from disk with - pickle.load(open(\"output_file."
            "pickle\", 'rb')) -")
        timestamp = str(self.timestamp)
        for fname in self.data_dict:
            fpath = self.base_path + fname
            directory = os.path.dirname(os.path.abspath(fpath))
            f = open(directory + '/DataPipeline_output-' + timestamp +
                '.pickle', "wb")
            pickle.dump(self.data_dict[fname], f)
        return

    def _reparse_methods(self, config_dict):
        """Reparse a part of the pipeline.

        Reparses a part of the config_dict. This is needed in order to replace
        the aggregate_data string with executable code as well. This cannot be
        parsed by the parser in the PipelineRunner since aggregate_data is a
        member of the DataPipeline instance.

        Args:
            config_dict (dict): Dictionary containing the parsed pipeline.

        Returns:
            dictionary: The input with replaced 'aggregate_data' methods.

        """
        if key_in_dict('method', config_dict):
            config_dict['method'] = getattr(self, config_dict['method'])
        return config_dict

    def run(self):
        """Run the data pipeline.

        This function runs the data pipeline. Instead of recursively running
        through the whole dict, the level containing the data pipeline is
        ommited. Due to this the data can be aggregated between steps of the
        data pipeline if wished.

        Args:
            None.

        Raises:
            Warning: If at least one of the files is not found.

        """
        timestamp = str(self.timestamp)
        for step in self.config_dict['pipeline']:
            self._run(step)
        timestamp = str(self.timestamp)
        for fname in self.fnames:
            directory = os.path.dirname(os.path.abspath(self.base_path +
                fname))
            try:
                shutil.copy(src=self.config_fname,
                            dst=directory + '/' + 'config-' + timestamp +
                            '.yaml')
            except FileNotFoundError:
                logger.warning('{} not found.\n'.format(directory))
        self._print_data()
        return

    def aggregate_data(self, data, once=False):
        """Aggregate data between pipeline steps.

        Checks if aggregated data already exists and saves a copy of the
        data_dict if not.

        Args:
            data (data): Data passed to the method by the data pipeline.
                This data will be discarded, since the function has access
                to the internal data dict.
            once (bool): If True state of data pipeline will be set to 'once'
                and the pipeline will abort the next step after one call.

        Returns:
            dict: the aggregated_data, i.e. the complete data_dict.

        """
        if not hasattr(self, "aggregated_data") or self.aggregated_data is \
        None:
            self.aggregated_data = copy.deepcopy(self.data_dict)
        if once:
            self.once = True
        return self.aggregated_data


def key_in_dict(keys, dict):
    """Check if keys appear in dictionary or are None.

    Args:
        keys (str or list): Keys to be checked.
        dict (dict): Dictionary.

    Returns:
        bool: True if all keys appear in dict and are not None, False o/w.

    Raises:
        AssertionError: If keys is neither a string nor list.

    """
    assert type(keys) in [str, list]
    if type(keys) == str:
        keys = [keys]
    for key in keys:
        if key not in dict.keys() or dict[key] is None:
            return False

    return True


if __name__ == '__main__':
    """Main routine.

    Args:
        keys (str or list): Keys to be checked
        dict (dict): Dictionary

    Returns:
        bool: True if all keys appear in dict and are not None, False o/w

    """
    arg_parser = argparse.ArgumentParser(description="Pipeline runner.")

    arg_parser.add_argument("--config_file", metavar="CONFIG_FILE",
                            help="Configuration file")

    args = arg_parser.parse_args()

    if os.path.isfile(args.config_file):
        config_fname = os.path.abspath(args.config_file)
    else:
        logger.info('There is no file with name {}.'.format(args.config_file))
        sys.exit()

    PipelineRunner(config_fname).run()
