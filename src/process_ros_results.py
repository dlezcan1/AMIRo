from abc import (
    ABC,
    abstractmethod,
)
import argparse as ap
import glob
import os
import re

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from typing import (
    Union,
    List,
    Dict,
    Any,
)

import numpy as np
import pandas as pd

import needle_shape_sensing as nss
from needle_shape_sensing.shape_sensing import (
    ShapeSensingFBGNeedle,
    ShapeSensingMCFNeedle,
)

class DataPostProcessor(ABC):
    TRIAL_REGEX_PATTERN = os.path.join(r".*/?Insertion(\d+)", r"(\d+\.?\d*)")
    def __init__(self, data_dir: str, data_file: str, out_file: str):
        self.data_dir       = data_dir
        self.base_data_file = data_file
        self.out_file       = out_file

        # data information
        self.insertion_dirs: List              = None
        self.trial_files: List[Dict[str, Any]] = None

    # __init__

    @property
    def is_configured(self):
        return (
            (self.insertion_dirs is not None)
            and (self.trial_files is not None)
        )

    # property: is_configured

    def configure_dataset(self):
        """ Get the dataset ready """
        self.insertion_dirs = sorted(glob.glob(os.path.join(self.data_dir, "Insertion*")))

        # find all of the trial files
        self.trial_files = list()
        for insertion_dir in self.insertion_dirs:
            for trial_file in glob.glob(os.path.join(insertion_dir, "*", self.base_data_file)):
                trial_meta = self.get_trial_meta(trial_file)

                if not self.is_valid_trial_meta(trial_meta):
                    continue


                self.trial_files.append(trial_meta)

            # for
        # for

    # configure_dataset

    @classmethod
    def get_trial_meta(cls, trial_file: str):
        """ Extracts the trieal meta information for processing """
        trial_meta = DataPostProcessor.match_trial_pattern(trial_file)
        trial_meta["filename"] = trial_file

        return trial_meta

    # get_trial_meta

    @classmethod
    def is_valid_trial_meta(cls, trial_meta: Dict[str, Any]):
        """ Determines if trial meta is valid or not"""
        is_valid = all(
            trial_meta.get(key, None) is not None
            for key in ["insertion", "depth", "filename"]
        )

        return is_valid

    # is_valid_trial_meta

    @classmethod
    def match_trial_pattern(cls, trial_file: str):
        """ Returns dict of trial information based on insertion trial path"""
        trial_meta = {
            "insertion": None, # integer of insertion number
            "depth"    : None, # float of insertion depth
        }

        # regex match
        re_match = re.match(cls.TRIAL_REGEX_PATTERN, trial_file)
        if not re_match:
            return trial_meta

        insertion, depth = re_match.groups()

        # assignthe data
        trial_meta["insertion"] = int(insertion)
        trial_meta["depth"]     = float(depth)

        return trial_meta

    # match_trial_pattern

    def process_data(self, save: bool = False, multi_thread_count: int = 0):
        """ Post process the data """
        assert self.is_configured, "Processor is not configured yet! Call the `configure_dataset` method first"

        results = list()

        def process_trial_target(trial_meta):
            result = self.process_trial(trial_meta, save=save)

            return result
        
        # process_trial_target

        # TODO: add multithreading (0-1 = single-thread, -1 = all threads, > 1 = # threads to use )
        num_threads = min(
            multiprocessing.cpu_count(),
            multi_thread_count if multi_thread_count >= 0 else multiprocessing.cpu_count()
        )

        if num_threads <= 1:
            for trial_meta in self.trial_files:
                results.append(process_trial_target(trial_meta))

            # for
        # if
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(process_trial_target, self.trial_files))

            # with
        # else

        return results
    
    # process_data

    @abstractmethod
    def process_trial(self, trial_meta: Dict[str, Any], save: bool=False):
        pass

    # process_trial

# class: DataPostProcessor

class NeedleDataPostProcessor(DataPostProcessor, ABC):
    def __init__(self, data_dir: str, fbg_needle: ShapeSensingFBGNeedle, data_file: str, out_file: str):
        super().__init__(data_dir, data_file, out_file)
        self.fbg_needle = fbg_needle

    # __init__

# class: NeedleDataPostProcessor

class FBGSensorDataPostProcessor(NeedleDataPostProcessor):
    SENSOR_DATA_FILE      = "fbg_sensor_data.xlsx"
    POST_SENSOR_DATA_FILE = SENSOR_DATA_FILE.replace(".xlsx", "_post-proc.xlsx")

    def __init__(
            self, 
            data_dir: str, 
            fbg_needle: ShapeSensingFBGNeedle, 
            data_file: str = None,
            out_file: str = None,
        ):
        super().__init__(
            data_dir,
            fbg_needle,
            data_file=data_file if data_file is not None else FBGSensorDataPostProcessor.SENSOR_DATA_FILE,
            out_file=out_file if out_file is not None else FBGSensorDataPostProcessor.POST_SENSOR_DATA_FILE,
        )

    # __init__

    def process_trial(self, trial_meta: Dict[str, Any], save: bool = False):
        # load the data
        proc_sensor_df: pd.DataFrame = pd.read_excel(
            trial_meta["filename"],
            sheet_name="processed wavelengths",
            header=0,
            index_col=0,
        )

        ch_aa = self.fbg_needle.generate_ch_aa()[0]
        proc_sensor_df.columns = ch_aa

        # perform temperature compensation
        proc_sensor_Tcomp_df        = proc_sensor_df.copy()
        proc_sensor_Tcomp_df[ch_aa] = self.fbg_needle.temperature_compensate(proc_sensor_df[ch_aa].to_numpy())
        mean_proc_sensor_Tcomp      = proc_sensor_Tcomp_df[ch_aa].mean(axis=0)
        
        results = {
            "Tcomp wavlength shifts"      : proc_sensor_Tcomp_df,
            "mean Tcomp wavelength shifts": mean_proc_sensor_Tcomp,
        }

        if save:
            out_file = os.path.join(
                os.path.split(trial_meta["filename"])[0],
                self.out_file
            )
            with pd.ExcelWriter(out_file, engine='xlsxwriter') as xl_writer:
                proc_sensor_Tcomp_df.to_excel(
                    xl_writer,
                    sheet_name="Tcomp processed wavelengths",
                    index=True,
                    header=True,
                )

                mean_proc_sensor_Tcomp.to_excel(
                    xl_writer,
                    sheet_name="mean Tcomp procesed wavelengths",
                )

            # with
            print("Wrote processed sensor data file to:", out_file)

        # if: save

        return results

    # process_trial


# class: FBGSensorDataPostProcessor

class ShapeDataPostProcessor(NeedleDataPostProcessor):
    NEEDLE_DATA_FILE      = "needle_data.xlsx"
    POST_NEEDLE_DATA_FILE = NEEDLE_DATA_FILE.replace(".xlsx", "_post-proc.xlsx")

    def __init__(
            self, 
            data_dir: str, 
            fbg_needle: ShapeSensingFBGNeedle, 
            data_file: str = None,
            out_file: str = None,
            sensor_data_file: str = None,
        ):
        super().__init__(
            data_dir,
            fbg_needle,
            data_file=data_file if data_file is not None else ShapeDataPostProcessor.NEEDLE_DATA_FILE,
            out_file=out_file if out_file is not None else ShapeDataPostProcessor.POST_NEEDLE_DATA_FILE,
        )
        self.sensor_data_file = sensor_data_file

    # __init__

    def process_trial(self, trial_meta: Dict[str, Any], save: bool = False):
        raise NotImplementedError("TODO: Implement this Post-processing of the shape data")

    # process_trial

# class: ShapeDataPostProcessor

def __parse_args(args = None):
    parser = ap.ArgumentParser("Process data dumps from ROS bag files")

    parser.add_argument(
        "--needle-param-file",
        type=str,
        required=True,
        default=None,
        help="The needle parameter file for processing"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        default=None,
        help="Experimental Data directory"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the output post-processed data"
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="The number of threads to multi-threaded process the data (-1 for all threads, 0-1 for single-threaded, > 1 for multi-threaded)"
    )

    toggle_grp = parser.add_argument_group(
        "Post Processing Toggles",
        "Options to determine which post processor gets used"
    )

    toggle_grp.add_argument(
        "--parse-fbg-sensor-data",
        action="store_true",
        help="Post-process the FBG sensor data"
    )


    return parser.parse_args(args)

# __parse_args

def main(args = None):
    ARGS = __parse_args(args)

    data_dir = ARGS.data_dir

    fbg_needle = ShapeSensingFBGNeedle.load_json(ARGS.needle_param_file)
    print(fbg_needle, end='\n\n')

    if isinstance(fbg_needle, ShapeSensingMCFNeedle):
        fbg_needle.options["use_centralcore_Tcomp"] = False # ignore the central core!

    # if

    post_processers: List[DataPostProcessor] = list()
    if ARGS.parse_fbg_sensor_data:
        post_processers.append(
            FBGSensorDataPostProcessor(
                data_dir,
                fbg_needle,
                data_file=None, # use defaults
                out_file=None,  # use defaults
            )
        )

    # if

    # TODO: handle needle shape data

    for processor in post_processers:
        print("Post-processing data with:", type(processor).__qualname__)
        processor.configure_dataset()

        processor.process_data(save=ARGS.save, multi_thread_count=ARGS.num_threads)

    # for

# main

if __name__ == "__main__":
    main()

# if __main__