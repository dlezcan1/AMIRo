from abc import (
    ABC,
    abstractmethod,
)
import argparse as ap
import glob
import itertools
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

import matplotlib.pyplot as plt
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
        self.results                           = list()

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

    def process_data(self, save: bool = False, multi_thread_count: int = 0, show_plots: bool = False):
        """ Post process the data """
        assert self.is_configured, "Processor is not configured yet! Call the `configure_dataset` method first"

        def process_trial_target(trial_meta):
            result = self.process_trial(trial_meta, save=save)

            if show_plots:
                plt.show()

            plt.close('all')

            return result

        # process_trial_target

        num_threads = min(
            multiprocessing.cpu_count(),
            multi_thread_count if multi_thread_count >= 0 else multiprocessing.cpu_count()
        )

        results = list()
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

        self.results = results

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

        ch_aa, ch_names, aa_names = self.fbg_needle.generate_ch_aa()
        proc_sensor_df.columns = ch_aa

        # perform temperature compensation
        proc_sensor_Tcomp_df        = proc_sensor_df.copy()
        proc_sensor_Tcomp_df[ch_aa] = self.fbg_needle.temperature_compensate(proc_sensor_df[ch_aa].to_numpy())
        mean_proc_sensor_Tcomp      = proc_sensor_Tcomp_df[ch_aa].mean(axis=0)

        # get curvatures
        self.fbg_needle.update_wavelengths(
            np.zeros(self.fbg_needle.num_signals),
            reference=True,
            processed=True,
            temp_comp=False
        )
        curvatures = np.stack(
            [
                self.fbg_needle.curvatures_processed(wl_Tcomp[ch_aa].to_numpy())
                for idx, wl_Tcomp in proc_sensor_df.iterrows()
            ],
            axis=0
        )
        curvatures_df = pd.DataFrame(
            curvatures.reshape(-1, curvatures.shape[1] * curvatures.shape[2], order='F'),
            columns=[
                f"{ax} curvature | {aa_name}"
                for aa_name, ax in itertools.product(aa_names, ["x", "y"])
            ]
        )
        curvatures_df.set_index(proc_sensor_df.index, inplace=True, drop=True)
        mean_curvatures_df = curvatures_df.mean(axis=0)

        results = {
            "Tcomp wavlength shifts"      : proc_sensor_Tcomp_df,
            "mean Tcomp wavelength shifts": mean_proc_sensor_Tcomp,
            "curvatures"                  : curvatures_df,
            "mean curvatures"             : mean_curvatures_df,
        }

        # plot
        # - signals
        fig_wls, axs_wls = plt.subplots(
            nrows=1,
            ncols=self.fbg_needle.num_activeAreas,
            sharex=True,
            sharey=True,
            figsize=(18, 6),
        )

        for aa_i in range(self.fbg_needle.num_activeAreas):
            mask_aa_i = self.fbg_needle.assigment_mask_aa(aa_i+1)
            proc_sensor_Tcomp_df[ch_aa[mask_aa_i]]\
                .rename(
                    columns=lambda x: x.split(" | ")[0]
                ).plot(
                subplots=False,
                ax=axs_wls[aa_i],
                legend=True,
            )

            axs_wls[aa_i].set_xlabel("Timestamp")
            axs_wls[aa_i].set_title(aa_names[aa_i])

        # for
        axs_wls[0].set_ylabel("Temp Comp. Processed Wavelengths (nm)")

        fig_wls.suptitle("Temperature Compensated Wavelength Shifts")

        # - curvatures
        fig_curv, axs_curv = plt.subplots(
            nrows=2,
            ncols=self.fbg_needle.num_activeAreas,
            sharex=True,
            sharey=True,
            figsize=(18,8),
        )
        curv_x_mask = np.asarray(["x curvature" in col for col in curvatures_df.columns])
        curv_y_mask = np.logical_not(curv_x_mask)
        for aa_i in range(self.fbg_needle.num_activeAreas):
            mask_aa_i = np.asarray([aa_names[aa_i] in col for col in curvatures_df.columns])
            axs_curv[0, aa_i].plot(
                curvatures_df.index,
                curvatures_df.loc[:, curv_x_mask & mask_aa_i],
                '.',
                label="Curvature",
            )
            axs_curv[0, aa_i].axhline(
                mean_curvatures_df[curv_x_mask & mask_aa_i].item(),
                ls='-',
                color='r',
                label="Avg. Curvature",
            )
            axs_curv[0, aa_i].legend()

            axs_curv[1, aa_i].plot(
                curvatures_df.index,
                curvatures_df.loc[:, curv_y_mask & mask_aa_i],
                '.',
                label="Curvature",
            )
            axs_curv[1, aa_i].axhline(
                mean_curvatures_df[curv_y_mask & mask_aa_i].item(),
                ls='-',
                color='r',
                label="Avg. Curvature",
            )
            axs_curv[1, aa_i].legend()

            axs_curv[0, aa_i].set_title(aa_names[aa_i])
            axs_curv[1, aa_i].set_xlabel("Timestamp")

        # for

        axs_curv[0, 0].set_ylabel("X Curvature (1/m)")
        axs_curv[1, 0].set_ylabel("Y Curvature (1/m)")
        fig_curv.suptitle("FBG-Sensed Curvatures")

        # saving
        if save:
            odir = os.path.split(trial_meta["filename"])[0]
            out_file = os.path.join( odir, self.out_file )
            with pd.ExcelWriter(out_file, engine='xlsxwriter') as xl_writer:
                proc_sensor_Tcomp_df.to_excel(
                    xl_writer,
                    sheet_name="Tcomp processed wavelengths",
                    index=True,
                    header=True,
                )

                mean_proc_sensor_Tcomp.to_excel(
                    xl_writer,
                    sheet_name="avg Tcomp processed wavelengths",
                )

                curvatures_df.to_excel(
                    xl_writer,
                    sheet_name="curvatures",
                    index=True,
                    header=True,
                )

            # with
            print("Wrote processed sensor data file to:", out_file)

            # save plots
            out_file = os.path.join(odir, "sensor_data_post-proc_Tcomp-wavelengths.png")
            fig_wls.savefig( out_file )
            print("Saved wavelength post-processed sensor data figure to:", out_file)

            out_file = os.path.join(odir, "sensor_data_post-proc_curvatures.png")
            fig_curv.savefig( out_file )
            print("Saved curvature post-processed sensor data figure to:", out_file)

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
        # get the updated FBG sensor data
        in_dir         = os.path.split(trial_meta["filename"])[0]
        fbgsensor_data = self.get_sensor_data(in_dir)

        if fbgsensor_data is None:
            print(F"[WARNING]: {in_dir} does not have any usable FBG sensor data")
            return
        # if

        results = dict()

        # get current shape parameters
        current_kc    = pd.read_excel(
            trial_meta["filename"],
            sheet_name="kappa_c",
            header=None,
            index_col=None,
        ).to_numpy().ravel()
        current_winit = pd.read_excel(
            trial_meta["filename"],
            sheet_name="winit",
            header=None,
            index_col=None,
        ).to_numpy().ravel()
        insertion_depth = trial_meta["depth"]

        # process curvature
        self.fbg_needle.update_wavelengths(
            np.zeros(self.fbg_needle.num_signals),
            reference=True,
            temp_comp=False,
        )
        _, curvatures = self.fbg_needle.update_wavelengths(
            fbgsensor_data,
            processed=True,
            reference=False,
            temp_comp=False, # Should already be temperature-compensated
        )

        results["curvatures"] = pd.DataFrame(
            np.reshape(curvatures, (-1, 2*self.fbg_needle.num_activeAreas), order='F'),
            columns = [
                f"curvature {ax} | AA {aa_i}"
                for aa_i, ax in itertools.product(
                    range(1, self.fbg_needle.num_activeAreas + 1),
                    ["x", "y"]
                )
            ]
        )

        # process shape
        self.fbg_needle.current_depth = insertion_depth
        results["shape"], _           = self.fbg_needle.get_needle_shape(*current_kc, current_winit)
        results["kappa_c"]            = np.asarray(self.fbg_needle.current_kc)
        results["winit"]              = np.asarray(self.fbg_needle.current_winit)

        if save:
            out_file = os.path.join(in_dir, self.out_file)
            with pd.ExcelWriter(out_file, engine='xlsxwriter') as xl_writer:
                results["curvatures"].transpose().to_excel(
                    xl_writer,
                    sheet_name="curvature",
                    header=False,
                    index=False,
                )

                pd.DataFrame(results["kappa_c"]).to_excel(
                    xl_writer,
                    sheet_name="kappa_c",
                    header=False,
                    index=False,
                )

                pd.DataFrame(results["winit"]).to_excel(
                    xl_writer,
                    sheet_name="winit",
                    header=False,
                    index=False,
                )

                pd.DataFrame(results["shape"]).to_excel(
                    xl_writer,
                    sheet_name="shape",
                    header=False,
                    index=False,
                )

            # with
            print("Saved needle shape data to:", out_file)

        # if: save

        return results

    # process_trial

    def get_sensor_data(self, dir: str):
        sensor_data_file = self.sensor_data_file
        if sensor_data_file is None:
            if os.path.isfile(os.path.join(dir, FBGSensorDataPostProcessor.POST_SENSOR_DATA_FILE)):
                sensor_data_file = FBGSensorDataPostProcessor.POST_SENSOR_DATA_FILE

            # if
            elif os.path.isfile(os.path.join(dir, FBGSensorDataPostProcessor.SENSOR_DATA_FILE)):
                sensor_data_file = FBGSensorDataPostProcessor.SENSOR_DATA_FILE

            # elif
            else:
                return None

        # if

        df = None
        try:
            df = pd.read_excel(
                os.path.join(dir,sensor_data_file),
                sheet_name="avg Tcomp processed wavelengths",
                index_col=0,
                header=0,
            ).to_numpy().ravel()

        except ValueError as e:
            pass # Do nothing

        return df

    # get_sensor_data





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
        "--show-plots",
        action="store_true",
        help="Show the plots (rather than save them)"
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

    toggle_grp.add_argument(
        "--parse-needle-shape-data",
        action="store_true",
        help="Post-process the needle shape data"
    )


    return parser.parse_args(args)

# __parse_args

def main(args = None):
    ARGS           = __parse_args(args)
    parser_run_options = dict(filter(lambda x: x[0].startswith("parse_"), ARGS._get_kwargs()))

    data_dir = ARGS.data_dir

    fbg_needle = ShapeSensingFBGNeedle.load_json(ARGS.needle_param_file)
    print(fbg_needle, end='\n\n')

    if isinstance(fbg_needle, ShapeSensingMCFNeedle):
        fbg_needle.options["use_centralcore_Tcomp"] = False # ignore the central core!

    # if

    all_processors = {
        "parse_fbg_sensor_data":
            lambda: FBGSensorDataPostProcessor(
                data_dir,
                fbg_needle,
                data_file=None, # use defaults
                out_file=None,  # use defaults
            ),
        "parse_needle_shape_data":
            lambda: ShapeDataPostProcessor(
                data_dir,
                fbg_needle,
                data_file=None,       # use defaults
                out_file=None,        # use defaults
                sensor_data_file=None # use defaults
            ),
    }

    post_processers: List[DataPostProcessor] = list()

    for parse_name, parser_const in all_processors.items():
        if not parser_run_options[parse_name]:
            continue

        post_processers.append(parser_const())

    # for

    for processor in post_processers:
        print("Post-processing data with:", type(processor).__qualname__)
        processor.configure_dataset()

        processor.process_data(
            save=ARGS.save,
            multi_thread_count=ARGS.num_threads,
            show_plots=ARGS.show_plots,
        )

    # for

# main

if __name__ == "__main__":
    main()

# if __main__