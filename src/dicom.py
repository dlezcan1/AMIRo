import datetime
import os
import glob

from typing import (
    Dict,
    List,
    Tuple,
)

import pydicom
import numpy as np

class ImageInfo:
    def __init__(self, dcm_img_info: pydicom.Dataset) -> None:
        assert dcm_img_info.DirectoryRecordType == "IMAGE", "Dataset provided is not an of 'IMAGE' directory record type"

        self.dcm_img_info = dcm_img_info

    # __init__

    @property
    def fileID(self) -> str:
        return os.path.join(*self.dcm_img_info.ReferencedFileID)

    # property: FileID

    @property
    def instance_number(self) -> int:
        return int(self.dcm_img_info.InstanceNumber)

    # property: instance_number

# class: ImageInfo

class Series:
    def __init__(self, dicom_series: pydicom.Dataset) -> None:
        REQ_ATTRS = [
            "SeriesDate",
            "SeriesNumber",
        ]

        for attr_name in REQ_ATTRS:
            assert hasattr(dicom_series, attr_name), f"DICOM Dataset object does not have '{attr_name}' attribute"

        self.dcm_series = dicom_series

    # __init__

    @property
    def date(self) -> datetime.datetime:
        return datetime.datetime.strptime(
            self.dcm_series.SeriesDate,
            "%Y%m%d"
        )

    # property: date

    @property
    def series_number(self) -> int:
        return int(self.dcm_series.SeriesNumber)

    # property: series_number

    def get_all_image_infos(self) -> List[ImageInfo]:
        return [
            ImageInfo(img_info)
            for img_info in self.dcm_series.children
            if img_info.DirectoryRecordType == "IMAGE"
        ]
    
    # get_all_image_infos


# class: Series

class Study:
    def __init__(self, dicom_study: pydicom.Dataset) -> None:
        REQ_ATTRS = [
            "StudyID",
            "StudyDate",
            "StudyTime",
        ]

        for attr_name in REQ_ATTRS:
            assert hasattr(dicom_study, attr_name), f"DICOM Dataset object does not have '{attr_name}' attribute"

        self.dcm_study = dicom_study

        self.study_datetime = datetime.datetime.strptime(
            f"{self.dcm_study.StudyDate}_{self.dcm_study.StudyTime}",
            "%Y%m%d_%H%M%S.%f"
        )

    # __init__

    @property
    def id(self) -> str:
        return self.dcm_study.StudyID

    # property: id

    def get_all_series(self) -> List[Series]:
        return [
            Series(series)
            for series in self.dcm_study.children
            if series.DirectoryRecordType == "SERIES"
        ]

    # get_all_series

# class: Study

class Patient:
    def __init__(self, dicom_patient: pydicom.Dataset):
        REQ_ATTRS = [
            "PatientName",
            "PatientID",
        ]

        for attr_name in REQ_ATTRS:
            assert hasattr(dicom_patient, attr_name), f"DICOM Dataset object does not have '{attr_name}' attribute"

        self.dcm_patient = dicom_patient

    # __init__

    @property
    def name(self):
        return self.dcm_patient.PatientName

    # property: name

    @property
    def id(self):
        return self.dcm_patient.PatientID

    # property: id

    @property
    def studies(self) -> List[Study]:
        return [
            Study(study)
            for study in self.dcm_patient.children
            if study.DirectoryRecordType == "STUDY"
        ]

    # studies

# class: Patient

class DICOMDIR:
    def __init__(self, dicomdir_file: str):
        self.dicom_dir = pydicom.dcmread(dicomdir_file)

    # __init__

    @property
    def patients(self) -> List[Patient]:
        return list(map(Patient, self.dicom_dir.patient_records))

    # property: patients

# class: DICOMDIR

class Image3D:
    def __init__(self) -> None:
        # File management
        self.files                                  = None
        self.slices: Dict[str, pydicom.FileDataset] = dict()

        # image properties
        self.pixel_spacing  : Tuple[float, float] = None
        self.slice_thickness: float               = None

        # meta information
        self.time: datetime.datetime = None

        # 3D image
        self.image: np.ndarray = None

    # __init__

    def __len__(self):
        return [
            len(self.slices)
            if self.image is None
            else self.image.shape[2]
        ]

    # __len__

    @property
    def aspect_ratio_axial(self):
        return self.pixel_spacing[1] / self.pixel_spacing[0]

    # property: aspect_ratio_axial

    @property
    def aspect_ratio_coronal(self):
        return self.slice_thickness / self.pixel_spacing[0]

    # property: aspect_ratio_coronal

    @property
    def aspect_ratio_saggital(self):
        return self.pixel_spacing[1] / self.slice_thickness

    # property: aspect_ratio_saggital

    @property
    def aspect_ratios(self):
        return {
            "axial"   : self.aspect_ratio_axial,
            "coronal" : self.aspect_ratio_coronal,
            "saggital": self.aspect_ratio_saggital,
        }

    # property: aspect_ratios

    def save(self, outfile: str):
        np.savez(
            outfile,
            image=self.image,
            pixel_spacing=np.asarray(self.pixel_spacing),
            slice_thickness=self.slice_thickness,
            time=self.time,
            allow_pickle=True,
        )

    # save

    @classmethod
    def from_npz_file(cls, npz_file: str):
        dcm_image3d = cls()

        with np.load(npz_file, allow_pickle=True) as npz_data:
            dcm_image3d.image           = npz_data["image"]
            dcm_image3d.pixel_spacing   = tuple(npz_data["pixel_spacing"])
            dcm_image3d.slice_thickness = npz_data["slice_thickness"].item()
            dcm_image3d.time            = npz_data["time"].item()

        # with

        return dcm_image3d

    # from_npz_file

    @classmethod
    def from_path(
        cls,
        dir: str,
        file_list: List[str] = None,
        expected_num_slices: int = None,
    ):
        dcm_image3d = cls()

        dcm_image3d.files  = (
            glob.glob(os.path.join(dir, "*")) 
            if file_list is None else 
            file_list
        )
        dcm_image3d.slices = dict()

        # load the slices and 3D image
        for file in dcm_image3d.files:
            dcm_file = pydicom.dcmread(file)
            if not hasattr(dcm_file, "SliceLocation"):
                continue

            dcm_image3d.slices[file] = dcm_file

        # for

        assert len(dcm_image3d.slices) > 0, "No proper slices found"
        if expected_num_slices is not None:
            assert len(dcm_image3d.slices) == expected_num_slices, (
                f"Number of slices is {len(dcm_image3d.slices)} != {expected_num_slices} expected"
            )

        # if

        dcm_image3d.slices = dict(
            sorted(
                dcm_image3d.slices.items(),
                key=lambda key_val: key_val[1].SliceLocation
            )
        )

        for i, (file, slice) in enumerate(dcm_image3d.slices.items()):
            # handle initializations
            if dcm_image3d.pixel_spacing is None:
                dcm_image3d.pixel_spacing = slice.PixelSpacing

            if dcm_image3d.slice_thickness is None:
                dcm_image3d.slice_thickness = float(slice.SliceThickness)

            if dcm_image3d.time is None:
                dcm_image3d.time = datetime.datetime.strptime(
                    f"{slice.ContentDate}_{slice.ContentTime}",
                    "%Y%m%d_%H%M%S"
                )

            # if

            if dcm_image3d.image is None:
                dcm_image3d.image = np.zeros(
                    (
                        *slice.pixel_array.shape,
                        len(dcm_image3d.slices),
                    ),
                    dtype=slice.pixel_array.dtype,
                )

            # if

            # handle pixel array
            dcm_image3d.image[:, :, i] = slice.pixel_array

        # for

        return dcm_image3d

    # from_path

# class: Image3D