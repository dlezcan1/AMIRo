import datetime
import json
import os
import re

import dicom

from typing import (
    Dict,
    Tuple
)


def main():
    in_dir = "/home/dlezcan1/data/7CH-4AA-0001-MCF-even/2023-06-15_2023-06-16_Beef-Insertion-Experiment/ct_images/mcf-beef-insertion-CT/"

    dicom_dir = dicom.DICOMDIR(os.path.join(in_dir, "BLDICOMDIR"))

    print("Loading CT Scan informations...")
    series_paths: Dict[str, Tuple[dicom.Patient, dicom.Study, dicom.Series]] = dict()
    for patient in dicom_dir.patients:
        for study in patient.studies:
            for series in study.get_all_series():
                img_infos = series.get_all_image_infos()
                if len(img_infos) < 100:
                    continue

                ser_path = os.path.split(img_infos[0].fileID)[0]

                series_paths[ser_path] = (
                    patient,
                    study,
                    series,
                )

            # for: series
        # for: study
    # for: patient
    
    print("Processing raw CT scans...")
    for path, (patient, study, series) in series_paths.items():
        ct_image = dicom.Image3D.from_path(os.path.join(in_dir, f"{path}.dicom"))
        
        odir = os.path.join(
            in_dir,
            "results",
            study.study_datetime.strftime("%Y-%m-%d"),
            ct_image.time.strftime("%Y-%m-%d_%H-%M-%S"),
        )
        os.makedirs(odir, exist_ok=True)
        outfile = os.path.join(odir, "ct_scan.npz")

        ct_image.save(outfile)
        print("Saved CT scan to:", outfile)

    # for
    
    print("Finished.")

# main

if __name__ == "__main__":
    main()

# if __main__