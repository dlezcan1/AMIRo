import datetime
import json
import os
import re

def main():
    in_dir = "~/data/2023-06-15/mcf-beef-insertion-CT"

    ct_info = None
    with open(os.path.join(in_dir, "BLDICOMDIR.json"), "r") as json_file:
        ct_info = json.load(json_file)

    # with

    

    for study in ct_info["studies"]:
        study: dict
        
        # parse the date
        year, month, day = re.match(
            r"(\d{4})(\d{2})(\d{2})",
            study["Payload"]["StudyDate"],
        ).groups()

        hour, minute, second, microsecond = re.match(
            r"(\d{2})(\d{2})(\d{2})\.(\d+)",
            study["Payload"]["StudyTime"]
        )

        date = datetime.datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hour),
            minute=int(minute),
            second=int(second),
            microsecond=(int(microsecond[:6])),
        )

        # parse the image files
        for image in study["Series"]["Images"]:
            image: dict

            ref_file_id = os.path.normpath( image["Payload", "ReferencedFileID"] )
            


        # for


    # for




# main

if __name__ == "__main__":
    main()

# if __main__