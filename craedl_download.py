import craedl

# get the profile
profile = craedl.auth()

# get the Needle Shape Sensing Project
project = profile.get_project('R01 - Needle Shape Sensing')

# find the home data directory
home = project.get_data()
data_dir = home.get('Needle Data')

data_dir.download("data/",
                    rescan=False,
                    output=True )

print("Downloaded craedl dataset")