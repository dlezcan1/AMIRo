import craedl
import glob

# get the profile
profile = craedl.auth()

# get the Needle Shape Sensing Project
project = profile.get_project('R01 - Needle Shape Sensing')

# find the home data directory
home = project.get_data()
data_dir = home.get('Needle Data')

# get all of the data pieces
dirs = glob.glob("needle*")

for d in dirs:
    print("Uploading: ", d)
    while True: 
        try:
            data_dir = data_dir.upload_directory(d,
                                    rescan=False,
                                    output=True)
            break
        
        # try        
        except:
            continue
            
        # except
    # while
    
    print("Upload finished", end='\n\n')
# for

# termination
print("All uploads have completed.")