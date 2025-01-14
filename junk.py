

def validate_zip_file(zip_path):
    try:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP file not found at {zip_path}")

        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("Validating ZIP file...")
            # Test the integrity of the ZIP file
            bad_file = zip_ref.testzip()
            if bad_file:
                raise zipfile.BadZipFile(f"Corrupted file found: {bad_file}")
            else:
                print("ZIP file is valid.")
            return True
    except zipfile.BadZipFile as e:
        print(f"Bad ZIP file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False
try:
    import zipfile
    from PIL import Image

    import numpy as np
    import os
    import traceback
    
    path_to_zip = r"/home/icb/milad.bassil/Desktop/GRAE/data/COIL100/coil100.zip"
    root =  r'/home/icb/milad.bassil/Desktop/GRAE/data/COIL100'
    is_valid = validate_zip_file(path_to_zip)
    if is_valid:
        print("Proceed with extraction.")
    else:
        print("Fix or replace the ZIP file.")
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(path=root)

    x = list()
    y = list()
    for i in range(1, 101):
        print(i)
        for j in range(0, 360, 5):
            img = Image.open(os.path.join(root, 'coil-100', 'coil-100', f'obj{i}__{j}.png'))
            print(i)
            img.load()
            img = np.asarray(img, dtype='int32')
            x.append(img.flatten()/255)
            y.append(np.array([i, j]))
except Exception as e:
    # Print the exception type and message
    print(f"An error occurred: {e}")
    
    # Print the full traceback
    traceback.print_exc()


x = np.vstack(x)
y = np.vstack(y)
np.save(os.path.join(root, 'x.npy'), x)
np.save(os.path.join(root, 'y.npy'), y)
