import os

def delete_files_in_directory(directory_path):
    """
    Function that deletes everything in the specified directory.
    """
   try:
     with os.scandir(directory_path) as entries:
       for entry in entries:
         if entry.is_file():
            os.unlink(entry.path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")
