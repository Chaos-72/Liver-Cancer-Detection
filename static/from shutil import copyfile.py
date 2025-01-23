from shutil import copyfile

# Source file path (where it was saved)
source_path = "/mnt/data/Liver_Cancer_Research_Paper.docx"

# Destination file path (your local directory)
destination_path = "D:\Liver_Cancer_Research_Paper.docx"

# Copy the file to a local directory
copyfile(source_path, destination_path)
print("File successfully copied to your local directory!")
