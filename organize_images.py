"""
Author: Dr. J (github.com/TBSDrJ)
Date: Nov 13 2025
Purpose: Organize images into folders. Linux/Mac only.
"""

"""
Starting assumptions (Feel encouraged to rename stuff.  It'll probably take 
some work to get things set up like this):
    - This code lives in, and is being run in a folder with at least 4 
    things in it:
        1. This code
        2. A folder called "images" that has all the images in it, none of 
        them are in subfolders, there is nothing else in this folder besides
        images.
        3. A file called "textfile.txt" that tells you which category each
        image is in.
        4. A folder called "data" that only has folders in it.  There is one 
        subfolder for each category, the name of the subfolder matches the 
        name of the category.  Those folders start empty.
"""
import pathlib
import subprocess

def get_categories() -> dict:
    with open("textfile.txt", "r") as f:
        lines = list(f.readlines())
    categories = {}
    for line in lines:
        # Get image_file and category from each line of the text file.
        # If a line doesn't work, use 'continue' to skip the rest of the loop.
        
        # Add your code here to read each line of text.

        categories[image_file] = category
    return categories

def test_run(categories: dict, path: pathlib.Path) -> None:
    """First run, for testing.

    You should copy-paste at least one or two of these into command-line and
      run them to make sure they work.  Check the destination folder and
      make sure the image ends up in the right place."""    
    for count, image_file in enumerate(path.iterdir()):
        print("mv", f"images/{image_file.name}", 
                f"data/{categories[image_file.name]}"),
        if count > 4: break

def real_run(categories: dict, path: pathlib.Path) -> None:
    """Actually move all the files if the test passed."""
    for image_file in path.iterdir():
        proc = subprocess.run(["mv", image_file, 
                f"data/{categories[image_file.name]}"], capture_output=True)
        if proc.stderr:
            print(f"[ERROR] {proc.stderr.decode()}")
        if proc.stdout:
            print(f"[INFO] {proc.stdout.decode()}")

def main():
    categories = get_categories()
    # This will allow us to iterate through all the images.
    path = pathlib.Path("images")
    # First run, use the next line.  For the real run, comment it out.
    test_run(categories, path)
    # Real run. Un-comment the next line if test is OK.
    # real_run(categories, path)
    


if __name__ == "__main__":
    main()
