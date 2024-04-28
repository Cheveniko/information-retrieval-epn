# Script to rename files to specific format
import os
import glob

books = glob.glob("../data/*")
for index, book in enumerate(books):
    os.rename(book, f"../data/book{index + 1}.txt")
