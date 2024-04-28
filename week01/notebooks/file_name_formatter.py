# Script to rename files to specific format
import os
import glob

books = glob.glob("../data/*")
for index, book in enumerate(books, start=1):
    os.rename(book, f"../data/book{index}.txt")
