''''
    This python script generates the box-whiskers plot for each csv file inside AmazonBookReviewsDS in a single window.
    Motivation : 
        1. Visualize how the ratings are spread (low, Q1, Q2, Q3, high and outliers)
        2. Compare the plots among other Books
        3. Understand the positivity/negativity in ratings
    Dataset : http://archive.ics.uci.edu/ml/datasets/Amazon+book+reviews (UCI Machine Learning Repository)
    Total Number of Reviews comprising 8 books : 213,335
'''

# Import the necessary python modules
import matplotlib.pyplot as pyplot
import csv
import os
import glob
import numpy

# Temporay storage for holding data from each file
dsfile_data_temp = []

# Holds the data collected from all the files (represented as list of numpy arrays)
boxplot_data = []

# Use the file names as yticklabels
book_names = []

# Read all the csv files inside the directory AmazonBookReviewsDS (uncompressed dataset)
path_csv_all = os.path.join(os.path.dirname(__file__),'AmazonBookReviewsDS/*.csv')
for path_csv in glob.glob(path_csv_all):
    # Open the current file 
    with open(path_csv) as dsfile:
        # Read the file as tab seperated using csv reader
        dataset = csv.reader(dsfile,delimiter='\t')
        # Take only the ratings column
        for row in dataset:
            dsfile_data_temp.append(float(row[0]))
        # Create a numpy array for each file and add it to boxplot_data
        boxplot_data.append(numpy.array(dsfile_data_temp))
        # Generate the name for the corresponding yticklabel
        book_names.append(os.path.basename(path_csv)+'\n ( Total Reviews:'+str(len(dsfile_data_temp))+' )')
        # Initialize it to process the next file
        dsfile_data_temp = []

# Draw box plot from the data collected from all csv files
figure,axes = pyplot.subplots(figsize=(50,50))
figure.canvas.set_window_title('Amazon Book Reviews - Box Plot')
axes.set_title('Amazon Book Reviews - Box Plot [Total : 8 Books]')
axes.set_xlabel('Ratings (1 to 5)')
axes.set_ylabel('Books')
axes.set_xlim(0.5, 5.5)
pyplot.boxplot(boxplot_data,0,'rs',0)
pyplot.setp(axes,yticklabels=book_names)
pyplot.show()