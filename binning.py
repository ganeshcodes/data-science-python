''''
    This python script generates the multi-colored histogram by reading all csv files inside AmazonBookReviewsDS in a single window.
    Motivation : 
        1. Visualize how the ratings are spread (low, Q1, Q2, Q3, high and outliers)
        2. Compare the histograms among other Books
        3. Understand the positivity/negativity in ratings for each book
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
plot_data = []

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
        # Create a numpy array for each file and add it to plot_data
        plot_data.append(numpy.array(dsfile_data_temp))
        # Generate the name for the corresponding yticklabel
        book_names.append(os.path.basename(path_csv))
        # Initialize it to process the next file
        dsfile_data_temp = []

# Draw histogram (binning) from the data collected from all csv files
figure,axes = pyplot.subplots(figsize=(50,50))
figure.canvas.set_window_title('Amazon Book Reviews - Histogram (Singleton Ratings Buckets)')
axes.legend(prop={'size': 10})

colors = ['green','blue','red','yellow','orange','tan','lime','purple']
pyplot.hist(plot_data, color=colors, label=book_names)
pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
pyplot.xlabel("Rating (1-5)")
pyplot.ylabel("Count/Number of reviews")
pyplot.show()