
import numpy
import pandas
import seaborn
import matplotlib

class PlotDataset:
    def __init__ (self):
        self.relplot_functions = {
            "scatter": seaborn.scatterplot,
            "line": seaborn.lineplot
        }
        self.displot_functions = {
            "hist": seaborn.histplot,
            "kde": seaborn.kdeplot,
            "ecdf": seaborn.ecdfplot
        }
        self.catplot_functions = {
            "strip": seaborn.stripplot,
            "box": seaborn.boxplot,
            "bar": seaborn.barplot,
            "count": seaborn.countplot
        }
