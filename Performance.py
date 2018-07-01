from __future__ import division
import sys
import pandas as pd
from numpy import mean


class Performance:
    # Initialize variables
    metrics = {}
    filename = None

    def __init__(self, metric_names, epoch, filename=None):
        # Initialize variables
        self.metric_names = metric_names
        self.filename = filename
        self.epoch = epoch

        # Empty metrics variable
        for name in self.metric_names:
            self.metrics[name] = []

        # Initialize progress variable
        self.progress = Progress(0, epoch, "Epoch", True)

        # Create file
        pd.DataFrame(data=self.metrics).to_csv('Results/{}.csv'.format(filename), index=False, columns=metric_names)

    def measure_performance(self, performance):
        # Add metrics
        for key in performance:
            self.metrics[key].append(performance[key])

        # Update progress
        self.progress.update_progress()

    def output_performance(self, run_through, description="Performance", aggregation=mean):
        # End epoch
        if not run_through % self.epoch:
            # Output performance
            print(description)
            print("Epoch: {}".format(run_through / self.epoch))

            # Print metrics
            for key in self.metrics:
                print("* {}: {}".format(key, aggregation(self.metrics[key])))
            print("")

            # Output metrics to file
            if self.filename is not None:
                with open('Results/{}.csv'.format(self.filename), 'a') as data_file:
                    pd.DataFrame(data=self.metrics).to_csv(data_file, index=False, header=False,
                                                           columns=self.metric_names)

            # Reset metrics
            self.reset()

    def reset(self):
        # Empty metrics variable
        for name in self.metric_names:
            self.metrics[name] = []

        # Re-initialize progress
        self.progress.reset()


# Display progress in console
class Progress:
    # Initialize progress measures
    progress_complete = 0.00
    progress_total = 0.00
    name = ""
    show = True

    def __init__(self, pc, pt, name, show):
        # Initialize variables
        self.progress_complete = pc
        self.progress_total = pt
        self.name = name
        self.show = show
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(0, name))
            sys.stdout.flush()

    def update_progress(self):
        # Update progress
        self.progress_complete += 1.00
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(self.progress_complete / self.progress_total, self.name))
            sys.stdout.flush()
        if (self.progress_complete == self.progress_total) and self.show:
            print("")

    def reset(self):
        self.progress_complete = 0.00
        if self.show:
            sys.stdout.write("\rProgress: {:.2%} [{}]".format(0, self.name))
            sys.stdout.flush()
