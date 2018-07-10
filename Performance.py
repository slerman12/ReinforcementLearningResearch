from __future__ import division
import sys
import pandas as pd
from numpy import mean


class Performance:
    # Initialize variables
    metrics = {}
    filename = None

    def __init__(self, metric_names, episodes_or_run_throughs_per_epoch, filename=None, print_output=True):
        # Initialize variables
        self.metric_names = metric_names
        self.filename = filename
        self.episodes_or_run_throughs_per_epoch = episodes_or_run_throughs_per_epoch
        self.print_output = print_output

        # Empty metrics variable
        for name in self.metric_names:
            self.metrics[name] = []

        # Initialize progress variable
        self.progress = None

        # Create file
        if filename is not None:
            pd.DataFrame(data=self.metrics).to_csv(filename, index=False, columns=metric_names)

    def measure_performance(self, performance):
        # Initialize progress
        if self.progress is None:
            self.progress = Progress(0, self.episodes_or_run_throughs_per_epoch - 1, "Epoch", False)
        else:
            # Set progress parameters
            self.progress.show = self.print_output
            self.progress.progress_total = self.episodes_or_run_throughs_per_epoch

            # Reset progress if needed
            if self.progress.progress_complete == self.progress.progress_total:
                self.progress.reset()

        # Add metrics
        for key in performance:
            self.metrics[key].append(performance[key])

        # Update progress
        if self.progress is not None:
            self.progress.update_progress()

    def output_performance(self, episode_or_run_through, description=None, aggregation=mean, special_aggregation=None):
        # Default no special aggregation
        if special_aggregation is None:
            special_aggregation = {}

        # End epoch
        if episode_or_run_through % self.episodes_or_run_throughs_per_epoch == 0 or episode_or_run_through == 1:
            # Print performance
            if self.print_output:
                # Print description and epoch
                if description is not None:
                    print(description)
                print("Epochs: {}".format(episode_or_run_through // self.episodes_or_run_throughs_per_epoch))

                # Print metrics
                for key in self.metrics:
                    print("* {}: {}".format(key, aggregation(self.metrics[key]) if key not in special_aggregation.keys()
                    else special_aggregation[key](self.metrics[key])))
                print("")

            # Output metrics to file
            if self.filename is not None:
                with open(self.filename, 'a') as data_file:
                    pd.DataFrame(data=self.metrics).to_csv(data_file, index=False, header=False,
                                                           columns=self.metric_names)

            # Empty metrics variable
            for name in self.metric_names:
                self.metrics[name] = []

    def reset(self):
        # Empty metrics variable
        for name in self.metric_names:
            self.metrics[name] = []

        # Re-initialize progress
        if self.progress is None:
            self.progress = Progress(0, self.episodes_or_run_throughs_per_epoch - 1, "Epoch", self.print_output)
        else:
            self.progress.progress_total = self.episodes_or_run_throughs_per_epoch
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
