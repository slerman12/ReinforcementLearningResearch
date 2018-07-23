from __future__ import division
import datetime
import sys
import pandas as pd
import time
from numpy import mean
import tensorflow as tf


class Performance:
    # Initialize variables
    metrics = {}
    filename = None

    def __init__(self, metric_names, run_throughs_per_epoch=None, filename=None, print_output=True, description=None):
        # Initialize variables
        self.metric_names = metric_names
        self.filename = filename
        self.run_throughs_per_epoch = run_throughs_per_epoch
        self.epochs = 0
        self.print_output = print_output
        self.time_training_began = time.time()
        self.tensorboard = None
        self.get_tensorboard_logs = None

        # Empty metrics variable
        for name in self.metric_names + ["Total Elapsed Time"]:
            self.metrics[name] = []

        # Initialize progress variable
        self.progress = None

        # Create file
        if filename is not None:
            pd.DataFrame(data=self.metrics).to_csv(filename, index=False, columns=metric_names)

        if description is not None:
            print(description)

    def measure_performance(self, performance):
        # Initialize progress
        if self.progress is None:
            self.progress = Progress(0, self.run_throughs_per_epoch - 1, "Epoch", False)
        else:
            # Set progress parameters
            self.progress.show = self.print_output
            self.progress.progress_total = self.run_throughs_per_epoch

            # Reset progress if needed
            if self.progress.progress_complete == self.progress.progress_total:
                self.progress.reset()

        # Add metrics
        self.metrics["Total Elapsed Time"].append(str(datetime.timedelta(seconds=round(time.time() -
                                                                                       self.time_training_began))))
        for key in performance:
            self.metrics[key].append(performance[key])

        # Update progress
        if self.progress is not None:
            self.progress.update_progress()

    def output_performance(self, run_through, description=None, aggregation=mean, special_aggregation=None,
                           logs_for_tensorboard=None):
        # Default no special aggregation
        if special_aggregation is None:
            special_aggregation = {}

        # Logs for tensorboard
        if logs_for_tensorboard is not None:
            if self.tensorboard is None:
                self.tensorboard = tf.summary.FileWriter("Logs", graph=tf.get_default_graph())
            if logs_for_tensorboard is True:
                self.tensorboard.flush()
            else:
                self.tensorboard.add_summary(logs_for_tensorboard, run_through)

        # End epoch
        if run_through % self.run_throughs_per_epoch == 0 or run_through == 1:
            # Print performance
            if self.print_output:
                # Print description and epoch
                if description is not None:
                    print(description)
                print("Epochs: {}, Total Elapsed Time: {}".format(run_through // self.run_throughs_per_epoch,
                      datetime.timedelta(seconds=round(time.time() - self.time_training_began))))

                # Print metrics
                for key in self.metrics:
                    if key != "Total Elapsed Time":
                        print("* {}: {}".format(key, aggregation(self.metrics[key]) if key not in special_aggregation
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

            return True
        return False

    def is_epoch(self, run_through):
        return run_through % self.run_throughs_per_epoch == 0 or run_through == 1

    def logs_for_tensorboard(self, scalars=None, gradients=None, variables=None):
        # Default lists
        if scalars is None:
            scalars = {}

        # Add scalars to logs
        for name in scalars:
            tf.summary.scalar(name, scalars[name])

        # Create summaries to visualize weights
        if variables is not None:
            for variable in variables:
                tf.summary.histogram(variable.name, variable)

        # Summarize all gradients
        if gradients is not None:
            if variables is None:
                variables = tf.trainable_variables()
            for gradient, variable in list(zip(gradients, variables)):
                tf.summary.histogram(variable.name + '/gradient', gradient)

        # Merge all logs into a single operation to be run by the agent
        self.get_tensorboard_logs = tf.summary.merge_all()

    def reset(self):
        # Empty metrics variable
        for name in self.metric_names:
            self.metrics[name] = []

        # Re-initialize progress
        if self.progress is None:
            self.progress = Progress(0, self.run_throughs_per_epoch - 1, "Epoch", self.print_output)
        else:
            self.progress.progress_total = self.run_throughs_per_epoch
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
