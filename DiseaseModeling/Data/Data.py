import pandas as pd
import numpy as np
from sklearn import preprocessing
import random


# Data handler
class ReadPD:
    def __init__(self, filename, targets, to_drop=None):
        # Processed data file
        file = pd.read_csv(filename)

        # Default drop list
        to_drop = ["PATNO"] if to_drop is None else ["PATNO"] + to_drop

        # Data
        self.PD_Data = []

        # Max number of records for any patient
        self.max_num_records = file.groupby(["PATNO"]).size().max() - 1

        # Dimension of a patient record
        self.patient_record_dim = len(file.drop(to_drop, axis=1).columns.values)

        # Dimension of input
        self.input_dim = self.patient_record_dim - len(targets)

        # Dimension of desired output
        self.desired_output_dim = len(targets)

        # Variable for iterating batches
        self.batch_begin = 0

        # List of inputs, desired outputs, and sequence lengths
        for patient in file["PATNO"].unique():
            # All patient records
            patient_records = file[file["PATNO"] == patient].drop(to_drop, axis=1)

            # Sequence lengths
            length = patient_records.shape[0] - 1

            # Inputs + padding
            inputs = np.zeros((self.max_num_records, self.input_dim), dtype=np.float64)
            inputs[:length] = patient_records.drop(targets, axis=1).values[:-1]

            # Desired outputs + padding (desired outputs are: next / previous)
            desired_outputs = np.zeros((self.max_num_records, self.desired_output_dim), dtype=np.float64)
            desired_outputs[:length] = patient_records[targets].values[1:]

            # # Desired outputs + padding (desired outputs are: next / previous)
            # desired_outputs[:length] = np.divide(patient_records[targets].values[1:],
            #                                      patient_records[targets].values[:-1],
            #                                      out=np.zeros_like(patient_records[targets].values[1:]),
            #                                      where=patient_records[targets].values[:-1] != 0)

            # # See which changes are very big
            # for ind, i in enumerate(desired_outputs[:length]):
            #     if (np.absolute(i) > 10).any():
            #         print("\nLarge values in targets:")
            #         print("patient: {}".format(patient))
            #         print(i)
            #         print(patient_records[targets].values[1:][ind])
            #         print(patient_records[targets].values[:-1][ind])

            # Add patient records to PD data
            self.PD_Data.append({"id": patient, "inputs": inputs, "desired_outputs": desired_outputs, "length": length})

        # Shuffle data
        random.shuffle(self.PD_Data)

    def iterate_batch(self, batch_size):
        # Reset and shuffle batch when all items have been iterated
        if self.batch_begin > len(self.PD_Data) - batch_size:
            # Reset batch index
            self.batch_begin = 0

            # Shuffle PD data
            random.shuffle(self.PD_Data)

        # Index of the end boundary of this batch
        batch_end = min(self.batch_begin + batch_size, len(self.PD_Data))

        # Batch
        batch = self.PD_Data[self.batch_begin:batch_end]

        # Update batch index
        self.batch_begin = batch_end

        # Return inputs, desired outputs, and lengths
        return np.stack([patient_records["inputs"] for patient_records in batch]), \
            np.stack([patient_records["desired_outputs"] for patient_records in batch]), \
            np.array([patient_records["length"] for patient_records in batch])


# Count missing variables per variable and visit and output summary to csv
def count_missing_values(data):
    # Variables with no records per patient
    missing = pd.DataFrame(data.groupby("PATNO").apply(lambda x: x.isnull().all()).sum(axis=0),
                           columns=["% Patients With No Record For Any Visit"])

    # Percent
    missing /= float(len(data["PATNO"].unique()))

    # Sort
    missing = missing.sort_values(by="% Patients With No Record For Any Visit")

    # Output file
    missing.to_csv("Stats/all_values_missing_per_patient_variable.csv")

    print("\nMean % of patients with no record at all for a variable: {0:.0%}".format(
        missing["% Patients With No Record For Any Visit"].mean()))

    # Return missing
    return missing


# Drop
def drop(data, to_drop=None):
    # Drop patients with only one observation
    num_observations = data.groupby(["PATNO"]).size().reset_index(name='num_observations')
    patients_with_only_one_observation = num_observations.loc[num_observations["num_observations"] == 1, "PATNO"]
    data = data[~data["PATNO"].isin(patients_with_only_one_observation)]

    print("\nNumber of patients dropped (due to them only having one observation): {}".format(
        len(patients_with_only_one_observation)))

    # Count missing
    missing = count_missing_values(data)

    # Cutoff for missing values
    cutoff = 0.6

    # Drop variables with too many missing
    data = data[missing[missing["% Patients With No Record For Any Visit"] < cutoff].index.values]

    print("\nNumber of variables dropped due to too many patients without any record of them: {} ".format(
        len(missing[missing["% Patients With No Record For Any Visit"] >= cutoff])) + "(cutoff={0:.0%})".format(cutoff))

    # Drop any other specific variables
    if to_drop is not None:
        data = data.drop(to_drop, axis=1)

    print("\nNumber of variables dropped by manual selection (due to, say, duplicates, lack of statistical meaning, or "
          "just being unwanted): {}".format(len(to_drop)))

    print("\nAll dropped variables:")
    print(list(missing[missing["% Patients With No Record For Any Visit"] >= 0.6].index.values) + to_drop)

    # Return data
    return data


# Impute missing values by interpolating by patient
def impute(data):
    # Manually encode meaningful strings
    data.loc[data["tTau"] == "<80", "tTau"] = 50
    data.loc[data["pTau"] == "<8", "pTau"] = 5

    # for column in data.columns.values:
    #     if data[column].isnull().any():
    #         if column in ["Serum Glucose", "ALT (SGPT)", "Serum Bicarbonate", "Albumin-QT", "Total Bilirubin",
    #                       "AST (SGOT)", "tTau", "pTau", "LAMB2(rep2)", "LAMB2(rep1)", "PSMC4(rep1)", "SKP1(rep1)",
    #                       "GAPDH(rep1)", "HSPA8(rep1)", "ALDH1A1(rep1)"]:
    #             print("\n{}:".format(column))
    #             print(data[column].value_counts())
    #
    # {"PSMC4": "Undetermined", "SKP1": "Undetermined", "LAMB2(rep1)": "Undetermined", "LAMB2(rep2)": "Undetermined",
    #  "tTao": "<80", "pTao": "<8", }

    # Variables that are numeric in nature but mixed in with strings
    coerce_to_numeric = ["Serum Glucose", "ALT (SGPT)", "Serum Bicarbonate", "Albumin-QT", "Total Bilirubin",
                         "AST (SGOT)", "tTau", "pTau", "LAMB2(rep2)", "LAMB2(rep1)", "PSMC4(rep1)", "SKP1(rep1)",
                         "GAPDH(rep1)", "HSPA8(rep1)", "ALDH1A1(rep1)"]

    # Coerce to numeric those numerics mixed with strings
    for column in coerce_to_numeric:
        data[column] = pd.to_numeric(data[column], "coerce")

    # Interpolation by previous or next per patient
    interpolated = data.groupby('PATNO').apply(lambda group: group.fillna(method="ffill").fillna(method="bfill"))

    # Output to file
    interpolated.to_csv("Processed/interpolated.csv")

    # Global median if still missing
    imputed = interpolated.fillna(data.median())

    # Most common occurrence for missing strings
    imputed = imputed.apply(lambda column: column.fillna(column.value_counts().index[0]))

    # Output to file
    imputed.to_csv("Processed/imputed.csv", index=False)

    # Return imputed
    return imputed


# Encode categorical values
def encode(data):
    # Ensure numbers are seen as numeric
    for column in data.columns.values:
        data[column] = pd.to_numeric(data[column], "ignore")

    # List of non-numeric variables
    variables_to_encode = [item for item in data.columns.values if item not in list(
        data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.values)]

    print("\nVariables converted to one-hot binary dummies:")
    print(variables_to_encode)

    # Encode strings to numeric (can also use labelbinarizer for one-hot instead!) TODO
    data[variables_to_encode] = data[variables_to_encode].apply(preprocessing.LabelEncoder().fit_transform)

    # Output to file
    data.to_csv("Processed/encoded.csv", index=False)

    # Return encoded data
    return data


# Process data
def process(filename, to_drop):
    # Get data
    data = pd.read_csv(filename)

    # Drop
    dropped = drop(data, to_drop)

    # Impute
    imputed = impute(dropped)

    # Encode
    data = encode(imputed)

    print("\nMean sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].mean()))
    print("\nMedian sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].median()))
    print("\nMin sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].min()))
    print("\nMax sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].max()))
    print("\nNumber of patients: {}".format(len(data["PATNO"].unique())))
    print("\nNumber of variables: {}".format(len(data.columns.values)))


# Main method
if __name__ == "__main__":
    # Preprocessed data
    preprocessed = "Preprocessed/preprocessed_data_treated_and_untreated_off_PD_GENPD_REGPD.csv"  # TODO: on & off dose

    # Variables to drop
    variables_to_drop = ["EVENT_ID", "GENDER", "GENDER.y", "SXDT", "PDDXDT", "SXDT_x",
                         "PDDXDT_x", "BIRTHDT.x", "INFODT", "INFODT_2", "ENROLL_DATE",
                         "INITMDDT", "INITMDVS", "UPDRS_II_AND_III"]

    # Data processing
    process(preprocessed, variables_to_drop)

    # Processed data
    processed = "Processed/encoded.csv"

    # Data reader
    ReadPD(processed, targets=["UPDRS_III"])
