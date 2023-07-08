## Dependencies 
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from fairness.algorithms.Algorithm import Algorithm



class RavaAlgorithm(Algorithm):
    """
    This is the base class for all implemented algorithms.  New algorithms should extend this
    class, implement run below, and set self.name in the init method.  Other optional methods to
    implement are described below.
    """

    def __init__(self):
        Algorithm.__init__(self)
        self.name = "Rava"

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given
        train and test data still contains the sensitive_attrs.  This run of the algorithm
        should focus on the single given sensitive attribute.

        params: a dictionary mapping from algorithm-specific parameter names to the desired values.
        If the implementation of run uses different values, these should be modified in the params
        dictionary as a way of returning the used values to the caller.

        Be sure that the returned predicted classifications are of the same type as the class
        attribute in the given test_df.  If this is not the case, some metric analyses may fail to
        appropriately compare the returned predictions to their desired values.

        TODO: figure out how to indicate that an algorithm that can handle multiple sensitive
        attributes should do so now.
        """
        d_train = train_df.sample(frac=0.5)
        d_cal = train_df.drop(d_train.index)
        d_test = test_df

        ## Separate out the data, the protected attribute, and the classification label
        ## Make sure these are all pandas series
        ## Separate out the data, the protected attribute, and the classification label
        ## Make sure these are all pandas series
        X_train = d_train.drop(class_attr, axis=1)
        y_train = d_train[class_attr]
        y_train = pd.to_numeric(y_train, errors='coerce')
        
        X_cal = d_cal.drop([class_attr], axis=1)
        y_cal = d_cal[class_attr]
        A_cal = d_cal[sensitive_attrs]
        
        X_test = d_test.drop([class_attr], axis=1)
        A_test = d_test[sensitive_attrs]
        
        # Step 1: Train a model on d_train
        X_train_num = X_train.select_dtypes(include='number')
        classifier = GaussianNB()
        classifier.fit(X_train_num, y_train)
        
        # Predict probabilities for d_test
        X_test_num = X_test.select_dtypes(include='number')
        s_test = classifier.predict_proba(X_test_num)
        s_test = pd.DataFrame(s_test)
        
        # Predict probabilities for d_cal
        X_cal_num = X_cal.select_dtypes(include='number')
        s_cal = classifier.predict_proba(X_cal_num)
        s_cal = pd.DataFrame(s_cal)
        
        
        # Initialize matrix to store rval values
        class_attr_unique = set(d_test[class_attr])
        class_cur_values = class_attr_unique
        rval_storage = np.zeros((len(s_test), len(class_cur_values)))

        A_cal = A_cal.reset_index()
        A_cal = A_cal.iloc[:, 1]
        A_test = A_test.reset_index()
        A_test = A_test.iloc[:, 1]
        y_cal = y_cal.reset_index()
        y_cal = y_cal.iloc[:, 1]
        
        for i, s_cur_bclass in s_test.iterrows():
            a_cur = A_test[i]
        
            # For this corresponding protected group, extract the scores and labels from the calibration and test data
            a_numerator_idx = A_cal[A_cal == a_cur].index.tolist()
            s_cal_a = s_cal.loc[a_numerator_idx]
            y_cal_a = y_cal.loc[a_numerator_idx]
        
            a_denominator_idx = A_test[A_test == a_cur].index.tolist()
            s_test_a = s_test.loc[a_denominator_idx]
            
            # Using the s_a scores for both test and cal, estimate an R-value
            for j, class_cur in enumerate(class_cur_values):
                s_cur = s_cur_bclass[class_cur]
                num_rval = (1 / (len(s_cal_a) + 1)) * (
                        ((s_cal_a[class_cur] > s_cur) & (y_cal_a != class_cur)).sum() + 1)
                denom_rval = (1 / (len(s_test_a) + len(s_cal_a) + 1)) * (
                        ((s_cal_a[class_cur] > s_cur)).sum() + ((s_test_a[class_cur] > s_cur)).sum() + 1)
                rval = num_rval / denom_rval
                rval_storage[i, j] = rval

        ## rval_storage is a matrix with both the high and low risk R-values. Use these for classification
        alpha = 0.1
        classification = np.zeros(rval_storage.shape[0])  # Initialize classification array

        for i in range(rval_storage.shape[0]):
            high_risk_val = rval_storage[i, 0]
            low_risk_val = rval_storage[i, 1]

            if high_risk_val <= alpha and low_risk_val <= alpha:
                if high_risk_val <= low_risk_val:
                    classification[i] = class_attr_unique[0]
                else:
                    classification[i] = class_attr_unique[1]
            elif high_risk_val <= alpha:
                classification[i] = class_attr_unique[0]
            elif low_risk_val <= alpha:
                classification[i] = class_attr_unique[1]
            else:
                classification[i] = 0

            ## We now have the classifications 2, 1 and 0 for all observations in d_test
            return classification, []

    def get_param_info(self):
        """
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to
        be explored.  This function should only be implemented if the algorithm has specific
        parameters that should be tuned, e.g., for trading off between fairness and accuracy.
        """
        return {}

    def get_supported_data_types(self):
        """
        Returns a set of datatypes which this algorithm can process.
        """
        return set(["numerical-binsensitive", "numerical"])

    def get_name(self):
        """
        Returns the name for the algorithm.  This must be a unique name, so it is suggested that
        this name is simply <firstauthor>.  If there are mutliple algorithms by the same author(s), a
        suggested modification is <firstauthor-algname>.  This name will appear in the resulting
        CSVs and graphs created when performing benchmarks and analysis.
        """
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with
        the algorithm.  If not implemented by a specific algorithm, this returns the empty
        dictionary.
        """
        return {}
