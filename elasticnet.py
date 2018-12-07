import numpy as np
import enet_helpers
from matplotlib import pyplot as plt

_allowed_objectives = ['l2']

# so that elasticnet.Data is literally just the c++ Data class produced by pybind
Data = enet_helpers.Data

def train(data, params=None):
    input_params = params
    params = {
                'lambda_path': None,
                'objective': 'l2',
                'reg_alpha': .5, 
                'tol': .001, 
                'max_coord_descent_rounds': 1000, 
                'num_threads': 4
            }
    for k,v in input_params.items():
        if k in params.keys():
            params[k] = v
        else:
            raise ValueError("param not recognized: {}".format(k))

    _input_checks(**params)

    # the lambda path = the sequence of global regulaization parameters
    if params['lambda_path'] is not None:
        params['lambda_path'] = sorted(set(params['lambda_path']))
    else:
        pass
        # TODO: implement automatic lambda path generation

    # copy and prepare data for estimation:
    N, D = data.N, data.D
    # intercept is just mean y due to standardization of x
    intercept = data.get_y().mean()

    # estimate parameters of the whole regularization path
    intercept_coef_list = []
    coefs_tmp = np.zeros(D)
    if params['objective']=='l2':
        for reg_lambda in params['lambda_path']:
            print("estimating coefficients for lambda = {:.3e}".format(reg_lambda))
            coefs_init = coefs_tmp # set the most recent computed coefs as the initialization
            coefs_tmp = np.empty(D) # holder for new coefs
            enet_helpers.cyclic_coordinate_descent(data, "l2",
                                                   coefs_init, coefs_tmp, 
                                                   reg_lambda, params['reg_alpha'], 
                                                   params['tol'], params['max_coord_descent_rounds'],
                                                   params['num_threads'])
            intercept_coef_list.append({'reg_lambda':reg_lambda, 'intercept':intercept, 'coefs':coefs_tmp})

    # dump this into a class and return
    out_models = ElasticNetPathModels(params, intercept_coef_list)
    return out_models


def _input_checks(lambda_path, objective, *args, **kwargs):
    """
    """
    if lambda_path is None:
        raise NotImplementedError("automatic regularization path generation not yet implemented,"
                                    + " please manually set lambda_path to a list of positive values")
    if objective not in _allowed_objectives:
        raise NotImplementedError("objective {} not yet implemented, ".format(objective)
                                    +" please choose one of these: {}".format(_allowed_objectives))


class ElasticNetPathModels():
    """
    a class representing a sequence of elastic net models, each one corresponding to some
    global regularization value from a regularization path.
    """

    def __init__(self, params, intercept_coef_list):
        self.params = params
        self._intercept_coef_list = intercept_coef_list

    def get_intercept_coefs(self):
        import pandas as pd
        list_for_pandas = []
        for ic in self._intercept_coef_list:
            coefs = ic['coefs']
            tmp_dict = {'coef_{}'.format(i):coefs[i] for i in range(len(coefs))}
            tmp_dict['reg_lambda'] = ic['reg_lambda']
            tmp_dict['intercept'] = ic['intercept']
            list_for_pandas.append(tmp_dict)
        df = pd.DataFrame.from_records(list_for_pandas).set_index('reg_lambda').reversed
        return df

    def plot_intercept_coefs(self, return_intercept_coef_dataframe=True):
        import pandas as pd
        list_for_pandas = []
        for ic in self._intercept_coef_list:
            coefs = ic['coefs']
            tmp_dict = {'coef_{}'.format(i):coefs[i] for i in range(len(coefs))}
            tmp_dict['reg_lambda'] = ic['reg_lambda']
            tmp_dict['intercept'] = ic['intercept']
            list_for_pandas.append(tmp_dict)
        df = pd.DataFrame.from_records(list_for_pandas).set_index('reg_lambda').sort_index(ascending=False)
        # plot it
        fig = plt.figure(figsize=(6, 4))
        ax = plt.subplot(111)
        for c in df.columns:
            if not c.startswith('coef_'):
                continue
            plt.plot(df.index, df[c], marker='o', markersize=5, linestyle='--')
        plt.xscale("log")
        plt.gca().invert_xaxis()
        plt.xlabel("global regularization strength")
        plt.title("elastic-net coefs vs regularization strength")
        ax.legend(bbox_to_anchor=(1, 1))
        if return_intercept_coef_dataframe:
            return df

    def predict(self, x):
        raise NotImplementedError("prediction not yet implemented")