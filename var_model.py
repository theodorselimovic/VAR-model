import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse, aic
import statistics

data = pd.read_excel('./Final Data.xlsx')

control_variables = ["us_unemp", "us_stir", "us_qe", "euro_unemp", "euro_stir", "euro_qe", "vix"]
endogenous_variables = ["swe_unemp", "housing", "swe_qe"]
exchange_varibles = ["sek_euro", "sek_usd"]
index_exchange = 2
inflation_variables = ["cpif_census", "cpif_stl", "cpifxe"]
index_inflation = 3
energy_variables = ["energy", "oil"]
interest_rate_variables = ["swe_stir", "policy_rate"]
index_interest_rate = 6
lag_variables = [1, 2, 3, 4, 5, 6]
measurement_error = [True, False] 
z_statistics = []
filtered_p_values = []
aic_results = []
bic_results = []
f_stat = []
filtered_z_values = []
f_test_pvalues = []

final = {
    "z_statistics": filtered_z_values,
    "f_test_pvalues": f_test_pvalues,
    "f_stat" : f_stat,
    "aic_results": aic_results,
    "bic_results": bic_results
}

def statistics(k, model):
    z_statistics: pd.Series = model.tvalues
    p_values: pd.Series = model.pvalues
# Append the z-values and p-values to your lists
    z_statistics_no_sqrt = z_statistics[~z_statistics.index.str.contains('sqrt')]
    for z in [z_statistics_no_sqrt[idx] for idx in range (len(z_statistics_no_sqrt))]:
        filtered_z_values.append(abs(z))
    p_values_no_sqrt = p_values[~p_values.index.str.contains('sqrt')]
    for p in [p_values_no_sqrt[idx] for idx in range(len(p_values_no_sqrt))]:
        filtered_p_values.append(p)
    #adds aic and bic results 
    aic_results.append(abs(model.aic))
    bic_results.append(abs(model.bic))
    # defines the summary results
#saves and f-stat
    R = np.eye(k)
    fres = model.f_test(R)
    f_stat.append(abs(fres.fvalue))
    f_test_pvalues.append(fres.pvalue)
    

model_nbr = 1

#loops through all the different VARS.
# Changes measurement error
for measu in measurement_error:
    # Changes inflation variables, inserts the variable in the third position    
    for inf in inflation_variables:
        endogenous_variables.insert(index_inflation, inf)
        # Same for interest rate variables, but 6th position
        for inte in interest_rate_variables:
            endogenous_variables.insert(index_interest_rate, inte)
            # sets two possibilities, one where exchange rates are endogenous, and one where they are exogenous
            # This part, of how to put in the variables, still needs to be done. 
            # right now, it just inserts the exchange variable into the endogenous.
            for j in range(2):
                if j == 0:
                    for exch in exchange_varibles:
                    #endegenous exchange rates
                        endogenous_variables.insert(index_exchange, exch)
                        # Changes lag order
                        for lag in lag_variables:
                            # control variables
                            for i in range(2):
                                if i == 0: 
                                    if (inf == "cpifxe"):
                                        for energ in energy_variables:
                                        # add energy variable
                                            print(f"Training model number: {model_nbr}")
                                            control_variables.append(energ)
                                            model = sm.tsa.VARMAX(
                                                data[endogenous_variables],
                                                exog = data[control_variables], 
                                                order = (lag, 0), trend='c', 
                                                error_cov_type='diagonal', 
                                                measurement_error = measu, 
                                                enforce_stationarity=True, 
                                                enforce_invertibility=True
                                                ).fit(
                                                    maxiter=1000, 
                                                    disp=False
                                                )
                                            k = len(model.params)
                                            statistics(k, model)
                                            model_nbr += 1
                                    else:
                                        print(f"Training model number: {model_nbr}")
                                        model = sm.tsa.VARMAX(
                                            data[endogenous_variables], 
                                            exog = data[control_variables], 
                                            order = (lag, 0), 
                                            trend='c', 
                                            error_cov_type='diagonal', 
                                            measurement_error = measu, 
                                            enforce_stationarity=True, 
                                            enforce_invertibility=True
                                            ).fit(
                                                maxiter=1000,
                                                disp=False
                                            )
                                        k = len(model.params)
                                        statistics(k, model)
                                        model_nbr += 1
                                # no control variables
                                else:
                                    # if cpifxe, then within the model energ
                                    if inf == "cpifxe":
                                        for energ in energy_variables:
                                            print(f"Training model number: {model_nbr}")
                                            model = sm.tsa.VARMAX(
                                                data[endogenous_variables], 
                                                exog = data[energ], 
                                                order = (lag, 0), 
                                                trend='c', 
                                                error_cov_type='diagonal', 
                                                measurement_error = measu, 
                                                enforce_stationarity=True, 
                                                enforce_invertibility=True
                                                ).fit(
                                                    maxiter=1000,
                                                    disp = False
                                                )
                                            k = len(model.params)
                                            statistics(k, model)
                                            model_nbr += 1
                                            control_variables.remove(energ)
                                    # if not, run with no control variables
                                    else:
                                        print(f"Training model number: {model_nbr}")
                                        model = sm.tsa.VARMAX(
                                            data[endogenous_variables], 
                                            order = (lag, 0), 
                                            trend='c', 
                                            error_cov_type='diagonal', 
                                            measurement_error = measu, 
                                            enforce_stationarity=True, 
                                            enforce_invertibility=True
                                            ).fit(
                                                maxiter=1000,
                                                disp=False
                                            )
                                        k = len(model.params)
                                        statistics(k, model)
                                        model_nbr += 1
                        endogenous_variables.remove(exch)      
                if j == 1:
                    for exch in exchange_varibles:
                    #endegenous exchange rates
                        control_variables.insert(index_exchange, exch)
                        # Changes lag order
                        for lag in lag_variables:
                            # control variables
                            for i in range(2):
                                if i == 0: 
                                    if (inf == "cpifxe"):
                                        for energ in energy_variables:
                                        # add energy variable
                                            control_variables.append(energ)
                                            print(f"Training model number: {model_nbr}")
                                            model = sm.tsa.VARMAX(
                                                data[endogenous_variables],
                                                exog = data[control_variables], 
                                                order = (lag, 0), trend='c', 
                                                error_cov_type='diagonal', 
                                                measurement_error = measu, 
                                                enforce_stationarity=True, enforce_invertibility=True
                                                ).fit(
                                                    maxiter=1000,
                                                    disp=False
                                                )
                                            k = len(model.params)
                                            statistics(k, model)
                                            model_nbr += 1
                                    else:
                                        print(f"Training model number: {model_nbr}")
                                        model = sm.tsa.VARMAX(
                                            data[endogenous_variables], 
                                            exog = data[control_variables], 
                                            order = (lag, 0), 
                                            trend='c', 
                                            error_cov_type='diagonal', 
                                            measurement_error = measu, 
                                            enforce_stationarity=True, 
                                            enforce_invertibility=True
                                            ).fit(
                                                maxiter=1000,
                                                dips=False
                                            )
                                        k = len(model.params)
                                        statistics(k, model)
                                        model_nbr += 1
                                # no control variables
                                else:
                                    # if cpifxe, then within the model energ
                                    if inf == "cpifxe":
                                        for energ in energy_variables:
                                            print(f"Training model number: {model_nbr}")
                                            model = sm.tsa.VARMAX(
                                                data[endogenous_variables], 
                                                exog = data[energ], 
                                                order = (lag, 0), 
                                                trend='c', 
                                                error_cov_type='diagonal', 
                                                measurement_error = measu, 
                                                enforce_stationarity=True, 
                                                enforce_invertibility=True
                                                ).fit(
                                                    maxiter=1000,
                                                    disp=False
                                                )
                                            k = len(model.params)
                                            statistics(k, model)
                                            model_nbr += 1
                                            control_variables.remove(energ)
                                    # if not, run with no control variables
                                    else:
                                        print(f"Training model number: {model_nbr}")
                                        model = sm.tsa.VARMAX(
                                            data[endogenous_variables], 
                                            order = (lag, 0), 
                                            trend='c', 
                                            error_cov_type='diagonal', 
                                            measurement_error = measu, 
                                            enforce_stationarity=True, 
                                            enforce_invertibility=True
                                            ).fit(
                                                maxiter=1000,
                                                disp=False
                                            )
                                        k = len(model.params)
                                        statistics(k, model)
                                        model_nbr += 1
                        control_variables.remove(exch)              
            endogenous_variables.remove(inte)
        endogenous_variables.remove(inf)

#calculations of descriptive statistics - gives mean, stdev, min, max, and quartiles.
for key, values in final.items():
    # Convert the list to a pandas Series
    values = pd.Series(values)
    # Use the describe function
    final[key] = values.describe()
    print(f"\nkey: {key}, value: {final[key]}")

p_values_less_than_0_1 = list(filter(lambda x : x < 0.1, filtered_p_values))

print(f"share of p-values less than 0.1: {len(p_values_less_than_0_1) / len(filtered_p_values)}")
