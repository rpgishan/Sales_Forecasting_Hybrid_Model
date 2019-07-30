# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
# import numpy as np
# from MyDataManupulator import MyOtherFunctions
from MyMongo import MyMongoDataFunctions
from pandas.plotting import autocorrelation_plot
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import MyConstants


class MyARIMA:

    def getARIMAModel(self, data, f_horizon):
        train_end_date = data

        # print("data")
        # print(data)
        # print()

        figure_size = (12, 9)
        # Differencing
        ts_data_diff = data - data.shift()
        # plt.figure(figsize=figuresize)
        # plt.plot(ts_data_diff, label='ts_data_diff')
        # plt.title('Differencing')
        # plt.show(block=False)

        # print("ts_data_diff:",ts_data_diff)

        ts_data_diff.dropna(inplace=True)

        # # Decomposing
        # decomposition = seasonal_decompose(data)
        #
        # trend = decomposition.trend
        # seasonal = decomposition.seasonal
        # residual = decomposition.resid
        #
        # trend.dropna(inplace=True)
        # seasonal.dropna(inplace=True)
        # residual.dropna(inplace=True)

        # ts_data = ts_data_diff

        ts_data_preprocessed = ts_data_diff

        # # ACF and PACF plots:
        # lag_acf = acf(ts_data_preprocessed, nlags=20)
        # lag_pacf = pacf(ts_data_preprocessed, nlags=20, method='ols')

        # # Plot ACF:
        # plt.figure(figsize=figure_size)
        # plt.subplot(211)
        # plt.plot(lag_acf)
        # plt.axhline(y=0, linestyle='--', color='gray')
        # plt.axhline(y=-1.96 / np.sqrt(len(ts_data_preprocessed)), linestyle='--', color='gray')
        # plt.axhline(y=1.96 / np.sqrt(len(ts_data_preprocessed)), linestyle='--', color='gray')
        # plt.title('Autocorrelation Function')
        #
        # # Plot PACF:
        # plt.subplot(212)
        # plt.plot(lag_pacf)
        # plt.axhline(y=0, linestyle='--', color='gray')
        # plt.axhline(y=-1.96 / np.sqrt(len(ts_data_preprocessed)), linestyle='--', color='gray')
        # plt.axhline(y=1.96 / np.sqrt(len(ts_data_preprocessed)), linestyle='--', color='gray')
        # plt.title('Partial Autocorrelation Function')
        # plt.tight_layout()
        #
        # plt.show(block=False)

        # ARIMA

        p_component = 1
        q_component = 1
        d_component = 1

        # plt.figure(figsize=figure_size)
        # # AR Model
        # model = ARIMA(data, order=(p_component, d_component, 0))
        # results_AR = model.fit(disp=-1)
        #
        # # plt.subplot(311)
        # # plt.plot(ts_data_preprocessed, label='ts_log_diff')
        # # plt.plot(results_AR.fittedvalues, color='red', label='results_AR')
        # # # plt.title('RSS: %.4f')  # % sum((results_AR.fittedvalues-ts_log_diff)**2))
        # # plt.title('AR Model')
        # # plt.legend(loc='best')
        # # # plt.show(block=False)
        #
        # # MA Model
        # model = ARIMA(data, order=(0, d_component, q_component))
        # results_MA = model.fit(disp=-1)
        #
        # # plt.subplot(312)
        # # plt.plot(ts_data_preprocessed, label='ts_log_diff')
        # # plt.plot(results_MA.fittedvalues, color='red', label='results_MA')
        # # # plt.title('RSS: %.4f')  # % sum((results_MA.fittedvalues-ts_log_diff)**2))
        # # plt.title('MA Model')
        # # plt.legend(loc='best')
        # # # plt.show(block=False)

        # Combined Model ARIMA
        model = ARIMA(data, order=(p_component, d_component, q_component))
        fit_results_ARIMA = model.fit(disp=-1)
        # predicted_results_ARIMA = model.predict()

        # plt.subplot(313)
        # plt.plot(ts_data_preprocessed, label='ts_log_diff')
        # plt.plot(results_ARIMA.fittedvalues, color='red', label='results_ARIMA')
        # # plt.title('RSS: %.4f')  # % sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
        # plt.title('Combined Model ARIMA')
        # plt.legend(loc='best')
        # plt.show(block=False)

        ts_data_2 = fit_results_ARIMA.fittedvalues

        # trained_prediction_series = pd.Series(data.values[0], index=data.index)
        # trained_prediction_series = trained_prediction_series.add(ts_data_2.cumsum(),fill_value=0)


        # plt.plot(data[value_col], color='blue')
        # plt.plot(trained_prediction_series, color='red')
        # plt.show()

        return ts_data_2


class MyARIMAMod:

    def getAutocorrelation(self, data):
        autocorrelation_plot(data)
        plt.show()

    def getPQ(self, data):
        autocorrelation_plot(data)
        plt.show()
        p = 1
        q = 1
        return p, q

    def getFittedARIMA(self, data):
        model = ARIMA(data, order=(10, 1, 0))
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        # plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        plt.show()
        residuals.plot(kind='kde')
        plt.show()
        print(residuals.describe())

    def getTestedARIMA(self, data):
        p, d, q = 15, 1, 5
        X = data.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
        # plot
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.show()

    def getPredictedARIMA(self, data, f_horizon):
        p, d, q = 1, 1, 1
        # print("ARIMAMod - getPredictedARIMA")
        prediction_start_date = data.index[len(data)-1] + timedelta(days=1)
        prediction_end_date = prediction_start_date + timedelta(days=f_horizon-1)
        predictions_series_index = pd.date_range(prediction_start_date,prediction_end_date)
        model = ARIMA(data, order=(p, d, q))
        model_fit = model.fit(disp=-1)
        output = model_fit.forecast(steps=f_horizon)
        predictions = output[0]
        # stderr = output[1]
        # conf_int = output[2]
        predictions_series = pd.DataFrame(predictions,predictions_series_index)
        model_fit_cumsum = model_fit.fittedvalues.cumsum()
        trained_prediction_series = pd.Series(data.values[0], index=data.index)
        trained_prediction_series = trained_prediction_series.add(model_fit_cumsum,fill_value=0)
        full_trained_prediction_series = trained_prediction_series.append(predictions_series)

        full_trained_prediction_diff_series = full_trained_prediction_series[0] - full_trained_prediction_series[0].shift()
        full_trained_prediction_diff_series.dropna(inplace=True)

        # print("getPredictedARIMA - predictions_series")
        # print(predictions_series)

        # print("getPredictedARIMA - full_trained_prediction_series")
        # print(full_trained_prediction_series)

        # plot
        # plt.plot(data[value_col], color='blue')
        # plt.plot(full_trained_prediction_series, color='red')
        # plt.show()
        return full_trained_prediction_diff_series


if __name__ == "__main__":
    database_name = MyConstants.database_name
    collection_name = MyConstants.collection_name
    value_col = "Mac"
    doc_id = "2"
    f_horizon = 3
    myARIMA = MyARIMAMod()
    myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)

    data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id, value_col, dbkeyname="sales_predicted")
    # myARIMA.getAutocorrelation(data)
    # myARIMA.getTestedARIMA(data)
    # MyARIMA().getARIMAModel(data,f_horizon)
    ARIMA_diff_series = myARIMA.getPredictedARIMA(data, f_horizon)
    print(ARIMA_diff_series)
