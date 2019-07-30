import numpy as np
import pandas as pd
import io
import requests
# from MyMongo import MyMongoBasicFunctions
import os
from datetime import datetime,timedelta
import math
import json
import MyConstants
from flask import jsonify
# from MyARIMA import MyARIMA
from MyMongo import MyMongoDataFunctions

VALID_TIME_RESOLUTION_STATUS = {"quarterly", "monthly", "daily"}


# used % to divide the dataset
class MyRNNDataManipulatorModified:

    def __init__(self, trainingpercentage=80):
        self.trainingpercentage = trainingpercentage

    # def __init__(self,trainingpercentage = 80,testingpercentage = 20):
    #     self.trainingpercentage = trainingpercentage
    #     self.testingpercentage = testingpercentage

    def getDifferentialDataset(self, data):

        ts_data_diff = data - data.shift()
        ts_data_diff.dropna(inplace=True)
        ts_data = ts_data_diff.copy()
        # ts_data = data.copy()
        return ts_data

    ## only works for Mac sales dataset
    def getDifferentialDatasetWithComments(self, data, f_horizon, random_period, value_col):

        ts_data_diff = data - data.shift()
        ts_data_diff.dropna(inplace=True)
        ts_data = ts_data_diff.copy()
        # ts_data = data.copy()
        data2 = data.copy()
        data2['shift'] = data2.shift(-f_horizon)[value_col]
        data2['Diff'] = data2['shift'] - data2[value_col]
        data2['Diff'] = data2['Diff'].fillna(np.random.randint(data2['Diff'].min(),data2['Diff'].max()))
        data2['Random'] = pd.Series(np.random.randint(1, random_period, size=len(data))).values
        # data2['Random_Pos'] = pd.Series(np.random.randint(0, random_period, size=len(data))).values
        # data2['Random_Neg'] = pd.Series(np.random.randint(0, random_period, size=len(data))).values
        ts_data['Positive'] = abs(data2['Diff'] + data2['Random']).astype(int)
        ts_data['Negative'] = abs(data2['Diff'] - data2['Random']).astype(int)
        # ts_data['Diff'] = data2['Diff']
        return ts_data

    def getOutputRNNTrainDataset(self, data, f_horizon, num_periods):
        TS_Output = np.array(data.values)
        TS_Output_Index = np.array(data.index)
        if f_horizon > num_periods:
            f_horizon = num_periods
        batch_end = math.floor(len(TS_Output) * self.trainingpercentage / 100) - f_horizon
        end_of_dataset = batch_end - (batch_end % num_periods)

        y_train_data = TS_Output[f_horizon:end_of_dataset + f_horizon]
        y_train_data_index = TS_Output_Index[f_horizon:end_of_dataset + f_horizon]
        y_train_batches = y_train_data.reshape(-1, num_periods, 1)
        y_train_data_series = pd.Series(np.ravel(y_train_data), y_train_data_index)
        return y_train_data, y_train_data_index, y_train_batches, y_train_data_series

    # Only works for past sales dataset type of inputs
    def getInputRNNTrainDatasetWithOneInput(self, data, f_horizon, num_periods):
        TS_Input = np.array(data.values)
        if f_horizon > num_periods:
            f_horizon = num_periods
        batch_end = math.floor(len(TS_Input) * self.trainingpercentage / 100) - f_horizon
        end_of_dataset = batch_end - (batch_end % num_periods)
        x_train_data = TS_Input[:end_of_dataset]
        x_train_batches = x_train_data.reshape(-1, num_periods, 1)
        return x_train_data, x_train_batches

    # Only works for past sales dataset, predicted ARIMA dataset type of inputs
    def getInputRNNTrainDatasetWithTwoInputs(self, data_1, data_2, f_horizon, num_periods):

        # TS_Input_1 = np.array(data_1.iloc[:, 0].values)
        # TS_Input_1_Index = np.array(data_1.iloc[:, 0].index)
        TS_Input_1 = np.array(data_1.values)
        # TS_Input_1_Index = np.array(data_1.index)

        TS_Input_2 = np.array(data_2.values)
        # TS_Input_2_Index = np.array(data_2.index)

        if f_horizon > num_periods:
            f_horizon = num_periods

        batch_end = math.floor(len(TS_Input_1) * self.trainingpercentage / 100) - f_horizon
        end_of_dataset = batch_end - (batch_end % num_periods)

        x_train_data_1 = TS_Input_1[:end_of_dataset]
        # x_train_data_1_index = TS_Input_1_Index[:bend_of_dataset]

        x_train_data_2 = TS_Input_2[f_horizon:end_of_dataset + f_horizon]
        # x_train_data_2_index = TS_Input_2_Index[f_horizon:end_of_dataset + f_horizon]

        x_train_data = []

        for i in range(len(x_train_data_1)):
            x_train_data = x_train_data + [x_train_data_1[i], x_train_data_2[i]]

        x_train_data = np.array(x_train_data)

        # print("x_data:",len(x_data))
        # print(x_data)

        x_train_batches = x_train_data.reshape(-1, num_periods, 2)

        return x_train_data, x_train_batches

    # Only works for past sales dataset, positive comments and negative comments type of inputs
    def getInputRNNTrainDatasetWithThreeInputs(self, data_1, data_2, data_3, f_horizon, num_periods):

        TS_Input_1 = np.array(data_1.values)
        TS_Input_2 = np.array(data_2.values)
        TS_Input_3 = np.array(data_3.values)

        if f_horizon > num_periods:
            f_horizon = num_periods

        batch_end = math.floor(len(TS_Input_1) * self.trainingpercentage / 100) - f_horizon
        end_of_dataset = batch_end - (batch_end % num_periods)

        x_train_data_1 = TS_Input_1[:end_of_dataset]
        x_train_data_2 = TS_Input_2[:end_of_dataset]
        x_train_data_3 = TS_Input_3[:end_of_dataset]

        x_train_data = []

        for i in range(len(x_train_data_1)):
            x_train_data = x_train_data + [x_train_data_1[i], x_train_data_2[i], x_train_data_3[i]]

        x_train_data = np.array(x_train_data)

        # print("x_data:",len(x_data))
        # print(x_data)

        x_train_batches = x_train_data.reshape(-1, num_periods, 3)

        return x_train_data, x_train_batches

    # Only works for past sales dataset, positive comments and negative comments, predicted ARIMA dataset type of inputs
    def getInputRNNTrainDatasetWithFourInputs(self, data_1, data_2, data_3, data_4, f_horizon, num_periods):
        TS_Input_1 = np.array(data_1.values)
        TS_Input_2 = np.array(data_2.values)
        TS_Input_3 = np.array(data_3.values)
        TS_Input_4 = np.array(data_4.values)
        if f_horizon > num_periods:
            f_horizon = num_periods
        batch_end = math.floor(len(TS_Input_1) * self.trainingpercentage / 100) - f_horizon
        end_of_dataset = batch_end - (batch_end % num_periods)
        x_train_data_1 = TS_Input_1[:end_of_dataset]
        x_train_data_2 = TS_Input_2[:end_of_dataset]
        x_train_data_3 = TS_Input_3[:end_of_dataset]
        x_train_data_4 = TS_Input_4[f_horizon:end_of_dataset + f_horizon]
        x_train_data = []
        for i in range(len(x_train_data_1)):
            x_train_data = x_train_data + [x_train_data_1[i], x_train_data_2[i], x_train_data_3[i], x_train_data_4[i]]
        x_train_data = np.array(x_train_data)
        x_train_batches = x_train_data.reshape(-1, num_periods, 4)
        return x_train_data, x_train_batches

    def getOutputRNNTestDataset(self, data, f_horizon, num_periods):

        TS_Output = np.array(data.values)
        TS_Output_Index = np.array(data.index)

        test_starts = math.ceil(len(TS_Output) * self.trainingpercentage / 100)
        # test_ends = len(TS_Output) - test_starts - f_horizon - ((len(TS_Output) - test_starts - f_horizon) % num_periods)
        test_ends = test_starts + (len(TS_Output) - test_starts - f_horizon - (
                    (len(TS_Output) - test_starts - f_horizon) % num_periods))

        # if test_ends-num_periods <= test_starts:
        #     raise ValueError("Dataset is not enough")

        # print("output test_starts - ",test_starts," \t\t test_ends - ",test_ends)

        y_test_data = TS_Output[(test_starts + f_horizon):(test_ends + f_horizon)]

        y_test_batches = y_test_data.reshape(-1, num_periods, 1)

        y_test_data_index = TS_Output_Index[(test_starts + f_horizon):(test_ends + f_horizon)]

        y_test_data_series = pd.Series(np.ravel(y_test_data), y_test_data_index)

        return y_test_batches, y_test_data_index, y_test_data_series, test_starts, test_ends

    # Only works for past sales dataset type of inputs
    def getInputRNNTestDatasetWithOneInput(self, data, f_horizon, num_periods):

        TS_Input = np.array(data.values)
        # TS_Input_Index = np.array(data.index)

        test_starts = math.ceil(len(TS_Input) * self.trainingpercentage / 100)
        # test_ends = len(TS_Input) - test_starts - f_horizon - ((len(TS_Input) - test_starts - f_horizon) % num_periods)
        test_ends = test_starts + (
                    len(TS_Input) - test_starts - f_horizon - ((len(TS_Input) - test_starts - f_horizon) % num_periods))

        # print("one input test_starts - ",test_starts," \t\t test_ends - ",test_ends)

        # if test_ends-num_periods <= test_starts:
        #     raise ValueError("Dataset is not enough")

        x_test_data = TS_Input[test_starts:test_ends]
        x_test_batches = x_test_data.reshape(-1, num_periods, 1)
        # x_test_data_index = test_x_index_setup[:(num_periods * 1)]

        return x_test_batches

    # Only works for past sales dataset, positive comments and negetive comments, predicted ARIMA dataset type of inputs
    def getInputRNNTestDatasetWithFourInputs(self, data_1, data_2, data_3, data_4, f_horizon, num_periods):

        TS_Input_1 = np.array(data_1.values)
        TS_Input_2 = np.array(data_2.values)
        TS_Input_3 = np.array(data_3.values)
        TS_Input_4 = np.array(data_4.values)

        test_starts = math.ceil(len(TS_Input_1) * self.trainingpercentage / 100)
        # test_ends = len(TS_Input_1) - test_starts - f_horizon - ((len(TS_Input_1) - test_starts - f_horizon) % num_periods)
        test_ends = test_starts + (len(TS_Input_1) - test_starts - f_horizon - (
                    (len(TS_Input_1) - test_starts - f_horizon) % num_periods))

        # print("four inputs test_starts - ",test_starts," \t\t test_ends - ",test_ends)

        # if test_ends-num_periods <= test_starts:
        #     raise ValueError("Dataset is not enough")

        x_test_data_1 = TS_Input_1[test_starts:test_ends]
        x_test_data_2 = TS_Input_2[test_starts:test_ends]
        x_test_data_3 = TS_Input_3[test_starts:test_ends]
        x_test_data_4 = TS_Input_4[test_starts + f_horizon:test_ends + f_horizon]

        x_test_data = []

        for i in range(len(x_test_data_1)):
            x_test_data = x_test_data + [x_test_data_1[i], x_test_data_2[i], x_test_data_3[i], x_test_data_4[i]]

        x_test_data = np.array(x_test_data)
        # print("\nx_test_data")
        # print(x_test_data)
        # print()
        x_test_batches = x_test_data.reshape(-1, num_periods, 4)

        return x_test_batches

    # Validation dataset
    def getOutputRNNValidationDataset(self, data, num_periods):

        TS_Output = np.array(data.values)
        TS_Output_Index = np.array(data.index)

        if len(TS_Output) < num_periods:
            raise ValueError("Verification dataset is not enough")

        validate_starts = len(TS_Output) - num_periods
        validate_ends = len(TS_Output)

        y_validate_data = TS_Output[validate_starts:validate_ends]

        y_validate_batches = y_validate_data.reshape(-1, num_periods, 1)

        y_validate_data_index = TS_Output_Index[validate_starts:validate_ends]

        y_validate_data_series = pd.Series(np.ravel(y_validate_data), y_validate_data_index)

        return y_validate_batches, y_validate_data_index, y_validate_data_series, validate_starts, validate_ends

    # have to modify this def again
    def getInputRNNValidationDatasetWithOneInputs(self, data_1, f_horizon, num_periods):

        TS_Input_1 = np.array(data_1.values)

        if len(TS_Input_1) < (num_periods + f_horizon):
            raise ValueError("Verification dataset is not enough")

        validate_starts = len(TS_Input_1) - num_periods - f_horizon
        validate_ends = len(TS_Input_1) - f_horizon

        x_validate_data = TS_Input_1[validate_starts:validate_ends]

        # print("\nx_test_data")
        # print(x_test_data)
        # print()
        x_validate_batches = x_validate_data.reshape(-1, num_periods, 1)
        return x_validate_data, x_validate_batches

    # have to modify this def again
    def getInputRNNValidationDatasetWithFourInputs(self, data_1, data_2, data_3, data_4, f_horizon, num_periods):

        TS_Input_1 = np.array(data_1.values)
        TS_Input_2 = np.array(data_2.values)
        TS_Input_3 = np.array(data_3.values)
        TS_Input_4 = np.array(data_4.values)

        if len(TS_Input_1) < (num_periods + f_horizon):
            raise ValueError("Verification dataset is not enough")

        validate_starts = len(TS_Input_1) - num_periods - f_horizon
        validate_ends = len(TS_Input_1) - f_horizon

        x_validate_data_1 = TS_Input_1[validate_starts:validate_ends]
        x_validate_data_2 = TS_Input_2[validate_starts:validate_ends]
        x_validate_data_3 = TS_Input_3[validate_starts:validate_ends]
        x_validate_data_4 = TS_Input_4[validate_starts + f_horizon:validate_ends + f_horizon]

        x_validate_data = []

        for i in range(len(x_validate_data_1)):
            x_validate_data = x_validate_data + [x_validate_data_1[i], x_validate_data_2[i], x_validate_data_3[i],
                                             x_validate_data_4[i]]

        x_validate_data = np.array(x_validate_data)
        # print("\nx_test_data")
        # print(x_test_data)
        # print()
        x_validate_batches = x_validate_data.reshape(-1, num_periods, 4)
        return x_validate_data, x_validate_batches

    # have to modify this def again
    def getInputRNNPredictionDatasetWithFourInputs(self, data_1, data_2, data_3, data_4, f_horizon, num_periods):

        TS_Input_1 = np.array(data_1.values)
        TS_Input_2 = np.array(data_2.values)
        TS_Input_3 = np.array(data_3.values)
        TS_Input_4 = np.array(data_4.values)

        if len(TS_Input_1) < (num_periods):
            raise ValueError("Prediction dataset is not enough")

        predict_starts = len(TS_Input_1) - num_periods
        predict_ends = len(TS_Input_1)

        x_validate_data_1 = TS_Input_1[predict_starts:predict_ends]
        x_validate_data_2 = TS_Input_2[predict_starts:predict_ends]
        x_validate_data_3 = TS_Input_3[predict_starts:predict_ends]
        x_validate_data_4 = TS_Input_4[predict_starts + f_horizon:predict_ends + f_horizon]

        x_validate_data = []

        for i in range(len(x_validate_data_1)):
            x_validate_data = x_validate_data + [x_validate_data_1[i], x_validate_data_2[i], x_validate_data_3[i],
                                             x_validate_data_4[i]]

        x_validate_data = np.array(x_validate_data)
        # print("\nx_test_data")
        # print(x_test_data)
        # print()
        x_validate_batches = x_validate_data.reshape(-1, num_periods, 4)
        return x_validate_data, x_validate_batches, predict_starts, predict_ends


    def getNonDifferentiatedSeries(self,raw_data_series,differentiated_data_series):
        diff_start = differentiated_data_series.index[0]#.to_pydatetime()
        days = 1
        while True:
            first_date = diff_start - timedelta(days=days)
            try:
                first_date_value = raw_data_series[first_date]
                break
            except:
                days += 1
        differentiated_data_series_cumsum = differentiated_data_series.cumsum()
        result_non_differentiated_data_series = differentiated_data_series_cumsum + first_date_value

        return result_non_differentiated_data_series



class MyCSVHandler:

    def getDatasetQuarter(self, source_path, index_col_name, value_col_name):
        data1 = pd.read_csv(source_path)

        csv_col_headers = list(data1)
        print(csv_col_headers)
        if index_col_name not in csv_col_headers:
            raise ValueError("Index column name must be one of %r." % csv_col_headers)
        if value_col_name not in csv_col_headers:
            raise ValueError("Value column name must be one of %r." % csv_col_headers)

        data = pd.read_csv(source_path, index_col=[index_col_name])

        data1[['q', 'year']] = data1[index_col_name].str.split('/', expand=True)

        data1[['', 'q']] = data1['q'].str.split('Q', expand=True)
        data1.q = (((data1.q.astype(int) - 1) * 3) + 1)
        data1.q = data1.q.astype(str)

        data1[index_col_name] = data1.year + '-' + data1.q
        data1[index_col_name] = pd.to_datetime(data1[index_col_name])

        data.index = data1[index_col_name]
        data[value_col_name] = data[value_col_name] * 1000  # 000000

        return data

    def getDatasetQuarterFromUrl(self, source_url, index_col_name, value_col_name):
        source = requests.get(source_url).content
        source_path1 = io.StringIO(source.decode('utf-8'))
        source_path2 = io.StringIO(source.decode('utf-8'))

        data1 = pd.read_csv(source_path2)

        csv_col_headers = list(data1)

        if index_col_name not in csv_col_headers:
            raise ValueError("Index column name must be one of %r." % csv_col_headers)
        if value_col_name not in csv_col_headers:
            raise ValueError("Value column name must be one of %r." % csv_col_headers)

        data = pd.read_csv(source_path1, index_col=[index_col_name])

        data1[['q', 'year']] = data1[index_col_name].str.split('/', expand=True)

        data1[['', 'q']] = data1['q'].str.split('Q', expand=True)
        data1.q = (((data1.q.astype(int) - 1) * 3) + 1)
        data1.q = data1.q.astype(str)

        data1[index_col_name] = data1.year + '-' + data1.q
        data1[index_col_name] = pd.to_datetime(data1[index_col_name])

        data.index = data1[index_col_name]
        data[value_col_name] = data[value_col_name] * 1000  # 000000

        return data

    def getDatasetMonthly(self, source_path, index_col_name):
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
        data = pd.read_csv(source_path, parse_dates=[index_col_name], index_col=[index_col_name], date_parser=dateparse)

        return data

    def getDatasetDaily(self, source_path, index_col_name):
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
        # dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
        data = pd.read_csv(source_path, parse_dates=[index_col_name], index_col=[index_col_name], date_parser=dateparse)

        return data

    def getOnlineCSVSourcePath(self, csv_url):
        url = csv_url
        source = requests.get(url).content
        sourcePath = io.StringIO(source.decode('utf-8'))
        # sourcePath = source.decode('utf-8')
        print(sourcePath)
        return sourcePath

    def writeToACSV(self,source_folder,file_name,data_series):
        if not os.path.exists(source_folder):
            os.makedirs(source_folder)
        if data_series.index.name is None:
            data_series.index.name = "Index"
        source_path = source_folder + "/" + file_name + ".csv"
        data_series.to_csv(source_path)


class MyOtherFunctions:
    def getDatasetARFromPath(self, source_path, index_col, value_col, time_resolution, turn_on_data_filler=False):
        if time_resolution not in VALID_TIME_RESOLUTION_STATUS:
            raise ValueError("Time Resolution must be one of %r." % VALID_TIME_RESOLUTION_STATUS)

        myCSVHandler = MyCSVHandler()
        myDatasetGenerator = MyDatasetGenerator()
        if time_resolution == 'quarterly':
            data = myCSVHandler.getDatasetQuarter(source_path, index_col, value_col)
            # Data Filler
            if turn_on_data_filler:
                data = myDatasetGenerator.dataFiller(data)
        elif time_resolution == 'monthly':
            data = myCSVHandler.getDatasetMonthly(source_path, index_col)
        elif time_resolution == 'daily':
            data = myCSVHandler.getDatasetDaily(source_path, index_col)
        return data

    def getDatasetAR(self, source_url, index_col, value_col, time_resolution, turn_on_data_filler=False):
        if time_resolution not in VALID_TIME_RESOLUTION_STATUS:
            raise ValueError("Time Resolution must be one of %r." % VALID_TIME_RESOLUTION_STATUS)

        myCSVHandler = MyCSVHandler()
        myDatasetGenerator = MyDatasetGenerator()
        if time_resolution == 'quarterly':
            data = myCSVHandler.getDatasetQuarterFromUrl(source_url, index_col, value_col)
            # Data Filler
            if turn_on_data_filler:
                data = myDatasetGenerator.dataFiller(data)
        elif time_resolution == 'monthly':
            source_path = myCSVHandler.getOnlineCSVSourcePath(source_url)
            data = myCSVHandler.getDatasetMonthly(source_path, index_col)
        return data

    def getNormalizedSeries(self, data_series):
        data_series_norm = (data_series - data_series.min()) / (data_series.max() - data_series.min())
        return data_series_norm

    def getIntersectionPoints(self, dataAR1, dataAR2):
        idx = np.argwhere(np.diff(np.sign(dataAR1 - dataAR2)) != 0).reshape(-1) + 0
        return idx

    def getMeanPercentageError(self,actual_data_series,predicted_data_series):
        mpe_data_series = (actual_data_series - predicted_data_series)/actual_data_series*100.0
        return mpe_data_series

    def getMeanAbsolutePercentageError(self,actual_data_series,predicted_data_series):
        actual_data_series_temp = actual_data_series.copy()
        predicted_data_series_temp = predicted_data_series.copy()
        # min = actual_data_series_temp.min()
        # min_pred = predicted_data_series_temp.min()
        # if min > min_pred:
        #     min = min_pred
        # min -= 1
        # actual_data_series_temp -= min
        # predicted_data_series_temp -= min
        mape_data_series = (abs(actual_data_series_temp - predicted_data_series_temp)/actual_data_series_temp)*100.0
        average_mape_data_series=mape_data_series.mean()
        return mape_data_series,average_mape_data_series


class MyDatasetGenerator:
    # for quarters
    def dataFiller(self, data):
        relData1 = data[0:data.size - 1]
        relData2 = data[1:]
        dataindexdiff = relData2.index - relData1.index

        indexnum = 0
        indexar = np.array([])
        valuear = np.array([])

        for daysdiff in dataindexdiff:
            startVal = data.values[indexnum]
            endVal = data.values[indexnum + 1]
            tempindexar = [data.index[indexnum]]
            randomvalar = np.random.uniform(startVal, endVal, daysdiff.days - 1)
            valuear = np.append(valuear, data.values[indexnum])
            valuear = np.append(valuear, randomvalar)
            for middledays in range(1, daysdiff.days):
                newmiddledate = data.index[indexnum] + timedelta(days=middledays)
                tempindexar.append(newmiddledate)

            indexar = np.append(indexar, tempindexar)
            indexnum += 1

        indexar = np.append(indexar, data.index[indexnum])
        valuear = np.append(valuear, data.values[indexnum])

        dataframe = pd.DataFrame(valuear, index=indexar, columns=['Mac'])
        dataframe.index.name = data.index.name

        return dataframe


class MyJsonGenerator:
    def getJsonFromDataSeries(self, name_of_the_data_series, data_series):
        json_result_ar = []

        for index in data_series.index:
            # print(index," - ",data_series[index])
            # print(type(index.to_pydatetime()))
            datetime_index = index.to_pydatetime()
            new_index = int((datetime_index - datetime.utcfromtimestamp(0)).total_seconds())
            json_obj = {"index": new_index, "value": float(data_series[index])}
            json_result_ar.append(json_obj)

        json_result = json.dumps({name_of_the_data_series: json_result_ar})
        return json_result

    def getJsonArFromDataSeries(self, data_series):
        json_result_ar = list()

        for index in data_series.index:
            # print(index," - ",data_series[index])
            # print(type(index.to_pydatetime()))
            new_index = int((index.to_pydatetime() - datetime.utcfromtimestamp(0)).total_seconds())
            json_obj = {"name": new_index, "value": float(data_series[index])}
            json_result_ar.append(json_obj)

        return json_result_ar


    def getJsonArFromDataSeries(self, data_series):
        json_result_ar = []

        for index in data_series.index:
            # print(index," - ",data_series[index])
            # print(type(index.to_pydatetime()))
            new_index = int((index.to_pydatetime() - datetime.utcfromtimestamp(0)).total_seconds())
            json_obj = json.dumps({"name": new_index, "value": float(data_series[index])})
            json_result_ar.append(json_obj)

        return json_result_ar

    # multi = [
    #     {
    #         "name": "Germany",
    #         "series": [
    #             {
    #                 "name": "2010",
    #                 "value": 7300000
    #             },
    #             {
    #                 "name": "2011",
    #                 "value": 8940000
    #             }
    #         ]
    #     },
    #
    #     {
    #         "name": "USA",
    #         "series": [
    #             {
    #                 "name": "2010",
    #                 "value": 7870000
    #             },
    #             {
    #                 "name": "2011",
    #                 "value": 8270000
    #             }
    #         ]
    #     },
    #
    #     {
    #         "name": "France",
    #         "series": [
    #             {
    #                 "name": "2010",
    #                 "value": 5000002
    #             },
    #             {
    #                 "name": "2011",
    #                 "value": 5800000
    #             }
    #         ]
    #     }
    # ]




# if __name__=="__main__":
#
#     database_name = MyConstants.database_name
#     collection_name = MyConstants.collection_name
#     value_col = MyConstants.value_col
#     doc_id = "300241"
#
#     f_horizon = MyConstants.f_horizon
#     random_period = MyConstants.random_period
#
#     myRNNDataManipulator = MyRNNDataManipulatorModified()
#     myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)
#
#     data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id, value_col, dbkeyname="sales_predicted")
#     # myARIMA = MyARIMA()
#     # data_ARIMA = myARIMA.getARIMAModel(data)
#
#     # rnn_data_with_comments = myRNNDataManipulator.getDifferentialDatasetWithComments(data, f_horizon, random_period,
#     #                                                                                  value_col)
#     #
#     # rnn_data_with_comments["ARIMA"] = data_ARIMA
#     #
#     # with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
#     #     print(rnn_data_with_comments)
#
#     # myJsonGenerator = MyJsonGenerator()
#     # result_json = myJsonGenerator.getJsonArFromDataSeries(data[value_col])
#
#     # print(result_json)
#
#     ts_data_res = myRNNDataManipulator.getDifferentialDatasetWithComments(data, f_horizon, random_period, value_col)
#     # print(ts_data_res)
#
#     result = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], ts_data_res[value_col])
#     print(result)




# if __name__ == "__main__":
#     # source_url = "https://gishtestings.azurewebsites.net/uploads/apple--mac-revenue-in-billions.csv"
#     # myCSVHandler = MyCSVHandler()
#     # index_col = "Index"
#     value_col = "Mac"
#     doc_id = "11"
#     database_name = MyConstants.database_name #"Invictus_Reviews_Analysis"
#     collection_name = MyConstants.collection_name
#     myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)
#     data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id,value_col, dbkeyname="sales_predicted")
#     myDatasetGenerator = MyDatasetGenerator()
#     data = myDatasetGenerator.dataFiller(data)
#     data.to_csv("Datasets/MacBookPredict.csv")
#     # print(data)


# if __name__ == "__main__old":
#     print("DataManupulator")
#     source_url = "https://gishtestings.azurewebsites.net/uploads/apple--mac-revenue-in-billions.csv"
#     index_col = "Category"
#     value_col = "Mac"
#     time_resolution = "quarterly"
#     f_horizon = 5
#     num_periods = 10
#     random_period = 100
#     training_percentage = 80
#
#     myCSVHandler = MyCSVHandler()
#     # myRNNDataManipulator = MyRNNDataManipulator()
#     myRNNDataManipulatorModified = MyRNNDataManipulatorModified(trainingpercentage=training_percentage)
#     myDatasetGenerator = MyDatasetGenerator()
#     data = myCSVHandler.getDatasetQuarterFromUrl(source_url, index_col, value_col)
#
#     # data = data[0:57]
#     # data = myDatasetGenerator.dataFiller(data)
#
#     # myARIMA = MyARIMA()
#     data_ARIMA = []#myARIMA.getARIMAModel(data)
#     # print()
#     # print("data_ARIMA")
#     # print(data_ARIMA)
#     # print(len(data_ARIMA))
#     # print()
#
#     # # data = myRNNDataManipulator.getDifferentialDatasetFromMongoDBWithComments(data,time_resolution)
#     # # data = myRNNDataManipulatorModified.getDifferentialDatasetWithComments(data, 1, 100)
#     print()
#     print("data")
#     # print(data)
#     print(len(data))
#     print()
#     # norm_data_series = MySeriesNormalizer().getNormalizedSeries(data)
#     #
#     # print("norm_data_series")
#     # print(norm_data_series)
#
#     rnn_data = myRNNDataManipulatorModified.getDifferentialDatasetWithComments(data, f_horizon, random_period)
#     # rnn_data = rnn_data[:len(rnn_data)-f_horizon+1]
#     print()
#     print("rnn_data")
#     print(rnn_data)
#     print(len(rnn_data))
#     print()
#     print()
#
#     if ((num_periods + f_horizon) * 100 / (100 - training_percentage)) > len(rnn_data):
#         raise ValueError("Dataset is not enough")
#
#     data_1 = rnn_data.iloc[:, 0]
#     data_2 = rnn_data.iloc[:, 1]
#     data_3 = rnn_data.iloc[:, 2]
#
#     x_train_data, x_train_batches = myRNNDataManipulatorModified.getInputRNNTrainDatasetWithFourInputs(data_1, data_2,
#                                                                                                        data_3,
#                                                                                                        data_ARIMA,
#                                                                                                        f_horizon,
#                                                                                                        num_periods)
#
#     print()
#     print("x_train_data")
#     # print(x_train_data)
#     print(len(x_train_data))
#     print()
#     print()
#     print("x_train_batches")
#     print(x_train_batches)
#     print(len(x_train_batches))
#     print()
#     print()
#
#     y_train_data, y_train_data_index, y_train_batches, y_train_data_series = myRNNDataManipulatorModified.getOutputTrainRNNDataset(
#         data_1, f_horizon, num_periods)
#
#     print()
#     print("y_train_data")
#     # print(y_train_data)
#     print(len(y_train_data))
#     print()
#     print()
#     print("y_train_batches")
#     print(y_train_batches)
#     print(len(y_train_batches))
#     print()
#     print()
#     print()
#
#     x_test_batches = myRNNDataManipulatorModified.getInputRNNTestDatasetWithFourInputs(data_1, data_2, data_3,
#                                                                                        data_ARIMA, f_horizon,
#                                                                                        num_periods)
#
#     print()
#     print("x_test_batches")
#     print(x_test_batches)
#     print(len(x_test_batches))
#     print()
#
#     y_test_batches, y_test_data_index, y_test_data_series, test_starts, test_ends = myRNNDataManipulatorModified.getOutputTestRNNDataset(
#         data_1, f_horizon, num_periods)
#
#     print()
#     print("y_test_batches")
#     print(y_test_batches)
#     print(len(y_test_batches))
#     print()