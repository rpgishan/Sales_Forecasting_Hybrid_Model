import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os

from MyARIMA import MyARIMAMod
from MyDataManupulator import MyRNNDataManipulatorModified, MyOtherFunctions, MyJsonGenerator, MyCSVHandler
from MyMongo import MyMongoDataFunctions
from MyRNNModified import MyRNNMod, MyRNNModForPredict
import MyConstants
import json


class MyFinalModelLSTM:
    status = ""

    def __init__(self, models_folder, num_periods, f_horizon):
        print("run - ", self.__class__.__name__)

        # models_folder += "LSTM/"
        self.models_folder = models_folder  # "../TFModels/Mac_Rev/"
        # if not os.path.exists(self.models_folder):
        #     os.makedirs(self.models_folder)

        self.num_periods = num_periods
        self.f_horizon = f_horizon

    def getARIMAResults(self, data,f_horizon):
        # myARIMA = MyARIMA()
        myARIMA = MyARIMAMod()
        data_ARIMA = myARIMA.getPredictedARIMA(data,f_horizon)
        # data_ARIMA = myARIMA.getARIMAModel(data)
        # data_ARIMA_diff = myOtherFunctio
        return data_ARIMA

    def getRNNResultsWithOneInput(self, past_sales_data, maxEpochs=1000, hidden=50, learning_rate=0.001,
                                  minimum_mse=0.01,training_percentage=80):
        models_folder = self.models_folder
        models_folder += "OneInput/"
        num_periods = self.num_periods
        f_horizon = self.f_horizon
        inputs = 1
        output = 1

        data_1 = past_sales_data

        myRNNDataManipulator = MyRNNDataManipulatorModified(trainingpercentage=training_percentage)

        x_train_data, x_train_batches = myRNNDataManipulator.getInputRNNTrainDatasetWithOneInput(data_1, f_horizon,
                                                                                                 num_periods)
        y_train_data, y_train_data_index, y_train_batches, y_train_data_series = myRNNDataManipulator.getOutputRNNTrainDataset(
            data_1, f_horizon, num_periods)

        x_test_data = myRNNDataManipulator.getInputRNNTestDatasetWithOneInput(data_1, f_horizon, num_periods)
        y_test_data, y_test_data_index, y_test_data_series, test_starts, test_ends = myRNNDataManipulator.getOutputRNNTestDataset(
            data_1,
            f_horizon,
            num_periods)

        myRNN = MyRNNMod(models_folder, num_periods, inputs, hidden, output, learning_rate)

        trainMSE_series, testMSE_series, trainMSE_series_norm, testMSE_series_norm, models_path_test_mse_series, optimum_model_path, optimum_model_epoch = myRNN.train_the_model(
            x_train_batches,
            y_train_batches,
            x_test_data,
            y_test_data,
            minimum_mse, maxEpochs)

        y_train_pred, finalMSETrain = myRNN.test_the_model(x_train_batches, y_train_batches)
        y_train_pred_series = pd.Series(np.ravel(y_train_pred), y_train_data_index)

        y_test_pred, finalMSETest = myRNN.test_the_model(x_test_data, y_test_data)
        y_test_pred_series = pd.Series(np.ravel(y_test_pred), y_test_data_index)

        result_dict = {'trainMSE_series': trainMSE_series, 'testMSE_series': testMSE_series,
                       'y_train_data_series': y_train_data_series, 'y_train_pred_series': y_train_pred_series,
                       'finalMSETrain': finalMSETrain, 'y_test_data_series': y_test_data_series,
                       'y_test_pred_series': y_test_pred_series, 'finalMSETest': finalMSETest, 'hidden': hidden,
                       'maxEpochs': maxEpochs,
                       'test_starts': test_starts, 'test_ends': test_ends,
                       'trainMSE_series_norm': trainMSE_series_norm, 'testMSE_series_norm': testMSE_series_norm,
                       'optimum_model_path': optimum_model_path,"optimum_model_epoch":optimum_model_epoch}

        return result_dict

    def getRNNResultsWithFourInputs(self, past_sales_data, positive_count, negative_count, data_ARIMA, maxEpochs=10000,
                                    hidden=150, learning_rate=0.001, minimum_mse=0.01,training_percentage=80):
        models_folder = self.models_folder
        models_folder += "FourInputs/"
        num_periods = self.num_periods
        f_horizon = self.f_horizon
        inputs = 4
        output = 1

        data_1 = past_sales_data
        data_2 = positive_count
        data_3 = negative_count
        data_4 = data_ARIMA

        myRNNDataManipulator = MyRNNDataManipulatorModified(trainingpercentage=training_percentage)

        x_train_data, x_train_batches = myRNNDataManipulator.getInputRNNTrainDatasetWithFourInputs(data_1, data_2,
                                                                                                   data_3, data_4,
                                                                                                   f_horizon,
                                                                                                   num_periods)
        y_train_data, y_train_data_index, y_train_batches, y_train_data_series = myRNNDataManipulator.getOutputRNNTrainDataset(
            data_1, f_horizon, num_periods)

        x_test_batches = myRNNDataManipulator.getInputRNNTestDatasetWithFourInputs(data_1, data_2, data_3, data_4,
                                                                                   f_horizon, num_periods)
        y_test_batches, y_test_data_index, y_test_data_series, test_starts, test_ends = myRNNDataManipulator.getOutputRNNTestDataset(
            data_1,
            f_horizon,
            num_periods)

        myRNN = MyRNNMod(models_folder, num_periods, inputs, hidden, output, learning_rate)

        trainMSE_series, testMSE_series, trainMSE_series_norm, testMSE_series_norm, models_path_test_mse_series, optimum_model_path, optimum_model_epoch = myRNN.train_the_model(
            x_train_batches,
            y_train_batches,
            x_test_batches,
            y_test_batches,
            minimum_mse, maxEpochs)

        y_train_pred, finalMSETrain = myRNN.test_the_model(x_train_batches, y_train_batches)
        y_train_pred_series = pd.Series(np.ravel(y_train_pred), y_train_data_index)

        y_test_pred, finalMSETest = myRNN.test_the_model(x_test_batches, y_test_batches)
        y_test_pred_series = pd.Series(np.ravel(y_test_pred), y_test_data_index)

        ## print only the last test batch
        # y_test_pred_series = pd.Series(np.ravel(y_test_pred)[-num_periods:], y_test_data_index[-num_periods:])

        # myRNN_min_test_mse_roughly = MyRNNModForPredictLSTM(models_path_test_mse_series.idxmin(), num_periods, 4,
        #                                                        hidden, 1)
        #
        # y_test_pred_min_test_mse_roughly = myRNN_min_test_mse_roughly.predict_the_model(x_test_batches)
        # y_test_pred_series_min_test_mse_roughly = pd.Series(np.ravel(y_test_pred_min_test_mse_roughly),
        #                                                     y_test_data_index)
        ## print only the last test batch
        # y_test_pred_series_min_test_mse_roughly = pd.Series(np.ravel(y_test_pred_min_test_mse_roughly)[-num_periods:], y_test_data_index[-num_periods:])

        # myRNN_min_test_mse = MyRNNModForPredictLSTM(min_test_mse_save_path, num_periods, 4, hidden, 1)
        #
        # y_test_pred_min_test_mse = myRNN_min_test_mse.predict_the_model(x_test_data)
        # y_test_pred_series_min_test_mse = pd.Series(np.ravel(y_test_pred_min_test_mse), y_test_data_index)
        #
        #
        # myRNN_min_test_mse_sess = MyRNNModForPredictLSTM(min_test_mse_save_path_sess_save, num_periods, 4, hidden, 1)
        #
        # y_test_pred_min_test_mse_sess = myRNN_min_test_mse_sess.predict_the_model(x_test_data)
        # y_test_pred_series_min_test_mse_sess = pd.Series(np.ravel(y_test_pred_min_test_mse_sess), y_test_data_index)

        # print("optimum_model_path")
        # print(optimum_model_path)

        # myRNN_optimum = MyRNNModForPredict(optimum_model_path, num_periods, inputs, hidden, output)
        #
        # y_test_pred_test_optimum = myRNN_optimum.predict_the_model(x_test_batches)
        # y_test_pred_series_test_optimum = pd.Series(np.ravel(y_test_pred_test_optimum), y_test_data_index)

        # print('\noptimum_model_path : ', optimum_model_path)
        # print()

        # result_dict = {'epochs_ar':epochs_ar, 'trainMSE_ar':trainMSE_ar, 'testMSE_ar':testMSE_ar,'y_train_data_series':y_train_data_series, 'y_train_pred_series':y_train_pred_series, 'finalMSETrain':finalMSETrain,'y_test_data_series':y_test_data_series, 'y_test_pred_series':y_test_pred_series, 'finalMSETest':finalMSETest,'hidden':hidden,'maxEpochs':maxEpochs,'y_test_pred_series_min_test_mse_roughly':y_test_pred_series_min_test_mse_roughly,'y_test_pred_series_min_test_mse':y_test_pred_series_min_test_mse,'y_test_pred_series_min_test_mse_sess':y_test_pred_series_min_test_mse_sess}
        result_dict = {'trainMSE_series': trainMSE_series, 'testMSE_series': testMSE_series,
                       'y_train_data_series': y_train_data_series, 'y_train_pred_series': y_train_pred_series,
                       'finalMSETrain': finalMSETrain, 'y_test_data_series': y_test_data_series,
                       'y_test_pred_series': y_test_pred_series, 'finalMSETest': finalMSETest, 'hidden': hidden,
                       'maxEpochs': maxEpochs,
                       'test_starts': test_starts, 'test_ends': test_ends,
                       'trainMSE_series_norm': trainMSE_series_norm, 'testMSE_series_norm': testMSE_series_norm,
                       'optimum_model_path': optimum_model_path,"optimum_model_epoch":optimum_model_epoch}

        # print('models_path_test_mse_series')
        # print(models_path_test_mse_series)
        # print('models_path_test_mse_series.idxmin()')
        # print(models_path_test_mse_series.idxmin())
        # print()
        # print('min_test_mse_save_path')
        # print(min_test_mse_save_path)
        # print('minimum_test_mse')
        # print(minimum_test_mse)

        return result_dict

    def getRNNValidationWithOneInputs(self, past_sales_data, hidden=150, learning_rate=0.001):
        models_folder = self.models_folder
        models_folder += "OneInput/"
        num_periods = self.num_periods
        f_horizon = self.f_horizon
        inputs = 1
        output = 1

        data_1 = past_sales_data

        myRNNDataManipulator = MyRNNDataManipulatorModified()

        myRNN = MyRNNMod(models_folder, num_periods, inputs, hidden, output, learning_rate)

        x_validate_data, x_validate_batches = myRNNDataManipulator.getInputRNNValidationDatasetWithOneInputs(data_1,
                                                                                                             f_horizon,
                                                                                                             num_periods)

        y_validate_batches, y_validate_data_index, y_validate_data_series, validate_starts, validate_ends = myRNNDataManipulator.getOutputRNNValidationDataset(
            past_sales_data, num_periods)

        y_pred, mseVerify = myRNN.validate_the_model(x_validate_batches, y_validate_batches)
        y_pred_series = pd.Series(np.ravel(y_pred), y_validate_data_index)

        return y_validate_data_series, y_pred_series, validate_starts, validate_ends, mseVerify

    def getRNNValidationWithFourInputs(self, past_sales_data, positive_count, negative_count, data_ARIMA, hidden=150,
                                       learning_rate=0.001):
        models_folder = self.models_folder
        models_folder += "FourInputs/"
        num_periods = self.num_periods
        f_horizon = self.f_horizon
        inputs = 4
        output = 1

        data_1 = past_sales_data
        data_2 = positive_count
        data_3 = negative_count
        data_4 = data_ARIMA

        myRNNDataManipulator = MyRNNDataManipulatorModified()

        myRNN = MyRNNMod(models_folder, num_periods, inputs, hidden, output, learning_rate)

        x_validate_data, x_validate_batches = myRNNDataManipulator.getInputRNNValidationDatasetWithFourInputs(data_1,
                                                                                                              data_2,
                                                                                                              data_3,
                                                                                                              data_4,
                                                                                                              f_horizon,
                                                                                                              num_periods)

        y_validate_batches, y_validate_data_index, y_validate_data_series, validate_starts, validate_ends = myRNNDataManipulator.getOutputRNNValidationDataset(
            past_sales_data, num_periods)

        y_pred, mseVerify = myRNN.validate_the_model(x_validate_batches, y_validate_batches)
        y_pred_series = pd.Series(np.ravel(y_pred), y_validate_data_index)

        return y_validate_data_series, y_pred_series, validate_starts, validate_ends, mseVerify

    def getRNNPredictionsWithFourInputs(self, past_sales_data, positive_count, negative_count, data_ARIMA, hidden=150):
        models_folder = self.models_folder
        models_folder += "FourInputs/"
        model_path = models_folder + "model"
        num_periods = self.num_periods
        f_horizon = self.f_horizon
        inputs = 4
        output = 1

        data_1 = past_sales_data
        data_2 = positive_count
        data_3 = negative_count
        data_4 = data_ARIMA

        myRNNDataManipulator = MyRNNDataManipulatorModified()

        myRNN = MyRNNModForPredict(model_path, num_periods, inputs, hidden, output)

        x_data, x_batches, predict_starts, predict_ends = myRNNDataManipulator.getInputRNNPredictionDatasetWithFourInputs(data_1, data_2,
                                                                                                      data_3, data_4,
                                                                                                      f_horizon,
                                                                                                      num_periods)

        y_data_index = data_4[predict_starts+f_horizon:predict_ends+f_horizon].index

        # print("getRNNPredictionsWithFourInputs - x_batches")
        # print(x_batches)

        y_pred = myRNN.predict_the_model(x_batches)
        # print("getRNNPredictionsWithFourInputs - y_pred")
        # print(y_pred)
        y_pred_series = pd.Series(np.ravel(y_pred), y_data_index)

        return y_pred_series, predict_starts, predict_ends


if __name__ == "__main__":

    num_periods = 10# MyConstants.num_periods
    f_horizon = 1 #MyConstants.f_horizon
    training_percentage = MyConstants.training_percentage
    # name_of_the_domain = "Reviews"
    doc_id = "2256becc-4787-4ae7-9cbd-ce39421f7139"
    doc_id2 = "MacBook"
    index_col = MyConstants.index_col
    value_col = MyConstants.value_col
    hidden_nodes = 20#MyConstants.hidden_nodes

    database_name = MyConstants.database_name #"Invictus_Reviews_Analysis"
    collection_name = MyConstants.collection_name  # "Reviews"  # "Mac_Sample"

    random_period = MyConstants.random_period
    figure_size = MyConstants.figure_size

    models_folder = "TFModels/" + str(doc_id2) + "/np_" + str(num_periods) + "/fh_" + str(
        f_horizon) + "/hidnod_" + str(hidden_nodes) + "/tp_"+str(training_percentage)+"/"


    plot_folder = "Plots/" + str(doc_id2) + "/"

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_folder += "np_" + str(num_periods) + "_fh_" + str(
        f_horizon) + "_hidnod_" + str(hidden_nodes) + "_tp_"+str(training_percentage)+"_"

    plt_extension = ".pdf"

# Train & test
if __name__ == "__main__":
    # myFinalModel = MyFinalModel(models_folder,num_periods,f_horizon)
    # myFinalModel = MyFinalModelLSTMMultilayers(models_folder,num_periods,f_horizon)
    # source_url = "https://gishtestings.azurewebsites.net/uploads/apple--mac-revenue-in-billions.csv"
    # source_path = "/Users/Gishan/PycharmProjects/ResearchProject/Datasets/apple--mac-revenue in billions.csv"
    # time_resolution = "quarterly"
    max_epochs_for_one = 100
    max_epochs_for_four = 40000

    myFinalModel = MyFinalModelLSTM(models_folder, num_periods, f_horizon)
    myOtherFunctions = MyOtherFunctions()
    myRNNDataManipulator = MyRNNDataManipulatorModified()

    myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)

    # data = myOtherFunctions.getDatasetAR(source_url, index_col, value_col, time_resolution, turn_on_data_filler=True)
    # data = myOtherFunctions.getDatasetARFromPath(source_path, index_col, value_col, time_resolution, turn_on_data_filler=False)
    # print("data - getDatasetARFromPath")
    # print(data)
    # print()
    # print()

    # data = myMongoDataFunctions.readSalesFromDB()
    data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id,value_col, dbkeyname="sales_train")
    print("data - readSalesFromDB")
    print(data)
    print()

    data_ARIMA = myFinalModel.getARIMAResults(data, f_horizon)
    # print("data_ARIMA")
    # print(data_ARIMA)
    # print()

    # RNN with only one input --> past sales data

    rnn_data = myRNNDataManipulator.getDifferentialDataset(data)

    only_rnn_past_sales_data = rnn_data.iloc[:, 0]
    only_rnn_results_dict = myFinalModel.getRNNResultsWithOneInput(only_rnn_past_sales_data, maxEpochs=max_epochs_for_one,
                                                                   hidden=hidden_nodes,training_percentage=training_percentage)
    only_rnn_trainMSE_series = only_rnn_results_dict['trainMSE_series']
    only_rnn_testMSE_series = only_rnn_results_dict['testMSE_series']
    only_rnn_y_train_data_series = only_rnn_results_dict['y_train_data_series']
    only_rnn_y_train_pred_series = only_rnn_results_dict['y_train_pred_series']
    only_rnn_finalMSETrain = only_rnn_results_dict['finalMSETrain']
    only_rnn_y_test_data_series = only_rnn_results_dict['y_test_data_series']
    only_rnn_y_test_pred_series = only_rnn_results_dict['y_test_pred_series']
    only_rnn_finalMSETest = only_rnn_results_dict['finalMSETest']
    only_rnn_hidden = only_rnn_results_dict['hidden']
    only_rnn_maxEpochs = only_rnn_results_dict['maxEpochs']
    # only_rnn_y_test_pred_series_min_test_mse_roughly=only_rnn_results_dict['y_test_pred_series_min_test_mse_roughly']
    # only_rnn_y_test_pred_series_min_test_mse=only_rnn_results_dict['y_test_pred_series_min_test_mse']
    # only_rnn_y_test_pred_series_min_test_mse_sess=only_rnn_results_dict['y_test_pred_series_min_test_mse_sess']
    only_rnn_trainMSE_series_norm = only_rnn_results_dict['trainMSE_series_norm']
    only_rnn_testMSE_series_norm = only_rnn_results_dict['testMSE_series_norm']
    only_rnn_test_starts = only_rnn_results_dict['test_starts']
    only_rnn_test_ends = only_rnn_results_dict['test_ends']
    only_rnn_optimum_model_path = only_rnn_results_dict['optimum_model_path']
    only_rnn_optimum_model_epoch = only_rnn_results_dict["optimum_model_epoch"]

    myMongoDataFunctions.writeRNNDataToDB(doc_id, 1, only_rnn_optimum_model_path)

    # RNN with 4 inputs --> past sales data, positive comments, negative comments, Predicted ARIMA


    rnn_data_with_comments = myRNNDataManipulator.getDifferentialDatasetWithComments(data, f_horizon, random_period,
                                                                                     value_col)

    print("rnn_data_with_comments")
    print(rnn_data_with_comments)
    print()

    past_sales_data = rnn_data_with_comments.iloc[:, 0]
    positive_count = rnn_data_with_comments.iloc[:, 1]
    negative_count = rnn_data_with_comments.iloc[:, 2]

    # print("past_sales_data")
    # print(past_sales_data.values)
    # print()
    # print("positive_count")
    # print(positive_count.values)
    # print()
    # print("negative_count")
    # print(negative_count.values)
    # print()

    rnn_results_dict = myFinalModel.getRNNResultsWithFourInputs(past_sales_data, positive_count, negative_count,
                                                                data_ARIMA,
                                                                maxEpochs=max_epochs_for_four, hidden=hidden_nodes,training_percentage=training_percentage)

    trainMSE_series = rnn_results_dict['trainMSE_series']
    testMSE_series = rnn_results_dict['testMSE_series']
    y_train_data_series = rnn_results_dict['y_train_data_series']
    y_train_pred_series = rnn_results_dict['y_train_pred_series']
    finalMSETrain = rnn_results_dict['finalMSETrain']
    y_test_data_series = rnn_results_dict['y_test_data_series']
    y_test_pred_series = rnn_results_dict['y_test_pred_series']
    finalMSETest = rnn_results_dict['finalMSETest']
    hidden = rnn_results_dict['hidden']
    maxEpochs = rnn_results_dict['maxEpochs']
    # y_test_pred_series_test_optimum = rnn_results_dict['y_test_pred_series_test_optimum']
    # y_test_pred_series_min_test_mse=rnn_results_dict['y_test_pred_series_min_test_mse']
    # y_test_pred_series_min_test_mse_sess=rnn_results_dict['y_test_pred_series_min_test_mse_sess']
    trainMSE_series_norm = rnn_results_dict['trainMSE_series_norm']
    testMSE_series_norm = rnn_results_dict['testMSE_series_norm']
    test_starts = rnn_results_dict['test_starts']
    test_ends = rnn_results_dict['test_ends']
    optimum_model_path = rnn_results_dict['optimum_model_path']
    optimum_model_epoch = rnn_results_dict["optimum_model_epoch"]

    myMongoDataFunctions.writeRNNDataToDB(doc_id, 4, optimum_model_path)

    # print()
    # print("y_test_data_series")
    # print(len(y_test_data_series))
    # print()

    y_test_data_series = y_test_data_series[(-num_periods):]
    y_test_data_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], y_test_data_series)

    y_test_pred_series = y_test_pred_series[(-num_periods):]
    y_test_pred_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], y_test_pred_series)

    only_rnn_y_test_pred_series = only_rnn_y_test_pred_series[(-num_periods):]
    only_rnn_y_test_pred_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], only_rnn_y_test_pred_series)

    test_arima_data = data_ARIMA[(test_ends - num_periods + f_horizon):(test_ends + f_horizon)]
    test_arima_data = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], test_arima_data)


    # print(y_test_pred_series)

    # print('only_rnn_epochs_ar')
    # print(only_rnn_epochs_ar)
    # print()
    # print('only_rnn_trainMSE_ar')
    # print(only_rnn_trainMSE_ar)
    # print()
    # print('only_rnn_testMSE_ar')
    # print(only_rnn_testMSE_ar)
    # print()
    #
    # print('epochs_ar')
    # print(epochs_ar)
    # print()
    # print('trainMSE_ar')
    # print(trainMSE_ar)
    # print()
    # print('testMSE_ar')
    # print(testMSE_ar)
    # print()

    details_text = " | Num periods: " + str(num_periods) + " | F_Horizon: " + str(f_horizon)



    plt_name = "MSE vs Epochs 1 input"
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 1 input" + details_text)
    plt.plot(only_rnn_trainMSE_series, color="green", label="trainMSE RNN model")
    plt.plot(only_rnn_testMSE_series, color="violet", label="testMSE RNN model")
    plt.legend(loc="best")
    plt.savefig(plot_folder+plt_name+plt_extension)
    plt.show()

    plt_name = "MSE vs Epochs 1 input Normalized"
    mse_epoch_1_test_label = "testMSE Hybrid model | opt epoch - "+str(only_rnn_optimum_model_epoch)
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 1 input Normalized" + details_text)
    plt.plot(only_rnn_trainMSE_series_norm, color="blue", label="trainMSE Hybrid model")
    plt.plot(only_rnn_testMSE_series_norm, color="red", label=mse_epoch_1_test_label)
    plt.legend(loc="best")
    plt.savefig(plot_folder+plt_name+plt_extension)
    plt.show()

    plt_name = "MSE vs Epochs 4 inputs series type"
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 4 inputs series type" + details_text)
    plt.plot(trainMSE_series, color="blue", label="trainMSE Hybrid model")
    plt.plot(testMSE_series, color="red", label="testMSE Hybrid model")
    plt.legend(loc="best")
    plt.savefig(plot_folder+plt_name+plt_extension)
    plt.show()

    plt_name = "MSE vs Epochs 4 inputs Normalized"
    mse_epoch_4_test_label = "testMSE Hybrid model | opt epoch - "+str(optimum_model_epoch)
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 4 inputs Normalized" + details_text)
    plt.plot(trainMSE_series_norm, color="blue", label="trainMSE Hybrid model")
    plt.plot(testMSE_series_norm, color="red", label=mse_epoch_4_test_label)
    plt.legend(loc="best")
    plt.savefig(plot_folder+plt_name+plt_extension)
    plt.show()

    # plt.figure(figsize=figure_size)
    # # plt.title(('F vs A Train MSE:', mseTrain, ' --  Test MSE:', mseTest, " - hidden:", hidden), fontsize=20)
    # plt.subplot(211)
    # plt.title(('Forecast vs Actual Train - MSE:', finalMSETrain, " - Hidden:", hidden, " - Epochs:", maxEpochs), fontsize=20)
    # plt.plot(y_train_data_series, "go", markersize=5, label="Actual Train Y")
    # plt.plot(y_train_data_series, color='green')
    # plt.plot(y_train_pred_series, "y.", markersize=10, label="Forecast Train")
    # plt.plot(y_train_pred_series, color='black')
    # plt.legend(loc="upper left")
    # plt.subplot(212)
    # plt.title(('Forecast vs Actual Test - MSE:', finalMSETest, " - Hidden:", hidden, " - Epochs:", maxEpochs), fontsize=20)
    # plt.plot(y_test_data_series, "bo", markersize=5, label="Actual Test Y")
    # plt.plot(y_test_data_series, color='blue')
    # plt.plot(y_test_pred_series, "r.", markersize=10, label="Forecast Test")
    # plt.plot(y_test_pred_series, color='red')
    # plt.legend(loc="upper left")
    # plt.show()

    # print()
    # print("data_ARIMA")
    # print(len(data_ARIMA))
    # print()

    only_rnn_label = "RNN Prediction RMSE: " + str(only_rnn_finalMSETest)
    actual_label = "Actual --> Train RMSE: " + str(finalMSETrain)
    hybrid_label = "Hybrid Prediction Test RMSE: " + str(finalMSETest)

    plt_name = "All Models - Test"
    plt.figure(figsize=figure_size)
    plt.title('All Models - Test' + details_text)

    plt.plot(y_test_data_series, "b.", markersize=30)
    plt.plot(y_test_data_series, color='blue', label='Actual')
    plt.plot(y_test_pred_series, "r.", markersize=25)
    plt.plot(y_test_pred_series, color='red', label=hybrid_label)
    plt.plot(only_rnn_y_test_pred_series, "y.", markersize=18)
    plt.plot(only_rnn_y_test_pred_series, color='yellow', label=only_rnn_label)
    plt.plot(test_arima_data, "g.", markersize=15)
    plt.plot(test_arima_data, color='green', label='ARIMA Prediction')
    plt.legend(loc='best')
    plt.savefig(plot_folder+plt_name+plt_extension)
    plt.show(block=False)

    ##MAPE

    test_four_mape_data_series, average_test_four_mape_data_series = myOtherFunctions.getMeanAbsolutePercentageError(
        y_test_data_series, y_test_pred_series)
    test_one_mape_data_series, average_test_one_mape_data_series = myOtherFunctions.getMeanAbsolutePercentageError(
        y_test_data_series, only_rnn_y_test_pred_series)
    test_arima_mape_data_series, average_test_arima_mape_data_series = myOtherFunctions.getMeanAbsolutePercentageError(
        y_test_data_series, test_arima_data)

    # print("test_four_mape_data_series")
    # print(test_four_mape_data_series)
    # print("test_one_mape_data_series")
    # print(test_one_mape_data_series)
    # print("test_arima_data")
    # print(test_arima_data)
    # print("test_arima_mape_data_series")
    # print(test_arima_mape_data_series)

    test_mape_data_series = pd.DataFrame({"Four Inputs RNN MAPE": test_four_mape_data_series.values,
                                          "One Input RNN MAPE": test_one_mape_data_series.values,
                                          "ARIMA MAPE": test_arima_mape_data_series.values},
                                         test_four_mape_data_series.index)
    # validate_mape_data_series.index.name = "Index"


    mape_source_folder = "MAPE CSV/Test/"
    mape_file_name = "mspe-doc_" + str(doc_id2) +"-np_" + str(num_periods) + "-fh_" + str(f_horizon) + "-hidnod_" + str(hidden_nodes)+ "_tp_"+str(training_percentage)

    MyCSVHandler().writeToACSV(mape_source_folder, mape_file_name, test_mape_data_series)

    # print("test_mape_data_series")
    # print(test_four_mape_data_series)
    # print(test_one_mape_data_series)
    # print(test_arima_mape_data_series)
    # print()

    # print("mean_test_mape_data_series")
    # print(average_test_four_mape_data_series)
    # print(average_test_one_mape_data_series)
    # print(average_test_arima_mape_data_series)
    # print()

    hybrid_mape_label_test = "Hybrid Prediction Test Average MAPE: " + str(average_test_four_mape_data_series)
    only_rnn_mape_label_test = "RNN Prediction Test Average MAPE: " + str(average_test_one_mape_data_series)
    arima_mape_label_test = "ARIMA Prediction Test Average MAPE: " + str(average_test_arima_mape_data_series)

    plt_name = "All Models - MAPE - Test"
    plt.figure(figsize=figure_size)
    plt.title('All Models - MAPE - Test' + details_text)
    plt.plot(test_four_mape_data_series, "r.", markersize=25)
    plt.plot(test_four_mape_data_series, color='red', label=hybrid_mape_label_test)
    plt.plot(test_one_mape_data_series, "y.", markersize=20)
    plt.plot(test_one_mape_data_series, color='yellow', label=only_rnn_mape_label_test)
    plt.plot(test_arima_mape_data_series, "g.", markersize=15)
    plt.plot(test_arima_mape_data_series, color='green', label=arima_mape_label_test)
    plt.legend(loc='best')
    plt.savefig(plot_folder+plt_name+plt_extension)
    plt.show(block=False)

# validate
if __name__ == "__main__2":
    myFinalModel = MyFinalModelLSTM(models_folder, num_periods, f_horizon)
    myOtherFunctions = MyOtherFunctions()
    myRNNDataManipulator = MyRNNDataManipulatorModified()

    myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)

    # data = myMongoDataFunctions.readSalesFromDB()
    data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id, value_col, dbkeyname="sales_validate")
    # print("data - readSalesFromDB")
    # print(data)
    # print()

    data_ARIMA = myFinalModel.getARIMAResults(data,f_horizon)

    # print("data_ARIMA")
    # print(data_ARIMA)
    # print()

    # 1 input
    rnn_data = myRNNDataManipulator.getDifferentialDataset(data)

    only_rnn_past_sales_data = rnn_data.iloc[:, 0]

    only_rnn_y_validate_data_series, only_rnn_y_validate_pred_series, only_rnn_validate_starts, only_rnn_validate_ends, only_rnn_mse_validate = myFinalModel.getRNNValidationWithOneInputs(
        only_rnn_past_sales_data, hidden=hidden_nodes)

    # 4 inputs
    rnn_data_with_comments = myRNNDataManipulator.getDifferentialDatasetWithComments(data, f_horizon, random_period,
                                                                                     value_col)
    #
    # print("rnn_data_with_comments")
    # print(rnn_data_with_comments)
    # print()

    past_sales_data = rnn_data_with_comments.iloc[:, 0]
    positive_count = rnn_data_with_comments.iloc[:, 1]
    negative_count = rnn_data_with_comments.iloc[:, 2]

    y_validate_data_series, y_validate_pred_series, validate_starts, validate_ends, mse_validate = myFinalModel.getRNNValidationWithFourInputs(
        past_sales_data, positive_count, negative_count, data_ARIMA, hidden=hidden_nodes)

    # print("validate_mape_data_series")
    # print(validate_mape_data_series)

    # y_validate_data_series = y_validate_data_series  # [(-num_periods):]
    # y_pred_series = y_pred_series  # [(-num_periods):]

    validate_arima_data = data_ARIMA[validate_starts:validate_ends]

    y_validate_data_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], y_validate_data_series)
    y_validate_pred_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], y_validate_pred_series)
    only_rnn_y_validate_data_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], only_rnn_y_validate_data_series)
    only_rnn_y_validate_pred_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], only_rnn_y_validate_pred_series)
    validate_arima_data = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], validate_arima_data)

    details_text = " | Num periods: " + str(num_periods) + " | F_Horizon: " + str(f_horizon)

    only_rnn_label_validate = "RNN Prediction Validate RMSE: " + str(only_rnn_mse_validate)
    hybrid_label_validate = "Hybrid Prediction Validate RMSE: " + str(mse_validate)

    plt.figure(figsize=figure_size)
    plt.title('All Models - Validate' + details_text)
    plt.plot(y_validate_data_series, "b.", markersize=30)
    plt.plot(y_validate_data_series, color='blue', label='Actual')
    plt.plot(y_validate_pred_series, "r.", markersize=25)
    plt.plot(y_validate_pred_series, color='red', label=hybrid_label_validate)
    # plt.plot(only_rnn_y_validate_pred_series, "y.", markersize=18)
    # plt.plot(only_rnn_y_validate_pred_series, color='yellow', label=only_rnn_label_validate)
    plt.plot(validate_arima_data, "g.", markersize=10)
    plt.plot(validate_arima_data, color='green', label='ARIMA Prediction')
    plt.legend(loc='best')
    plt.show(block=False)

    ##MAPE

    validate_four_mape_data_series, average_validate_four_mape_data_series = myOtherFunctions.getMeanAbsolutePercentageError(
        y_validate_data_series, y_validate_pred_series)
    validate_one_mape_data_series, average_validate_one_mape_data_series = myOtherFunctions.getMeanAbsolutePercentageError(
        y_validate_data_series, only_rnn_y_validate_pred_series)
    validate_arima_mape_data_series, average_validate_arima_mape_data_series = myOtherFunctions.getMeanAbsolutePercentageError(
        y_validate_data_series, validate_arima_data)

    validate_mape_data_series = pd.DataFrame({"Four Inputs RNN MAPE": validate_four_mape_data_series.values,
                                              "One Input RNN MAPE": validate_one_mape_data_series.values,
                                              "ARIMA MAPE": validate_arima_mape_data_series.values},
                                             validate_four_mape_data_series.index)
    # validate_mape_data_series.index.name = "Index"
    mape_source_folder = "MAPE CSV/Validate/"
    mape_file_name = "mspe-doc_" + str(doc_id2) +"-np_" + str(num_periods) + "-fh_" + str(f_horizon) + "-hidnod_" + str(hidden_nodes)
    MyCSVHandler().writeToACSV(mape_source_folder, mape_file_name, validate_mape_data_series)

    # print("validate_mape_data_series")
    # print(validate_four_mape_data_series)
    # print(validate_one_mape_data_series)
    # print(validate_arima_mape_data_series)
    # print()

    # print("mean_validate_mape_data_series")
    # print(average_validate_four_mape_data_series)
    # print(average_validate_one_mape_data_series)
    # print(average_validate_arima_mape_data_series)
    # print()

    hybrid_mape_label_validate = "Hybrid Prediction Validate Average MAPE: " + str(average_validate_four_mape_data_series)
    only_rnn_mape_label_validate = "RNN Prediction Validate Average MAPE: " + str(average_validate_one_mape_data_series)
    arima_mape_label_validate = "ARIMA Prediction Validate Average MAPE: " + str(average_validate_arima_mape_data_series)

    plt.figure(figsize=figure_size)
    plt.title('All Models - MAPE - Validate' + details_text)
    plt.plot(validate_four_mape_data_series, "r.", markersize=25)
    plt.plot(validate_four_mape_data_series, color='red', label=hybrid_mape_label_validate)
    plt.plot(validate_one_mape_data_series, "y.", markersize=20)
    plt.plot(validate_one_mape_data_series, color='yellow', label=only_rnn_mape_label_validate)
    plt.plot(validate_arima_mape_data_series, "g.", markersize=15)
    plt.plot(validate_arima_mape_data_series, color='green', label=arima_mape_label_validate)
    plt.legend(loc='best')
    plt.show(block=False)


    ## create json results
    y_validate_actual_json_ar = list()

    for index in y_validate_data_series.index:
        new_index_1 = str(index.to_pydatetime().year)+"/"+str(index.to_pydatetime().month)+"/"+str(index.to_pydatetime().day)
        json_obj_1 = {"name": new_index_1, "value": float(y_validate_data_series[index])}
        y_validate_actual_json_ar.append(json_obj_1)

    y_validate_pred_json_ar = list()

    for index in y_validate_pred_series.index:
        new_index_2 = str(index.to_pydatetime().year)+"/"+str(index.to_pydatetime().month)+"/"+str(index.to_pydatetime().day)
        json_obj_2 = {"name": new_index_2, "value": float(y_validate_pred_series[index])}
        y_validate_pred_json_ar.append(json_obj_2)

    validate_arima_json_ar = list()

    for index in validate_arima_data.index:
        new_index_3 = str(index.to_pydatetime().year)+"/"+str(index.to_pydatetime().month)+"/"+str(index.to_pydatetime().day)
        json_obj_3 = {"name": new_index_3, "value": float(validate_arima_data[index])}
        validate_arima_json_ar.append(json_obj_3)


    y_validate_actual_json = {'name':'Validate Actual','series':y_validate_actual_json_ar}
    y_validate_pred_json = {'name':'Validate Predict','series':y_validate_pred_json_ar}
    validate_arima_json = {'name':'Validate Predict','series':validate_arima_json_ar}

    multi_json_ar = [y_validate_actual_json,y_validate_pred_json]

    result_json = json.dumps({'multi':multi_json_ar})

    print("result_json")
    print(result_json)
    print()

# predict
if __name__ == "__main__3":
    myFinalModel = MyFinalModelLSTM(models_folder, num_periods, f_horizon)
    # myOtherFunctions = MyOtherFunctions()
    myRNNDataManipulator = MyRNNDataManipulatorModified()

    myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)

    # data = myMongoDataFunctions.readSalesFromDB()
    data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id, value_col, dbkeyname="sales_predicted")
    # print("data - readSalesFromDB")
    # print(data)
    # print()

    data_ARIMA = myFinalModel.getARIMAResults(data,f_horizon)

    # print("data_ARIMA")
    # print(data_ARIMA)
    # print()

    # 4 inputs
    rnn_data_with_comments = myRNNDataManipulator.getDifferentialDatasetWithComments(data, f_horizon, random_period, value_col)
    #
    # print("rnn_data_with_comments")
    # print(rnn_data_with_comments)
    # print()

    past_sales_data = rnn_data_with_comments.iloc[:, 0]
    positive_count = rnn_data_with_comments.iloc[:, 1]
    negative_count = rnn_data_with_comments.iloc[:, 2]

    y_pred_series, predict_starts, predict_ends = myFinalModel.getRNNPredictionsWithFourInputs(past_sales_data, positive_count, negative_count, data_ARIMA, hidden=hidden_nodes)
    y_pred_series_size = len(y_pred_series)
    y_pred_series_1 = y_pred_series[:y_pred_series_size-f_horizon]
    y_pred_series_2 = y_pred_series[y_pred_series_size-f_horizon:]

    predict_arima_data = data_ARIMA[predict_starts+f_horizon:predict_ends+f_horizon]

    details_text = " | Num periods: " + str(num_periods) + " | F_Horizon: " + str(f_horizon)


    plt.figure(figsize=figure_size)
    plt.title('All Models - Predict' + details_text)
    plt.plot(y_pred_series_1, "b.", markersize=25)
    plt.plot(y_pred_series_1, color='black', label="Hybrid Prediction 1")
    plt.plot(y_pred_series_2, "r.", markersize=25)
    plt.plot(y_pred_series_2, color='red', label="Hybrid Prediction 2")
    plt.plot(predict_arima_data, "g.", markersize=10)
    plt.plot(predict_arima_data, color='green', label='ARIMA Prediction')
    plt.legend(loc='best')
    plt.show(block=False)