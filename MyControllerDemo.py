from flask import Flask, request, jsonify
from MyFinalModel import MyFinalModelLSTM
from MyDataManupulator import MyOtherFunctions,MyRNNDataManipulatorModified,MyJsonGenerator,MyCSVHandler
from MyMongo import MyMongoDataFunctions
# import json
import pandas as pd
from werkzeug.utils import secure_filename
import os
import matplotlib.pylab as plt
from flask_cors import CORS
import MyConstants
# from datetime import datetime

app = Flask(__name__)
CORS(app)

status = -1  # not initialized

doc_id2 = 0

# database_name = "Research_Project_DB"
database_name = MyConstants.database_name  # "Invictus_Reviews_Analysis"
collection_name = MyConstants.collection_name  # "Reviews"
ALLOWED_EXTENSIONS = set(['csv'])
file_upload_folder = MyConstants.temp_file_upload_folder  #  "Temp_Uploads/"
if not os.path.exists(file_upload_folder):
    os.makedirs(file_upload_folder)

num_periods = MyConstants.num_periods
f_horizon = MyConstants.f_horizon
training_percentage = MyConstants.training_percentage
hidden_nodes = MyConstants.hidden_nodes
random_period = MyConstants.random_period
index_col = MyConstants.index_col
value_col = MyConstants.value_col
# time_resolution = "quarterly"

figure_size = MyConstants.figure_size

myOtherFunctions = MyOtherFunctions()
myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)


def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def hello():
    result = jsonify({"API GET":"/api","write predict csv doc POST  - file and time_resolution(quarterly, monthly, daily)":"/api/writepredictcsv","Get prediction GET":"/api/predict/<doc_id>",})
    return result


@app.route("/api", methods=["GET"])
def hello_api():
    return "Hello API"


#### have to modify
@app.route("/api/valid_dates/<doc_id>", methods=["POST"])
def getValidateDates(doc_id):
    data_date_indices = myMongoDataFunctions.readSalesAvailableDatesFromDBWithColName(dbkeyname="sales_train")
    return status


@app.route("/api/status", methods=["GET"])
def getcurrentstatus():
    global status
    return status


@app.route("/api/train", methods=["POST"])
def trainTheModel():
    doc_id = request.args.get('processid')
    doc_id2 = request.args.get('docid')
    global status

    status = 1  # training starts

    max_epochs_for_one = 100
    max_epochs_for_four = 20000


    models_folder = "TFModels/" + str(doc_id2) + "/np_" + str(num_periods) + "/fh_" + str(
        f_horizon) + "/hidnod_" + str(hidden_nodes) + "/tp_"+str(training_percentage)+"/"

    plot_folder = "Plots/" + str(doc_id2) + "/"

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_folder += "np_" + str(num_periods) + "_fh_" + str(
        f_horizon) + "_hidnod_" + str(hidden_nodes) + "_tp_" + str(training_percentage) + "_"

    plt_extension = ".pdf"

    mape_source_folder = "MAPE CSV/Test/"
    mape_file_name = "mspe-doc_" + str(doc_id2) +"-np_" + str(num_periods) + "-fh_" + str(f_horizon) + "-hidnod_" + str(
        hidden_nodes) + "_tp_" + str(training_percentage)

    myFinalModel = MyFinalModelLSTM(models_folder, num_periods, f_horizon)
    myOtherFunctions = MyOtherFunctions()
    myRNNDataManipulator = MyRNNDataManipulatorModified()

    myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)
    data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id, value_col, dbkeyname="sales_train")

    data_ARIMA = myFinalModel.getARIMAResults(data, f_horizon)

    # RNN with only one input --> past sales data

    rnn_data = myRNNDataManipulator.getDifferentialDataset(data)

    only_rnn_past_sales_data = rnn_data.iloc[:, 0]
    only_rnn_results_dict = myFinalModel.getRNNResultsWithOneInput(only_rnn_past_sales_data,
                                                                   maxEpochs=max_epochs_for_one,
                                                                   hidden=hidden_nodes,
                                                                   training_percentage=training_percentage)
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

    past_sales_data = rnn_data_with_comments.iloc[:, 0]
    positive_count = rnn_data_with_comments.iloc[:, 1]
    negative_count = rnn_data_with_comments.iloc[:, 2]


    rnn_results_dict = myFinalModel.getRNNResultsWithFourInputs(past_sales_data, positive_count, negative_count,
                                                                data_ARIMA,
                                                                maxEpochs=max_epochs_for_four, hidden=hidden_nodes,
                                                                training_percentage=training_percentage)

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
    trainMSE_series_norm = rnn_results_dict['trainMSE_series_norm']
    testMSE_series_norm = rnn_results_dict['testMSE_series_norm']
    test_starts = rnn_results_dict['test_starts']
    test_ends = rnn_results_dict['test_ends']
    optimum_model_path = rnn_results_dict['optimum_model_path']
    optimum_model_epoch = rnn_results_dict["optimum_model_epoch"]

    myMongoDataFunctions.writeRNNDataToDB(doc_id, 4, optimum_model_path)


    y_test_data_series = y_test_data_series[(-num_periods):]
    y_test_pred_series = y_test_pred_series[(-num_periods):]
    only_rnn_y_test_pred_series = only_rnn_y_test_pred_series[(-num_periods):]
    test_arima_data = data_ARIMA[(test_ends - num_periods + f_horizon):(test_ends + f_horizon)]


    details_text = " | Num periods: " + str(num_periods) + " | F_Horizon: " + str(f_horizon)

    plt_name = "MSE vs Epochs 1 input"
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 1 input" + details_text)
    plt.plot(only_rnn_trainMSE_series, color="green", label="trainMSE RNN model")
    plt.plot(only_rnn_testMSE_series, color="violet", label="testMSE RNN model")
    plt.legend(loc="best")
    plt.savefig(plot_folder + plt_name + plt_extension)
    plt.show()

    plt_name = "MSE vs Epochs 1 input Normalized"
    mse_epoch_1_test_label = "testMSE Hybrid model | opt epoch - " + str(only_rnn_optimum_model_epoch)
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 1 input Normalized" + details_text)
    plt.plot(only_rnn_trainMSE_series_norm, color="blue", label="trainMSE Hybrid model")
    plt.plot(only_rnn_testMSE_series_norm, color="red", label=mse_epoch_1_test_label)
    plt.legend(loc="best")
    plt.savefig(plot_folder + plt_name + plt_extension)
    plt.show()

    plt_name = "MSE vs Epochs 4 inputs series type"
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 4 inputs series type" + details_text)
    plt.plot(trainMSE_series, color="blue", label="trainMSE Hybrid model")
    plt.plot(testMSE_series, color="red", label="testMSE Hybrid model")
    plt.legend(loc="best")
    plt.savefig(plot_folder + plt_name + plt_extension)
    plt.show()

    plt_name = "MSE vs Epochs 4 inputs Normalized"
    mse_epoch_4_test_label = "testMSE Hybrid model | opt epoch - " + str(optimum_model_epoch)
    plt.figure(figsize=figure_size)
    plt.title("MSE vs Epochs 4 inputs Normalized" + details_text)
    plt.plot(trainMSE_series_norm, color="blue", label="trainMSE Hybrid model")
    plt.plot(testMSE_series_norm, color="red", label=mse_epoch_4_test_label)
    plt.legend(loc="best")
    plt.savefig(plot_folder + plt_name + plt_extension)
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
    plt.savefig(plot_folder + plt_name + plt_extension)
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
    plt.savefig(plot_folder + plt_name + plt_extension)
    plt.show(block=False)

    status = 2  # training ends

    # print('rnn_results_dict')
    # print(rnn_results_dict)

    # return "POST"
    # return rnn_results_dict
    # return json.dumps(rnn_results_dict)
    return "Training started"


@app.route("/api/writetraincsv/<doc_id>", methods=["POST"])
def writeTrainDatasetCSVFileToDb(doc_id):
    # doc_id = int(doc_id)
    # print(type(doc_id))
    if 'file' not in request.files:
        return "No file part"
    elif 'time_resolution' not in request.form:
        return "specify time resolution"
    else:
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            global doc_id2
            doc_id2 = os.path.splitext(filename)[0]
            # doc_id = doc_id2
            file_path = os.path.join(file_upload_folder, filename)
            file.save(file_path)
            # print("file_path : ",file_path)
            time_resolution = request.form["time_resolution"]
            # print("time_resolution - ",time_resolution)
            data = myOtherFunctions.getDatasetARFromPath(file_path, index_col, value_col, time_resolution, turn_on_data_filler=True)
            # print(data.keys())
            myMongoDataFunctions.writeSalesToDb(doc_id, data,value_col,dbkeyname="sales_train")
            try:
                os.remove(file_path)
                print("file is deleted")
            except:
                print("file couldn't be deleted")

            # success = jsonify({"success": True, "docid":doc_id2})
            return doc_id2


@app.route("/api/writevalidatecsv/<doc_id>", methods=["POST"])
def writeValidateDatasetCSVFileToDb(doc_id):
    # doc_id = int(doc_id)
    # print(type(doc_id))
    if 'file' not in request.files:
        return "No file part"
    elif 'time_resolution' not in request.form:
        return "specify time resolution"
    else:
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            global doc_id2
            doc_id2 = os.path.splitext(filename)[0]
            # doc_id = doc_id2
            file_path = os.path.join(file_upload_folder, filename)
            file.save(file_path)
            # print("file_path : ",file_path)
            time_resolution = request.form["time_resolution"]
            # print("time_resolution - ",time_resolution)
            data = myOtherFunctions.getDatasetARFromPath(file_path, index_col, value_col, time_resolution, turn_on_data_filler=True)
            # print(data.keys())
            myMongoDataFunctions.writeSalesToDb(doc_id, data,value_col,dbkeyname="sales_validate")
            try:
                os.remove(file_path)
                print("file is deleted")
            except:
                print("file couldn't be deleted")

            # success = jsonify({"success": True, "docid":doc_id2})
            return doc_id2


@app.route("/api/validate", methods=["GET"])
def getValidatePredictions():

    doc_id = request.args.get('processid')
    doc_id2 = request.args.get('docid')

    models_folder = "TFModels/" + str(doc_id2) + "/np_" + str(num_periods) + "/fh_" + str(
    f_horizon) + "/hidnod_" + str(hidden_nodes) + "/tp_"+str(training_percentage)+"/"

    myFinalModel = MyFinalModelLSTM(models_folder, num_periods, f_horizon)
    myOtherFunctions = MyOtherFunctions()
    myRNNDataManipulator = MyRNNDataManipulatorModified()

    myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)

    # data = myMongoDataFunctions.readSalesFromDB()
    data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id, value_col, dbkeyname="sales_validate")
    # print("data - readSalesFromDB")
    # print(data)
    # print()

    data_ARIMA = myFinalModel.getARIMAResults(data, f_horizon)

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
    only_rnn_y_validate_data_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col],
                                                                                      only_rnn_y_validate_data_series)
    only_rnn_y_validate_pred_series = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col],
                                                                                      only_rnn_y_validate_pred_series)
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
    mape_file_name = "mspe-doc_" + str(doc_id2) + "-np_" + str(num_periods) + "-fh_" + str(
        f_horizon) + "-hidnod_" + str(hidden_nodes)
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

    hybrid_mape_label_validate = "Hybrid Prediction Validate Average MAPE: " + str(
        average_validate_four_mape_data_series)
    only_rnn_mape_label_validate = "RNN Prediction Validate Average MAPE: " + str(average_validate_one_mape_data_series)
    arima_mape_label_validate = "ARIMA Prediction Validate Average MAPE: " + str(
        average_validate_arima_mape_data_series)

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
        new_index_1 = str(index.to_pydatetime().year) + "/" + str(index.to_pydatetime().month) + "/" + str(
            index.to_pydatetime().day)
        json_obj_1 = {"name": new_index_1, "value": float(y_validate_data_series[index])}
        y_validate_actual_json_ar.append(json_obj_1)

    y_validate_pred_json_ar = list()

    for index in y_validate_pred_series.index:
        new_index_2 = str(index.to_pydatetime().year) + "/" + str(index.to_pydatetime().month) + "/" + str(
            index.to_pydatetime().day)
        json_obj_2 = {"name": new_index_2, "value": float(y_validate_pred_series[index])}
        y_validate_pred_json_ar.append(json_obj_2)

    validate_arima_json_ar = list()

    for index in validate_arima_data.index:
        new_index_3 = str(index.to_pydatetime().year) + "/" + str(index.to_pydatetime().month) + "/" + str(
            index.to_pydatetime().day)
        json_obj_3 = {"name": new_index_3, "value": float(validate_arima_data[index])}
        validate_arima_json_ar.append(json_obj_3)

    y_validate_actual_json = {'name': 'Validate Actual', 'series': y_validate_actual_json_ar}
    y_validate_pred_json = {'name': 'Validate Predict', 'series': y_validate_pred_json_ar}
    validate_arima_json = {'name': 'Validate Predict', 'series': validate_arima_json_ar}

    multi_json_ar = [y_validate_actual_json, y_validate_pred_json]

    result_json = jsonify({'multi': multi_json_ar})

    return result_json


@app.route("/api/upload/<doc_id>", methods=["POST"])
def writePredictCSVFileToDb(doc_id):
    # doc_id = int(doc_id)
    # print(type(doc_id))
    if 'file' not in request.files:
        return "No file part"
    else:
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            global doc_id2
            doc_id2 = os.path.splitext(filename)[0]
            # print(doc_id2)
            # doc_id = doc_id2
            file_path = os.path.join(file_upload_folder, filename)
            file.save(file_path)
            # print("file_path : ",file_path)
            # time_resolution = "quarterly"  # request.form["time_resolution"]
            time_resolution = "daily"  # request.form["time_resolution"]
            # print("time_resolution - ",time_resolution)
            data = myOtherFunctions.getDatasetARFromPath(file_path, index_col, value_col, time_resolution, turn_on_data_filler=True)
            myMongoDataFunctions.writeSalesToDb(doc_id, data, value_col,dbkeyname="sales_predicted")
            # myMongoDataFunctions.writeSalesToDb(doc_id2, data, value_col,dbkeyname="sales_predicted")
            try:
                os.remove(file_path)
                print("file is deleted")
            except:
                print("file couldn't be deleted")

            # success = json.dumps({"success": True, "docid":doc_id2})
            return doc_id2


@app.route("/api/predict", methods=["GET"])
def getPredictions():
    doc_id = request.args.get('processid')
    doc_id2 = request.args.get('docid')
    # global doc_id2
    # doc_id = doc_id2

    models_folder = "TFModels/" + str(doc_id2) + "/np_" + str(num_periods) + "/fh_" + str(f_horizon) + "/hidnod_" + str(hidden_nodes) + "/tp_"+str(training_percentage)+"/"

    myFinalModel = MyFinalModelLSTM(models_folder, num_periods, f_horizon)
    # myOtherFunctions = MyOtherFunctions()
    myRNNDataManipulator = MyRNNDataManipulatorModified()

    myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)

    data = myMongoDataFunctions.readSalesFromDBWithColName(doc_id, value_col, dbkeyname="sales_predicted")

    data_ARIMA = myFinalModel.getARIMAResults(data, f_horizon)

    # print("getPredictions - data_ARIMA")
    # print(data_ARIMA)
    # print()

    # 4 inputs
    rnn_data_with_comments = myRNNDataManipulator.getDifferentialDatasetWithComments(data, f_horizon, random_period,
                                                                                     value_col)

    past_sales_data = rnn_data_with_comments.iloc[:, 0]
    positive_count = rnn_data_with_comments.iloc[:, 1]
    negative_count = rnn_data_with_comments.iloc[:, 2]

    y_pred_series, predict_starts, predict_ends = myFinalModel.getRNNPredictionsWithFourInputs(past_sales_data,
                                                                                               positive_count,
                                                                                               negative_count,
                                                                                               data_ARIMA,
                                                                                               hidden=hidden_nodes)

    y_pred_series_non_diff = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], y_pred_series)

    y_pred_series = y_pred_series_non_diff


    y_pred_series_size = len(y_pred_series)
    y_pred_series_1 = y_pred_series[:y_pred_series_size - f_horizon]
    y_pred_series_2 = y_pred_series[y_pred_series_size - f_horizon:]

    # print("y_pred_series_size - ",y_pred_series_size)
    # print()
    # print("y_pred_series")
    # print(y_pred_series)
    # print()
    # print("y_pred_series_1")
    # print(y_pred_series_1)
    # print()
    # print("y_pred_series_2")
    # print(y_pred_series_2)
    # print()


    predict_arima_data = data_ARIMA[predict_starts + f_horizon:predict_ends + f_horizon]

    predict_arima_data_non_diff = myRNNDataManipulator.getNonDifferentiatedSeries(data[value_col], predict_arima_data)
    predict_arima_data = predict_arima_data_non_diff


    myJsonGenerator = MyJsonGenerator()

    # y_validate_actual_json_ar = myJsonGenerator.getJsonArFromDataSeries(y_validate_data_series)

    # only_rnn_y_validate_pred_json_ar = myJsonGenerator.getJsonArFromDataSeries(only_rnn_y_validate_pred_series)
    # y_pred_series_1_json_ar = myJsonGenerator.getJsonArFromDataSeries(y_pred_series_1)
    # y_pred_series_2_json_ar = myJsonGenerator.getJsonArFromDataSeries(y_pred_series_2)
    # validate_arima_json_ar = myJsonGenerator.getJsonArFromDataSeries(validate_arima_data)

    y_pred_series_1_json_ar = list()

    for index in y_pred_series_1.index:
        # print(index," - ",data_series[index])
        # print(type(index.to_pydatetime()))
        # new_index_1 = int((index.to_pydatetime() - datetime.utcfromtimestamp(0)).total_seconds())
        new_index_1 = str(index.to_pydatetime().year)+"/"+str(index.to_pydatetime().month)+"/"+str(index.to_pydatetime().day)
        json_obj_1 = {"name": new_index_1, "value": float(y_pred_series_1[index])}
        y_pred_series_1_json_ar.append(json_obj_1)


    y_pred_series_2_json_ar = list()

    for index in y_pred_series_2.index:
        # new_index_2 = int((index.to_pydatetime() - datetime.utcfromtimestamp(0)).total_seconds())
        new_index_2 = str(index.to_pydatetime().year)+"/"+str(index.to_pydatetime().month)+"/"+str(index.to_pydatetime().day)
        json_obj_2 = {"name": new_index_2, "value": float(y_pred_series_2[index])}
        y_pred_series_2_json_ar.append(json_obj_2)

    # y_validate_json = json.dumps({"y_validate_actual":y_validate_actual_json_ar,"y_validate_pred":y_validate_pred_json_ar,"only_rnn_y_validate_pred":only_rnn_y_validate_pred_json_ar,"validate_arima":validate_arima_json_ar})

    # y_actual_json_string = '{"name":"Actual","series":y_pred_series_1_json_ar}'
    # y_predict_json_string = '{"name":"Predict","series":y_pred_series_2_json_ar}'

    # print(type(y_pred_series_2_json_ar))

    # y_actual_json = json.loads(y_actual_json_string)
    # y_predict_json = json.loads(y_actual_json_string)
    y_actual_json = ({"name":"Actual","series":y_pred_series_1_json_ar})
    y_predict_json = ({"name":"Predict","series":y_pred_series_2_json_ar})

    multi_json_list = [y_actual_json,y_predict_json]


    # result_json = json.dumps({"multi":multi_json_list})
    result_json = jsonify({"multi":multi_json_list})


    details_text = " | Num periods: " + str(num_periods) + " | F_Horizon: " + str(f_horizon)
    figure_size = (16, 10)
    plt.figure(figsize=figure_size)
    plt.title('All Models - Predict' + details_text)
    plt.plot(y_pred_series_1, "b.", markersize=25)
    plt.plot(y_pred_series_1, color='black', label="Hybrid Prediction 1")
    plt.plot(y_pred_series_2, "r.", markersize=25)
    plt.plot(y_pred_series_2, color='red', label="Hybrid Prediction 2")
    # plt.plot(predict_arima_data, "g.", markersize=10)
    # plt.plot(predict_arima_data, color='green', label='ARIMA Prediction')
    plt.legend(loc='best')
    plt.show(block=False)

    return result_json


if __name__ == '__main__':
    app.run(debug=True, port=8082)