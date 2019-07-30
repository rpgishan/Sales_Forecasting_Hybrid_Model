import tensorflow as tf
import time
import pandas as pd
import numpy as np
from MyDataManupulator import MyOtherFunctions
import os
import shutil

# TODO
#  has to modify to save models in every epoch and take the best model

max_no_of_tf_models = 1000


class MyRNNMod:
    def __init__(self, models_folder, num_periods, inputs, hidden, output, learning_rate=0.01, basicCellType="LSTM"):
        tf.reset_default_graph()
        self.inputs = inputs
        self.hidden = hidden
        self.output = output
        self.learning_rate = learning_rate
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        self.model_path = models_folder + "model"
        self.temp_model_folder = models_folder + "temp/"
        self.X = tf.placeholder(tf.float32, [None, num_periods, self.inputs])
        self.Y = tf.placeholder(tf.float32, [None, num_periods, self.output])
        if basicCellType == "RNN":
            self.basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden, activation=tf.nn.relu)
        else:
            self.basic_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden, state_is_tuple=True)
        self.rnn_output, states = tf.nn.dynamic_rnn(self.basic_cell, self.X, dtype=tf.float32)
        self.stacked_rnn_output = tf.reshape(self.rnn_output, [-1, self.hidden])
        self.stacked_output = tf.layers.dense(self.stacked_rnn_output, self.output)
        self.outputs = tf.reshape(self.stacked_output, [-1, num_periods, self.output])
        self.loss_square = tf.squared_difference(self.outputs, self.Y)
        self.loss = tf.sqrt(tf.reduce_mean(self.loss_square))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

        # self.tf_epochs = tf.Variable(10)
        # self.tf_epochs_ar = tf.Variable([0])
        # self.tf_train_MSE = tf.Variable([0])
        # self.tf_test_MSE = tf.Variable([0])

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=max_no_of_tf_models)
        []

    def train_the_model(self, input_training_batch, output_training_batch, input_test_batch, output_test_batch, minimum_mse=10, max_epochs=10000):
        with tf.Session() as sess:
            self.init.run()
            try:
                self.saver.restore(sess, self.model_path)
                print("Got the model")
            except:
                print("No models")

            # print(self.tf_epochs.value())
            # print(self.tf_epochs_ar.value())
            # print(self.tf_train_MSE.value())
            # print(self.tf_test_MSE.value())

            epochs = 0
            epochs_ar = []
            train_MSE = []
            test_MSE = []
            model_path_eps_ar = []
            model_eps_test_MSE_ar = []
            max_epochs += epochs

            mse = self.loss.eval(feed_dict={self.X: input_training_batch, self.Y: output_training_batch})
            premse = mse
            mseTest = self.loss.eval(feed_dict={self.X: input_test_batch, self.Y: output_test_batch})
            premseTest = mseTest
            beginningtime = time.time()
            pretime = beginningtime
            while mse >= minimum_mse and epochs < max_epochs:
                sess.run(self.training_op, feed_dict={self.X: input_training_batch, self.Y: output_training_batch})
                mse = self.loss.eval(feed_dict={self.X: input_training_batch, self.Y: output_training_batch})
                epochs_ar.append(epochs)
                train_MSE.append(mse)
                mseTest = self.loss.eval(feed_dict={self.X: input_test_batch, self.Y: output_test_batch})
                test_MSE.append(mseTest)
                if (epochs % 100 == 0) or (epochs == 10):
                    print(epochs, "\tTrain_MSE: ", mse, "\tTrain_Diff:", (premse - mse), "\tTest_MSE: ", mseTest,
                          "\tTrain_Diff:", (premseTest - mseTest))
                    premse = mse
                    premseTest = mseTest
                    # if epochs % 100 == 0:
                    nowtime = time.time()
                    time_diff = nowtime - pretime
                    totTime = (nowtime - beginningtime)
                    if epochs != 0:
                        estimate_time = (totTime * max_epochs / epochs) - totTime
                        print("Time Diff:", time_diff, "\tEstimated Time:", estimate_time, "\tEstimated Time Mins:", (estimate_time / 60))

                    print("Time Sec:", totTime, "\t Mins:", (totTime / 60))
                    pretime = nowtime

                    # self.tf_epochs.assign(epochs)
                    # self.tf_epochs_ar.assign(epochs_ar)
                    # self.tf_train_MSE.assign(train_MSE)
                    # self.tf_test_MSE.assign(test_MSE)

                    save_path_eps = self.saver.save(sess, (self.temp_model_folder + "model_" + str(epochs)))
                    print("Model saved EPS in file: %s" % save_path_eps)
                    model_path_eps_ar.append(save_path_eps)
                    model_eps_test_MSE_ar.append(mseTest)
                epochs = epochs + 1



            # self.tf_epochs.assign(epochs)
            # self.tf_epochs_ar.assign(epochs_ar)
            # self.tf_train_MSE.assign(train_MSE)
            # self.tf_test_MSE.assign(test_MSE)


            print(epochs, "\tMSE:", mse)
            nowtime = time.time()
            totTime = (nowtime - beginningtime)
            print("Time Sec:", totTime, "\t Mins:", (totTime / 60))

            save_path_eps = self.saver.save(sess, (self.temp_model_folder + "model_" + str(epochs)))
            print("Model saved EPS in file: %s" % save_path_eps)
            model_path_eps_ar.append(save_path_eps)
            model_eps_test_MSE_ar.append(mseTest)

            models_path_test_mse_series = pd.Series(model_eps_test_MSE_ar, model_path_eps_ar)

            trainMSE_series = pd.Series(train_MSE, epochs_ar)
            testMSE_series = pd.Series(test_MSE, epochs_ar)

            myOtherFunctions = MyOtherFunctions()

            trainMSE_series_norm = myOtherFunctions.getNormalizedSeries(trainMSE_series)
            testMSE_series_norm = myOtherFunctions.getNormalizedSeries(testMSE_series)

            trainMSE_norm_ar = np.array(trainMSE_series_norm.values)
            testMSE_norm_ar = np.array(testMSE_series_norm.values)

            intersection_ids = myOtherFunctions.getIntersectionPoints(trainMSE_norm_ar, testMSE_norm_ar)

            intersection_ids_floor = np.array(np.floor(intersection_ids / 100) * 100)
            intersection_ids_ceil = np.array(np.ceil(intersection_ids / 100) * 100)

            intersection_ids_rounded = np.concatenate([intersection_ids_floor, intersection_ids_ceil])
            intersection_ids_rounded = np.unique(intersection_ids_rounded)
            intersection_ids_rounded = intersection_ids_rounded.astype(int)

            intersection_ids_rounded[intersection_ids_rounded >= max_epochs] = max_epochs - 1

            intersection_series = pd.Series(testMSE_series.values[intersection_ids_rounded],
                                            testMSE_series.index[intersection_ids_rounded])

            min_interaction_id = intersection_series.idxmin()

            optimum_model_epoch = min_interaction_id

            if min_interaction_id == max_epochs - 1:
                min_interaction_id = max_epochs

            ar_index = int(min_interaction_id / 100)

            optimum_model_path = models_path_test_mse_series.index[ar_index]

            print('\noptimum_model_path from series : ', optimum_model_path)
            print()

            ##Save the optimum model
            try:
                self.saver.restore(sess, optimum_model_path)
                print("Got the optimum model")
                try:
                    shutil.rmtree(self.temp_model_folder)
                    print("Deleted : "+self.temp_model_folder)
                except:
                    print("Couldn't delete : "+self.temp_model_folder)

            except:
                print("No optimum models")

            # Save the variables to disk.
            save_path = self.saver.save(sess, self.model_path)
            print("Optimum model saved in file: %s" % save_path)

            return trainMSE_series, testMSE_series, trainMSE_series_norm, testMSE_series_norm, models_path_test_mse_series, self.model_path,optimum_model_epoch

    def test_the_model(self, input_test_batch, output_test_batch):

        with tf.Session() as sess:
            self.init.run()
            try:
                self.saver.restore(sess, self.model_path)
                print("Got the model")
            except:
                print("No models")
            y_pred = sess.run(self.outputs, feed_dict={self.X: input_test_batch})
            mseTest = self.loss.eval(feed_dict={self.X: input_test_batch, self.Y: output_test_batch})
            # print("mseTest:", mseTest)
            # print(y_pred)
            return y_pred, mseTest

    def validate_the_model(self, input_verify_batch, output_verify_batch):

        with tf.Session() as sess:
            self.init.run()
            try:
                self.saver.restore(sess, self.model_path)
                print("Got the model")
            except:
                print("No models")
            y_pred = sess.run(self.outputs, feed_dict={self.X: input_verify_batch})
            mseVerify = self.loss.eval(feed_dict={self.X: input_verify_batch, self.Y: output_verify_batch})
            # print("mseTest:", mseTest)
            # print(y_pred)
            return y_pred, mseVerify

    def predict_the_model(self, input_batch):

        with tf.Session() as sess:
            self.init.run()
            try:
                self.saver.restore(sess, self.model_path)
                print("Got the model")
            except:
                print("No models")
            y_pred = sess.run(self.outputs, feed_dict={self.X: input_batch})
            # print(y_pred)
            return y_pred

    def save_the_given_model(self, given_model_path):

        with tf.Session() as sess:
            self.init.run()
            try:
                self.saver.restore(sess, given_model_path)
                print("Got the model")
            except:
                print("No models")

            # Save the variables to disk.
            save_path = self.saver.save(sess, self.model_path)
            print("Model saved in file: %s" % save_path)

            return save_path


class MyRNNModForPredict:

    def __init__(self, model_path, num_periods, inputs, hidden, output, basicCellType="LSTM"):
        tf.reset_default_graph()
        # self.model_path = model_folder
        self.model_path = model_path
        # self.models_folder = ""
        # self.num_periods = num_periods
        self.inputs = inputs
        self.hidden = hidden
        self.output = output
        # self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float32, [None, num_periods, self.inputs])
        # self.Y = tf.placeholder(tf.float32, [None, num_periods, self.output])

        if basicCellType == "RNN":
            self.basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden, activation=tf.nn.relu)
        else:
            self.basic_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden, state_is_tuple=True)

        self.rnn_output, states = tf.nn.dynamic_rnn(self.basic_cell, self.X, dtype=tf.float32)

        self.stacked_rnn_output = tf.reshape(self.rnn_output, [-1, self.hidden])
        self.stacked_output = tf.layers.dense(self.stacked_rnn_output, self.output)
        self.outputs = tf.reshape(self.stacked_output, [-1, num_periods, self.output])

        # self.loss_square = tf.square(self.outputs - self.Y)
        # self.loss_square = tf.squared_difference(self.outputs, self.Y)
        # self.loss = tf.reduce_mean(self.loss_square)
        # self.loss = tf.sqrt(tf.reduce_mean(self.loss_square))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.training_op = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(max_to_keep=max_no_of_tf_models)

    # def test_the_model(self, input_test_batch):
    #
    #     with tf.Session() as sess:
    #         self.init.run()
    #         try:
    #             self.saver.restore(sess, self.model_path)
    #             print("Got the model")
    #         except:
    #             print("No models")
    #         y_pred = sess.run(self.outputs, feed_dict={self.X: input_test_batch})
    #         return y_pred

    def predict_the_model(self, input_batch):

        with tf.Session() as sess:
            self.init.run()
            try:
                self.saver.restore(sess, self.model_path)
                print("Got the model")
            except:
                print("No models")
            y_pred = sess.run(self.outputs, feed_dict={self.X: input_batch})
            # print(y_pred)
            return y_pred
