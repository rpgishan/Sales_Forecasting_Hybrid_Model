from pymongo import MongoClient
# from MyDataManupulator import MyRNNDataManipulatorModified, MyOtherFunctions
from datetime_periods import period
from datetime import datetime
import random
import pandas as pd


class MyMongoBasicFunctions:
    def __init__(self, database):
        client = MongoClient('localhost', 27017)
        # self.db = client.database
        self.db = client[database]

    def insertone(self, collection, docjson):
        collection = self.db[collection]
        document_inserted = collection.insert_one(docjson)
        return document_inserted

    def updateone(self,collection,doc_id,docjson):
        collection = self.db[collection]
        collection.update_one({'_id': doc_id}, {"$set": docjson}, upsert=True)

    def findall(self, collection):
        self.db.collection_names(include_system_collections=False)
        collection = self.db[collection]
        result = collection.find({})
        return result

    def findfirst(self, collection):
        self.db.collection_names(include_system_collections=False)
        collection = self.db[collection]
        result = collection.find_one()
        return result

    def findone(self, collection, key, value):
        self.db.collection_names(include_system_collections=False)
        collection = self.db[collection]
        result = collection.find_one({key: value})
        return result

    def findbyid(self, collection, doc_id):
        self.db.collection_names(include_system_collections=False)
        collection = self.db[collection]
        result = collection.find_one({'_id': doc_id})
        return result

    # def findallinadaterange(self, collection, fromdate, todate):
    #     self.db.collection_names(include_system_collections=False)
    #     collection = self.db[collection]
    #     result = collection.find({'comments.date': {'$gte': fromdate, '$lt': todate}}).sort('comments.date')
    #     return result

    def findallinadaterange(self, collection,doc_id,key_name, fromdate, todate):
        self.db.collection_names(include_system_collections=False)
        collection = self.db[collection]
        key = str(key_name + '.date')
        result = collection.find({'_id': doc_id, key: {'$gte': fromdate, '$lt': todate}}).sort(key)
        return result


class MyMongoOtherFuctions:

    def __init__(self, myMongoBasicFunctions, collection):
        self.myMongoBasicFunctions = myMongoBasicFunctions
        self.collection = collection

    def getARandomDateTime(self, start_date):
        # start_date = datetime(start_datetime_index.year[0], start_datetime_index.month[0], start_datetime_index.day[0])
        start_time, end_time = period(start_date, 'quarter')
        time_diff = end_time - start_time
        rand_num = random.random()
        random_date = start_time + rand_num * time_diff
        return random_date

    # def insertData(self, comment_number, comment_status, random_date):
    #     comment_datetime = random_date
    #     comment = 'Comment No - ' + str(comment_number)
    #     # comment_status = None
    #     docjson = {self.dbdate: comment_datetime, self.dbreviewtext: comment, self.dbreviewstatus: comment_status}
    #
    #     inserted_doc = self.myMongo.insertone(self.collection, docjson)
    #     # print("inserted_doc")
    #     # print(inserted_doc)
    #     return inserted_doc

    def insertDataJsonAR(self, key_name, docjsonAR, id= int((datetime.now()-datetime.utcfromtimestamp(0)).total_seconds())):
        docjson = {"_id": id, key_name: docjsonAR}
        # docjson = {key_name: docjsonAR}
        inserted_doc = self.myMongoBasicFunctions.insertone(self.collection, docjson)
        #     # print("inserted_doc")
        #     # print(inserted_doc)
        return inserted_doc

    def updateDataJsonAR(self,doc_id, key_name, docjsonAR):
        # docjson = {"_id": doc_id, key_name: docjsonAR}
        docjson = {key_name: docjsonAR}
        updated_doc = self.myMongoBasicFunctions.updateone(self.collection, doc_id, docjson)
        #     # print("inserted_doc")
        #     # print(inserted_doc)
        return updated_doc


# class MyMongoTempWriteFunctions:
#
#     def __init__(self,database_name,collection_name):
#         # self.source_url = "https://gishtestings.azurewebsites.net/uploads/apple--mac-revenue-in-billions.csv"
#         self.source_path = "/Users/Gishan/PycharmProjects/ResearchProject/Datasets/apple--mac-revenue in billions.csv"
#         self.index_col = "Index"
#         self.value_col = "Value"
#         self.time_resolution = "quarterly"
#         self.f_horizon = 1
#         self.random_period = 100
#
#         ##db attrs
#         self.myMongoBasicFunctions = MyMongoBasicFunctions(database_name)
#         self.collection = collection_name
#         self.dbdate = "date"
#         self.dbsales = "values"
#         # self.dbkeyname = "sales_train"
#
#         self.dbreviewtext = "comment"
#         self.dbreviewstatus = "comment_status"
#         # self.dbkeyname = "comments"
#
#         # self.id = 2
#
#     def writeSalesToDb(self,dbkeyname = "sales_train"):
#         myOtherFunctions = MyOtherFunctions()
#         # myRNNDataManipulator = MyRNNDataManipulatorModified()
#         myMongoOtherFuctions = MyMongoOtherFuctions(self.myMongoBasicFunctions, self.collection)
#
#         # data = myOtherFunctions.getDatasetAR(self.source_url, self.index_col, self.value_col, self.time_resolution,
#         #                                      turn_on_data_filler=True)
#
#         data = myOtherFunctions.getDatasetARFromPath(self.source_path, self.index_col, self.value_col, self.time_resolution,
#                                              turn_on_data_filler=True)
#
#         print("data")
#         print(data)
#
#         # rnn_data = myRNNDataManipulator.getDifferentialDatasetWithComments(data, self.f_horizon, self.random_period)
#
#         # for num in range(0,100):
#         #     start_datetime_index = rnn_data.iloc[:1].index
#         #     random_date=getARandomDateTime(start_datetime_index)
#         #     print('random_date - ',random_date)
#
#         docjsonAR = []
#         row_num = 0
#         data_size = len(data)
#
#         for index, row in data.iterrows():
#             # print()
#             print("row_num/size : ", row_num, "/", data_size)
#             row_num += 1
#
#             docjson = {self.dbdate: index, self.dbsales: row[self.value_col]}
#             docjsonAR.append(docjson)
#
#         myMongoOtherFuctions.insertDataJsonAR(dbkeyname, docjsonAR,id=2)
#         print("for end")
#
#
#     def writeCommentsToDb(self,dbkeyname = "comments"):
#         myOtherFunctions = MyOtherFunctions()
#         myRNNDataManipulator = MyRNNDataManipulatorModified()
#         myMongoOtherFuctions = MyMongoOtherFuctions(self.myMongoBasicFunctions, self.collection)
#
#         # data = myOtherFunctions.getDatasetAR(self.source_url, self.index_col, self.value_col, self.time_resolution,
#         #                                      turn_on_data_filler=False)
#
#         data = myOtherFunctions.getDatasetARFromPath(self.source_path, self.index_col, self.value_col, self.time_resolution,
#                                              turn_on_data_filler=True)
#
#
#         rnn_data = myRNNDataManipulator.getDifferentialDatasetWithComments(data, self.f_horizon, self.random_period)
#
#         # for num in range(0,100):
#         #     start_datetime_index = rnn_data.iloc[:1].index
#         #     random_date=getARandomDateTime(start_datetime_index)
#         #     print('random_date - ',random_date)
#
#         # print("rnn_data")
#         # print(rnn_data)
#
#         commentNumber = 1
#         docjsonAR = []
#         row_num = 0
#         rnn_data_size = len(rnn_data)
#
#         for index, row in rnn_data.iterrows():
#             # print()
#             # print("index : ",index)
#             print("row_num/size : ", row_num, "/", rnn_data_size)
#             row_num += 1
#             # print("row Diff- ",row["Diff"])
#             try:
#                 positive = int(row["Positive"])
#                 negative = int(row["Negative"])
#             except ValueError:
#                 continue  # skips to next iteration
#
#             # print("row Positive- ", positive)
#             for pos in range(0, positive):
#                 random_date = myMongoOtherFuctions.getARandomDateTime(index)
#                 comment_status = "Positive"
#                 comment_datetime = random_date
#                 comment = 'Comment No - ' + str(commentNumber)
#                 docjson = {self.dbdate: comment_datetime, self.dbreviewtext: comment,
#                            self.dbreviewstatus: comment_status}
#                 docjsonAR.append(docjson)
#                 commentNumber += 1
#
#             # print()
#             # print("row Negative- ", negative)
#             for neg in range(positive, (negative + positive)):
#                 random_date = myMongoOtherFuctions.getARandomDateTime(index)
#                 comment_status = "Negative"
#                 comment_datetime = random_date
#                 comment = 'Comment No - ' + str(commentNumber)
#                 docjson = {self.dbdate: comment_datetime, self.dbreviewtext: comment,
#                            self.dbreviewstatus: comment_status}
#                 docjsonAR.append(docjson)
#                 commentNumber += 1
#
#         myMongoOtherFuctions.insertDataJsonAR(dbkeyname, docjsonAR)
#         print("for end")
#

class MyMongoDataFunctions:

    def __init__(self,database_name,collection_name):
        ##db attrs
        self.myMongoBasicFunctions = MyMongoBasicFunctions(database_name)
        self.collection = collection_name
        self.dbdate = "date"
        self.dbsales = "values"
        # self.id = 1

        self.index_col_new = "date"
        self.value_col_new = "sales"
        self.dbreviewtext = "comment"
        self.dbreviewstatus = "comment_status"
        self.dbkeyname = "comments"



    def writeSalesToDb(self, doc_id, data,value_col, dbkeyname = "sales_train"):
        myMongoOtherFuctions = MyMongoOtherFuctions(self.myMongoBasicFunctions, self.collection)

        docjsonAR = []
        row_num = 0
        # data_size = len(data)

        for index, row in data.iterrows():
            row_num += 1
            docjson = {self.dbdate: index, self.dbsales: row[value_col]}
            docjsonAR.append(docjson)
        myMongoOtherFuctions.updateDataJsonAR(doc_id, dbkeyname, docjsonAR)
        print("db write successful")

    def writeRNNDataToDB(self, doc_id,number_of_inputs,optimum_model_path):
        myMongoOtherFuctions = MyMongoOtherFuctions(self.myMongoBasicFunctions, self.collection)
        key = "optimum model path with " + str(number_of_inputs) + " inputs"
        docjson = {key: optimum_model_path}

        dbkeyname = "rnn_data"
        myMongoOtherFuctions.updateDataJsonAR(doc_id, dbkeyname, docjson)

    # def readSalesFromDB(self, doc_id, dbkeyname = "sales_train"):
    #
    #     index_data_ar = []
    #     sales_data_ar = []
    #     results = self.myMongoBasicFunctions.findbyid(self.collection, doc_id)
    #     for result in results[dbkeyname]:
    #         index_data_ar.append(result[self.dbdate])
    #         sales_data_ar.append(result[self.dbsales])
    #
    #     data = pd.Series(sales_data_ar,index_data_ar)
    #     return data

    def readSalesFromDBWithColName(self, doc_id, col_name, dbkeyname = "sales_train"):

        index_data_ar = []
        sales_data_ar = []

        # results = self.myMongoBasicFunctions.findall(self.collection)
        results = self.myMongoBasicFunctions.findbyid(self.collection, doc_id)
        for result in results[dbkeyname]:
            index_data_ar.append(result[self.dbdate])
            sales_data_ar.append(result[self.dbsales])

        data = pd.DataFrame({col_name:sales_data_ar},index=index_data_ar)
        return data

    def readSalesAvailableDatesFromDBWithColName(self, doc_id,dbkeyname = "sales_train"):

        index_data_ar = []

        results = self.myMongoBasicFunctions.findbyid(self.collection, doc_id)
        for result in results[dbkeyname]:
            index_data_ar.append(result[self.dbdate])

        data_date_indices = pd.DataFrame(index_data_ar)
        return data_date_indices

    # def readCommentsFromDBOld(self):
    #     dbkeyname = "comments"
    #
    #     rnn_data = self.readSalesFromDB()
    #
    #     positive_comments_ar = []
    #     negative_comments_ar = []
    #
    #
    #     rnn_data_indices = rnn_data.index
    #     rnn_data_indices_size = len(rnn_data_indices)
    #
    #     row_num = 0
    #
    #     for index_id in range(0,rnn_data_indices_size):
    #         print("row_num/size : ", row_num, "/", rnn_data_indices_size)
    #         row_num += 1
    #         start_time = rnn_data_indices[index_id]
    #         if index_id < rnn_data_indices_size-1:
    #             end_time = rnn_data_indices[index_id+1]
    #         else:
    #             start_time, end_time = period(end_time, "day")
    #
    #         results = self.myMongoBasicFunctions.findallinadaterange(self.collection, dbkeyname, start_time, end_time)
    #         # print('results')
    #         # print(results)
    #         # print()
    #         positive = 0
    #         negative = 0
    #         unknown = 0
    #         for resultList in results:
    #             # print('resultList')
    #             # print(len(resultList['comments']))
    #             # print()
    #             if dbkeyname not in resultList:
    #                 continue
    #             for result in resultList[dbkeyname]:
    #                 # print('result')
    #                 # print(len(result))
    #                 # print(result['comment_status'])
    #                 # print()
    #                 comment_status = result[self.dbreviewstatus]
    #                 if comment_status == 'Positive':
    #                     positive += 1
    #                 elif comment_status == 'Negative':
    #                     negative += 1
    #                 else:
    #                     unknown += 1
    #         # print('Positive - ',str(positive),'\t\tNegative - ',str(negative),'\t\tUnknown - ',str(unknown))
    #         positive_comments_ar.append(positive)
    #         negative_comments_ar.append(negative)
    #
    #
    #     rnn_data["Positive"] = positive_comments_ar
    #     rnn_data["Negative"] = negative_comments_ar
    #     rnn_data_positive = pd.Series(positive_comments_ar,rnn_data_indices)
    #     rnn_data_negative = pd.Series(negative_comments_ar,rnn_data_indices)
    #
    #     return rnn_data_positive,rnn_data_negative

## have to modify

    #have to modify
    def readCommentsFromDB(self, doc_id,value_col):
        dbkeyname = "reviews"
        dbkeynamesales = "sales_train"

        rnn_data = self.readSalesFromDBWithColName(doc_id,value_col,dbkeynamesales)

        positive_comments_ar = []
        negative_comments_ar = []


        rnn_data_indices = rnn_data.index
        rnn_data_indices_size = len(rnn_data_indices)

        row_num = 0
        start_time = rnn_data_indices[0]
        _, end_time = period(rnn_data_indices[rnn_data_indices_size-1], "day")
        results = self.myMongoBasicFunctions.findallinadaterange(self.collection, doc_id, dbkeyname, start_time, end_time)

        # for resultList in results:
        #     print('resultList')
        #     print(len(resultList['comments']))
        #     print()
        #     if dbkeyname not in resultList:
        #         continue
        #     for result in resultList[dbkeyname]:
        for result in results[dbkeyname]:
            print('result')
            print(len(result))
            # print(result.keys())
            print(result['date']," - ",result['comment_status'])
            print()
            # print('Positive - ',str(positive),'\t\tNegative - ',str(negative),'\t\tUnknown - ',str(unknown))
            # positive_comments_ar.append(positive)
            # negative_comments_ar.append(negative)


        # rnn_data["Positive"] = positive_comments_ar
        # rnn_data["Negative"] = negative_comments_ar
        rnn_data_positive = pd.Series()#(positive_comments_ar,rnn_data_indices)
        rnn_data_negative = pd.Series()#(negative_comments_ar,rnn_data_indices)

        return rnn_data_positive,rnn_data_negative

if __name__ == "__main__":
    database_name = "Research_Project_DB"
    collection_name = "Reviews"

    # myMongoTempWriteFunctions = MyMongoTempWriteFunctions(database_name,collection_name)
    # myMongoTempWriteFunctions.writeSalesToDb(dbkeyname = "sales_train")
    # myMongoTempWriteFunctions.writeCommentsToDb(dbkeyname = "comments")


    # myMongoDataFunctions = MyMongoDataFunctions(database_name,collection_name)
    # rnn_data = myMongoDataFunctions.readSalesFromDB()
    # print("rnn_data")
    # print(rnn_data)

    # rnn_data_positive, rnn_data_negative = myMongoDataFunctions.readCommentsFromDB()
    # print("rnn_data_negative")
    # print(rnn_data_negative)
    # print()
    # print()
    # print("rnn_data_positive")
    # print(rnn_data_positive)

    # value_col = "Mac"
    # myMongoDataFunctions = MyMongoDataFunctions(database_name, collection_name)
    # # data = myMongoDataFunctions.readSalesFromDBWithColName(value_col, dbkeyname="sales_train")
    # data_date_indices = myMongoDataFunctions.readSalesAvailableDatesFromDBWithColName(dbkeyname="sales_train")
    # with pd.option_context('display.max_rows', 20, 'display.max_columns', 3):
    #     print(data_date_indices)
