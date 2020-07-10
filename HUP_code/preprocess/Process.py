# coding=utf-8
from Preprocess import *
from ProcessTrain import *
from Config import *
import os


# from PlotRes.Plot import *

class Process():
    def __init__(self):
        pass

    @classmethod
    def build_base_data(cls):
        Preprocess.build_raw_data()
        Preprocess.get_embedding_items()
        Preprocess.get_data_item_mapping_items()
        Preprocess.get_recall_topsku()  # embedding最相近的1000个物品
        pass

    @classmethod
    def build_train_data(cls):
        # 得到文件session.SBCGD,里面存所有reindex以后的数据
        ProcessTrain.get_data_session_train_itemid(Config.folder, Config.file_data_src, "session", mode="SBCGD")
        # 得到文件session.SBCGD.id, session.SBCGD.id.mapping把一整个物品(包括SBCGD)映射成一个id
        Preprocess.get_file_to_id_mapping(Config.folder, "session.SBCGD", "session.SBCGD.id",
                                          "session.SBCGD.id.mapping")
        # 分train,test
        ProcessTrain.split_data_train_test(Config.folder, 'session.SBCGD.id', "session.SBCGD.id.mapping", -1,
                                           Config.train_ratio)
        # 截断train,test
        ProcessTrain.format_data_train_test(Config.folder, 'session.SBCGD.id', "session.SBCGD.id.mapping",
                                            Config.seq_len)
        ProcessTrain.get_file_micro_items_sequence(Config.folder, "session.SBCGD.id", "session.SBCGD.id.mapping",
                                                   Config.seq_len)
        ProcessTrain.get_file_micro_items_sequence_train_data(Config.folder, "session.SBCGD.id")
        pass

    @classmethod
    def analyse_data(cls):
        FileTool.func_begin("analyse_data")
        Preprocess.get_data_statis()
        # ProcessTrain.analyse_file_data_sbcd()
        FileTool.func_end("analyse_data")
        pass

    @classmethod
    def start(cls):
        Process.build_base_data()
        Process.build_train_data()
        # Process.analyse_data()
        pass


if __name__ == "__main__":
    Process.start()
