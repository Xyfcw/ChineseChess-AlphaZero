# policy & value network model
import hashlib
import json
import os
from logging import getLogger

import tensorflow as tf

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, ActionLabelsBlack
import os

logger = getLogger(__name__)

class CChessModel:

    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.n_labels = len(ActionLabelsRed)
        self.graph = None
        self.api = None
    # 生成model
    def build(self):
        # class ModelConfig:
        #     def __init__(self):
        #         self.cnn_filter_num = 256
        #         self.cnn_first_filter_size = 5
        #         self.cnn_filter_size = 3
        #         self.res_layer_num = 7
        #         self.l2_reg = 1e-4
        #         self.value_fc_size = 256
        #         self.distributed = False
        #         self.input_depth = 14
        mc = self.config.model
        # 14 是所有棋子种类（红/黑算不同种类）
        # 整体的输入就是14个棋盘堆叠在一起，每个棋盘表示一种棋子的位置：棋子所在的位置为1，其余位置为0。
        in_x = x = Input((14, 10, 9)) # 14 x 10 x 9

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        # 批量标准化层
        # 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
        # axis：整数，指定当 mode=0 时规范化的轴。为1，意味着对每个特征图进行规范化
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        # 将激活函数应用于输出。
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, 
                    kernel_regularizer=l2(mc.l2_reg), name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(self.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, 
                    kernel_regularizer=l2(mc.l2_reg), name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="cchess_model")
        self.graph = tf.get_default_graph()

    # 构建残差连接
    def _build_residual_block(self, x, index):
        mc = self.config.model
        in_x = x

        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        return None


    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"loading model from {config_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.digest = self.fetch_digest(weight_path)
            self.graph = tf.get_default_graph()
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path):
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

    def get_pipes(self, num=1, api=None, need_reload=True):
        if self.api is None:
            self.api = CChessModelAPI(self.config, self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)

    def close_pipes(self):
        if self.api is not None:
            self.api.close()
            self.api = None

