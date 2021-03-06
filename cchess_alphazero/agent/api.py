# neural networks' prediction api(神经网络的预测API)
from multiprocessing import connection, Pipe
from threading import Thread

import os
import numpy as np
import shutil

from cchess_alphazero.config import Config
from cchess_alphazero.lib.model_helper import load_best_model_weight, need_to_reload_best_model_weight
from cchess_alphazero.lib.web_helper import http_request, download_file
from time import time
from logging import getLogger

logger = getLogger(__name__)

class CChessModelAPI:

    def __init__(self, config: Config, agent_model):  
        self.agent_model = agent_model  # CChessModel
        self.pipes = []     # use for communication between processes/threads（用于进程/线程之间的通信）
        self.config = config
        self.need_reload = True
        self.done = False

    def start(self, need_reload=True):
        self.need_reload = need_reload
        # 创建线程:创建Thread类的实例，并传递一个函数
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self, need_reload=True):
        # multiprocessing.Pipe([duplex])
        # 返回一对 Connection`对象  ``(conn1, conn2)` ， 分别表示管道的两端。
        me, you = Pipe()
        self.pipes.append(me)
        self.need_reload = need_reload
        return you

    def predict_batch_worker(self):
        if self.config.internet.distributed and self.need_reload:
            self.try_reload_model_from_internet()
        # time() : 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
        last_model_check_time = time()
        while not self.done:
            if last_model_check_time + 600 < time() and self.need_reload:
                self.try_reload_model()
                last_model_check_time = time()
            # multiprocessing.connection.wait(object_list, timeout=None):接收多个对象
            # 一直等待直到 object_list 中某个对象处于就绪状态。返回 object_list 中处于就绪状态的对象。如果 timeout 是一个浮点型，该方法会最多阻塞这么多秒。
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes, data_len = [], [], []
            for pipe in ready:
                # poll([timeout])
                # 返回连接对象中是否有可以读取的数据。
                while pipe.poll():
                    try:
                        # recv()
                        # 返回一个由另一端使用 send() 发送的对象。该方法会一直阻塞直到接收到对象。 如果对端关闭了连接或者没有东西可接收，将抛出 EOFError  异常。
                        tmp = pipe.recv()
                    except EOFError as e:
                        logger.error(f"EOF error: {e}")
                        pipe.close()
                    else:
                        data.extend(tmp)
                        data_len.append(len(tmp))
                        result_pipes.append(pipe)
            if not data:
                continue
            # 将data输入转换为数组。
            data = np.asarray(data, dtype=np.float32)
            #
            with self.agent_model.graph.as_default():
                policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
            buf = []
            k, i = 0, 0
            for p, v in zip(policy_ary, value_ary):
                buf.append((p, float(v)))
                k += 1
                if k >= data_len[i]:
                    result_pipes[i].send(buf)
                    buf = []
                    k = 0
                    i += 1

    def try_reload_model(self, config_file=None):
        if config_file:
            config_path = os.path.join(self.config.resource.model_dir, config_file)
            shutil.copy(config_path, self.config.resource.model_best_config_path)
        try:
            if self.config.internet.distributed and not config_file:
                self.try_reload_model_from_internet()
            else:
                if self.need_reload and need_to_reload_best_model_weight(self.agent_model):
                    with self.agent_model.graph.as_default():
                        load_best_model_weight(self.agent_model)
        except Exception as e:
            logger.error(e)

    def try_reload_model_from_internet(self, config_file=None):
        response = http_request(self.config.internet.get_latest_digest)
        if response is None:
            logger.error(f"无法连接到远程服务器！请检查网络连接，并重新打开客户端")
            return
        digest = response['data']['digest']

        if digest != self.agent_model.fetch_digest(self.config.resource.model_best_weight_path):
            logger.info(f"正在下载最新权重，请稍后...")
            if download_file(self.config.internet.download_url, self.config.resource.model_best_weight_path):
                logger.info(f"权重下载完毕！开始训练...")
                try:
                    with self.agent_model.graph.as_default():
                        load_best_model_weight(self.agent_model)
                except ValueError as e:
                    logger.error(f"权重架构不匹配，自动重新加载 {e}")
                    self.try_reload_model(config_file='model_192x10_config.json')
                except Exception as e:
                    logger.error(f"加载权重发生错误：{e}，稍后重新下载")
                    os.remove(self.config.resource.model_best_weight_path)
                    self.try_reload_model_from_internet()
            else:
                logger.error(f"权重下载失败！请检查网络连接，并重新打开客户端")
        else:
            logger.info(f"检查完毕，权重未更新")

    def close(self):
        self.done = True
