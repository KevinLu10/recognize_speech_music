# encoding=gbk
import random
import wave
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import json
from pydub import AudioSegment
import pickle
import pickle_utils

# encoding=gbk
import random
import wave
import matplotlib.pyplot as plt
import numpy as np
import os

# nchannels 声道
# sampwidth 样本宽度
# framerate 帧率，也就是一秒有多少帧
# nframes 文件一共有多少帧

WORK_ROOT = r'.\work'  # 工作目录


def get_index(framerate, min, sec):
    return int(framerate * (min * 60 + sec))


def pre_deal(file_path):
    """音频解析，返回音频数据"""
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int

    waveData = waveData[::nchannels]  # 根据声道数，转换为单声道
    rate = 20.00
    framerate = framerate / rate  # 降低帧率
    nframes = nframes / rate  # 降低帧率
    waveData = waveData[::int(rate)]

    # wave幅值归一化
    max_ = float(max(abs(waveData)))
    waveData = waveData / max_

    return waveData, framerate, nframes


def plpot(waveData):
    """画图"""
    time = [i for i, v in enumerate(waveData)]
    plt.plot(time, waveData)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid('on')  # 标尺，on：有，off:无。
    plt.show()


def mp3towav(file_path, to_file_path):
    """mp3文件转wav文件"""
    if os.path.exists(to_file_path):
        return to_file_path
    from pydub import AudioSegment
    print file_path
    song1 = AudioSegment.from_mp3(file_path)
    song1.export(to_file_path, 'wav')
    return to_file_path


class LeaningTest():
    chg_path = r'%s\test\chg' % WORK_ROOT
    raw_path = r'%s\test\raw' % WORK_ROOT
    model = None

    @classmethod
    def load_model(cls):
        cls.model = pickle_utils.load('knn.model.pkl')

    @classmethod
    def get_path(cls, i, t):
        p = cls.chg_path if t == 'chg' else cls.raw_path
        return p + '\\' + '%s.mp3' % i

    @classmethod
    def sample_cnt(cls, sample):
        """
        转换样本数据，返回每个区间的计数。
        例如从[0.1,0.1,0.8]转换为[2,1]
        2是[0,0.5)区间的计数
        1是[0.5,1)区间的计数
        """
        step = 0.025
        qujians = []
        start = 0
        while start < 1:
            qujians.append((start, start + step))
            start += step
        new_sample = [0 for i in range(len(qujians))]
        for s in sample:
            for i, qujian in enumerate(qujians):
                if qujian[0] <= s < qujian[1]:
                    new_sample[i] += 1
        return new_sample

    @classmethod
    def get_sample(cls, i):
        """
        获取用于机器学习的数据
        return [([100,200],0)]
        """
        chg = cls.to_wav(cls.get_path(i, 'chg'))
        raw = cls.to_wav(cls.get_path(i, 'raw'))

        data_chg, framerate_chg, n_frames_chg = pre_deal(chg)
        total_sec_chg = int(n_frames_chg / framerate_chg)

        data_raw, framerate_raw, n_frames_raw = pre_deal(raw)
        total_sec_raw = int(n_frames_raw / framerate_raw)

        length = 1
        samples = []
        for i in range(60, total_sec_raw, length):
            if total_sec_chg + 5 < i < total_sec_chg + 5:
                continue  # 不要这部分

            flag = 0 if i < total_sec_chg else 1
            # print get_index(framerate, 0, i),get_index(framerate, 0, i + length),total_sec
            sample = data_raw[get_index(framerate_raw, 0, i):get_index(framerate_raw, 0, i + length)]

            sample = cls.sample_cnt(sample)

            samples.append((sample, flag))
        return samples

    @classmethod
    def to_wav(cls, file_path):
        """转换mp3为wav"""
        if 'mp3' in file_path:
            to_file_path = file_path.replace('mp3', 'wav')
            mp3towav(file_path, to_file_path)
            file_path = to_file_path
        return file_path

    @classmethod
    def get_all_sample(cls, ):
        """获取所有样本"""
        file_name = 'sample4.json'
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                return json.loads(f.read())
        else:
            samples = []
            for i in range(1):
                print 'get sample', i
                samples.extend(cls.get_sample(i))
            with open(file_name, 'w') as f:
                f.write(json.dumps(samples))
            return samples

    @classmethod
    def train_wrapper(cls):
        """训练"""
        samples = cls.get_all_sample()
        label0 = [s for s in samples if s[1] == 0]
        label1 = [s for s in samples if s[1] == 1]
        random.shuffle(label0)
        random.shuffle(label1)
        train_datas_sets = [i[0] for i in label0[:int(len(label0) * 0.7)]] + [i[0] for i in
                                                                              label1[:int(len(label1) * 0.7)]]
        train_labels_set = [i[1] for i in label0[:int(len(label0) * 0.7)]] + [i[1] for i in
                                                                              label1[:int(len(label1) * 0.7)]]
        test_datas_set = [i[0] for i in label0[int(len(label0) * 0.7):]] + [i[0] for i in
                                                                            label1[int(len(label1) * 0.7):]]
        test_labels_set = [i[1] for i in label0[int(len(label0) * 0.7):]] + [i[1] for i in
                                                                             label1[int(len(label1) * 0.7):]]
        # print len(train_datas_sets)
        # cls.train(train_datas_sets, train_labels_set, test_datas_set, test_labels_set)
        cls.train_knn(train_datas_sets, train_labels_set, test_datas_set, test_labels_set)

    @classmethod
    def train(cls, train_datas_sets, train_labels_set, test_datas_set, test_labels_set):
        """
        训练结果：
        score <class 'sklearn.svm.classes.SVC'> 0.7203252032520325
        score <class 'sklearn.linear_model.logistic.LogisticRegression'> 0.8886178861788618
        score <class 'sklearn.linear_model.base.LinearRegression'> 0.40864632529611417
        score <class 'sklearn.tree.tree.DecisionTreeClassifier'> 0.8888888888888888
        score <class 'sklearn.neighbors.classification.KNeighborsClassifier'> 0.9224932249322493
        score <class 'sklearn.neural_network.multilayer_perceptron.MLPClassifier'> 0.835230352303523
        score <class 'sklearn.naive_bayes.GaussianNB'> 0.8035230352303523

        """
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import LinearRegression
        from sklearn import tree
        from sklearn import svm
        from sklearn.neural_network import MLPClassifier
        from sklearn import neighbors
        for mechine in [svm.SVC, LogisticRegression, LinearRegression, tree.DecisionTreeClassifier,
                        neighbors.KNeighborsClassifier, MLPClassifier, GaussianNB]:
            clf = mechine()
            clf.fit(train_datas_sets, train_labels_set)  # 训练
            score = clf.score(test_datas_set, test_labels_set)  # 预测测试集，并计算正确率
            print 'score', mechine, score

    @classmethod
    def train_knn(cls, train_datas_sets, train_labels_set, test_datas_set, test_labels_set):
        from sklearn import neighbors
        mechine = neighbors.KNeighborsClassifier
        clf = mechine()
        clf.fit(train_datas_sets, train_labels_set)  # 训练
        score = clf.score(test_datas_set, test_labels_set)  # 预测测试集，并计算正确率
        print 'score', mechine, score
        pickle_utils.dump(clf, 'knn.model.pkl')

    @classmethod
    def get_cut_sce(cls, file_path, model):
        """获取分割的秒数，找不到返回None"""
        file_path = cls.to_wav(file_path)
        data_raw, framerate, n_frames = pre_deal(file_path)
        total_sec = int(n_frames / framerate)

        length = 1
        rets = []
        for i in range(60, total_sec, length):
            # print file_path, i
            sample = data_raw[get_index(framerate, 0, i):get_index(framerate, 0, i + length)]

            sample = cls.sample_cnt(sample)
            ret = model.predict([sample])
            rets.append(ret)
            if ret == 1 and len(rets) >= 3 and rets[-2] == 1 and rets[-3] == 1:
                return i

        return None

    @classmethod
    def get_min(cls, sec):
        """转换秒数为 分秒格式"""
        print '%s:%s' % (int(sec / 60), int(sec % 60))

    @classmethod
    def predict(cls, ):
        """预测"""
        file_path = r'%s\c.mp3' % WORK_ROOT
        model = pickle_utils.load('knn.model.pkl')
        sec = cls.get_cut_sce(file_path, model)
        print 'sec', sec, cls.get_min(sec)

    @classmethod
    def cut_song(cls, file_path, to_file_path, file_name):
        """分割歌曲"""
        print 'cut_song', file_name.decode('gbk'), file_path
        sec = cls.get_cut_sce(file_path, cls.model)
        if sec is None:
            print 'error can not find sec', file_path, file_name.decode('gbk')
            return 0
        song = AudioSegment.from_mp3(file_path)
        # to_file_path=file_path.replace('mp3','wav')
        song = song[:sec * 1000]
        song.export(to_file_path, 'mp3', bitrate='64k')
        return 1

    @classmethod
    def cut_songs(cls, root_path):
        """分割某个文件夹下面的所有歌曲"""
        del_path = r'%s\to_del' % WORK_ROOT
        for f in os.listdir(root_path):
            if 'mp3' in f and 'cut' not in f:
                file_path = root_path + '\\' + f
                if os.path.exists(file_path + '.cut.mp3'):
                    print 'exist', file_path.decode('gbk') + '.cut.mp3'
                    continue
                # 由于pydub不支持windows的中文路径，所以只能把源文件已到一个临时的英文目录，然后执行分割 然后把临时文件移走
                tmp_file_path = '%s\\test.mp3' % WORK_ROOT  # pydub不支持中文地址，只能这样
                tmp_wav_path = tmp_file_path.replace('mp3', 'wav')
                tmp_to_file_path = tmp_file_path + '.cut.mp3'
                shutil.copy(file_path, tmp_file_path)
                ret = cls.cut_song(tmp_file_path, tmp_to_file_path, f)
                shutil.move(tmp_file_path, del_path + '\\del1_' + f)
                shutil.move(tmp_wav_path, del_path + '\\del3_' + f)
                try:
                    # 有可能找不到分割点，导致没有分割，所以加上try
                    shutil.copy(tmp_to_file_path, file_path + '.cut.mp3')
                    shutil.move(tmp_to_file_path, del_path + '\\del2_' + f)

                except:
                    import traceback
                    print traceback.format_exc()


def test_audio_to_data():
    """测试脚本，转换音频文件为数据，并画图"""
    file_path = r'%s\test\chg\0.mp3' % WORK_ROOT
    file_path = mp3towav(file_path, file_path.replace('mp3', 'wav'))
    data, _, _ = pre_deal(file_path)
    plpot(data)


def test_train():
    """测试脚本，训练数据"""
    LeaningTest.train_wrapper()


def test_cut_song():
    """测试脚本，分割歌曲"""
    LeaningTest.load_model()
    root_path = r'%s\predict' % WORK_ROOT
    LeaningTest.cut_songs(root_path)


if __name__ == '__main__':
    test_audio_to_data()
    # test_train()
    # test_cut_song()
    # pass
