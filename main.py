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

# nchannels ����
# sampwidth �������
# framerate ֡�ʣ�Ҳ����һ���ж���֡
# nframes �ļ�һ���ж���֡

WORK_ROOT = r'.\work'  # ����Ŀ¼


def get_index(framerate, min, sec):
    return int(framerate * (min * 60 + sec))


def pre_deal(file_path):
    """��Ƶ������������Ƶ����"""
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # ��ȡ��Ƶ���ַ�����ʽ
    waveData = np.fromstring(strData, dtype=np.int16)  # ���ַ���ת��Ϊint

    waveData = waveData[::nchannels]  # ������������ת��Ϊ������
    rate = 20.00
    framerate = framerate / rate  # ����֡��
    nframes = nframes / rate  # ����֡��
    waveData = waveData[::int(rate)]

    # wave��ֵ��һ��
    max_ = float(max(abs(waveData)))
    waveData = waveData / max_

    return waveData, framerate, nframes


def plpot(waveData):
    """��ͼ"""
    time = [i for i, v in enumerate(waveData)]
    plt.plot(time, waveData)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid('on')  # ��ߣ�on���У�off:�ޡ�
    plt.show()


def mp3towav(file_path, to_file_path):
    """mp3�ļ�תwav�ļ�"""
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
        ת���������ݣ�����ÿ������ļ�����
        �����[0.1,0.1,0.8]ת��Ϊ[2,1]
        2��[0,0.5)����ļ���
        1��[0.5,1)����ļ���
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
        ��ȡ���ڻ���ѧϰ������
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
                continue  # ��Ҫ�ⲿ��

            flag = 0 if i < total_sec_chg else 1
            # print get_index(framerate, 0, i),get_index(framerate, 0, i + length),total_sec
            sample = data_raw[get_index(framerate_raw, 0, i):get_index(framerate_raw, 0, i + length)]

            sample = cls.sample_cnt(sample)

            samples.append((sample, flag))
        return samples

    @classmethod
    def to_wav(cls, file_path):
        """ת��mp3Ϊwav"""
        if 'mp3' in file_path:
            to_file_path = file_path.replace('mp3', 'wav')
            mp3towav(file_path, to_file_path)
            file_path = to_file_path
        return file_path

    @classmethod
    def get_all_sample(cls, ):
        """��ȡ��������"""
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
        """ѵ��"""
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
        ѵ�������
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
            clf.fit(train_datas_sets, train_labels_set)  # ѵ��
            score = clf.score(test_datas_set, test_labels_set)  # Ԥ����Լ�����������ȷ��
            print 'score', mechine, score

    @classmethod
    def train_knn(cls, train_datas_sets, train_labels_set, test_datas_set, test_labels_set):
        from sklearn import neighbors
        mechine = neighbors.KNeighborsClassifier
        clf = mechine()
        clf.fit(train_datas_sets, train_labels_set)  # ѵ��
        score = clf.score(test_datas_set, test_labels_set)  # Ԥ����Լ�����������ȷ��
        print 'score', mechine, score
        pickle_utils.dump(clf, 'knn.model.pkl')

    @classmethod
    def get_cut_sce(cls, file_path, model):
        """��ȡ�ָ���������Ҳ�������None"""
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
        """ת������Ϊ �����ʽ"""
        print '%s:%s' % (int(sec / 60), int(sec % 60))

    @classmethod
    def predict(cls, ):
        """Ԥ��"""
        file_path = r'%s\c.mp3' % WORK_ROOT
        model = pickle_utils.load('knn.model.pkl')
        sec = cls.get_cut_sce(file_path, model)
        print 'sec', sec, cls.get_min(sec)

    @classmethod
    def cut_song(cls, file_path, to_file_path, file_name):
        """�ָ����"""
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
        """�ָ�ĳ���ļ�����������и���"""
        del_path = r'%s\to_del' % WORK_ROOT
        for f in os.listdir(root_path):
            if 'mp3' in f and 'cut' not in f:
                file_path = root_path + '\\' + f
                if os.path.exists(file_path + '.cut.mp3'):
                    print 'exist', file_path.decode('gbk') + '.cut.mp3'
                    continue
                # ����pydub��֧��windows������·��������ֻ�ܰ�Դ�ļ��ѵ�һ����ʱ��Ӣ��Ŀ¼��Ȼ��ִ�зָ� Ȼ�����ʱ�ļ�����
                tmp_file_path = '%s\\test.mp3' % WORK_ROOT  # pydub��֧�����ĵ�ַ��ֻ������
                tmp_wav_path = tmp_file_path.replace('mp3', 'wav')
                tmp_to_file_path = tmp_file_path + '.cut.mp3'
                shutil.copy(file_path, tmp_file_path)
                ret = cls.cut_song(tmp_file_path, tmp_to_file_path, f)
                shutil.move(tmp_file_path, del_path + '\\del1_' + f)
                shutil.move(tmp_wav_path, del_path + '\\del3_' + f)
                try:
                    # �п����Ҳ����ָ�㣬����û�зָ���Լ���try
                    shutil.copy(tmp_to_file_path, file_path + '.cut.mp3')
                    shutil.move(tmp_to_file_path, del_path + '\\del2_' + f)

                except:
                    import traceback
                    print traceback.format_exc()


def test_audio_to_data():
    """���Խű���ת����Ƶ�ļ�Ϊ���ݣ�����ͼ"""
    file_path = r'%s\test\chg\0.mp3' % WORK_ROOT
    file_path = mp3towav(file_path, file_path.replace('mp3', 'wav'))
    data, _, _ = pre_deal(file_path)
    plpot(data)


def test_train():
    """���Խű���ѵ������"""
    LeaningTest.train_wrapper()


def test_cut_song():
    """���Խű����ָ����"""
    LeaningTest.load_model()
    root_path = r'%s\predict' % WORK_ROOT
    LeaningTest.cut_songs(root_path)


if __name__ == '__main__':
    test_audio_to_data()
    # test_train()
    # test_cut_song()
    # pass
