
# recognize_speech_music

Machine learning is used to identify the speech and music of the audio file, then the audio file containing the speech and music is split and the music is cut away.

The accuracy of the one-second prediction was 92%.The correct split rate is 147/150=98%.



Usage:

1. Call the test_train function of main.py to train the model

2. Call the test_cut_song function of main.py to split the audio file and cut away the music



Other:

* Training files is in work/test/，chg  directory is the result of manual segmentation, raw directory is the source file before segmentation.

* During the actual training, 18 training files were used, but due to the large size of the files, only 2 training files were uploaded.

* audio file to be splited is  in word/predict
* To learn more,you can visit this [blog](https://www.cnblogs.com/Xjng/p/12560707.html)




# recognize_speech_music项目

使用机器学习方法识别音频文件的演讲和音乐，然后对包含演讲和音乐的音频文件进行分割，把音乐部分切走。
单秒预测的正确率为92%。分割文件的正确率是147/150=98%。

使用方法：
1. 调用main.py的test_train函数训练模型
2. 调用main.py的test_cut_song函数来分割音频文件，把音乐部分切走

其他：
* 训练文件在work/test/，chg是人工分割的结果，raw是分割前的源文件。
* 实际训练时，训练文件用了18个，但是由于文件较大，只上传了2个训练文件。
* 待分割的文件在word/predict
* 更多细节，可以看[博客](https://www.cnblogs.com/Xjng/p/12560707.html)