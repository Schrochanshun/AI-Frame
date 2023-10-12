### AI-Frame使用指南
#### 介绍
* 本应用是在Windows平台上搭建,基于机器学习通过插帧和抽帧实现视频帧率转换
#### 安装
* 提前安装Python,为了保证后续第三方库的顺利安装,可能需要切换镜像源
* 下载该软件安装包后,运行install.cmd该脚本文件,会自动安装该软件所需的第三方库
* 等待第三方库安装完成后,到达https://ffmpeg.org/download.html#build-windows 安装ffmpeg库, 安装完成后将该安装包中的bin文件添加至系统环境变量后重启生效
* 启动运行AI-Frame.exe,将对应mp4文件拖入上方方框中,选定帧率或自定义帧率点击下方启动按钮进行转化
* 因为numpy的后续更新导致对np.int和np.float不再支持,弹出命令行窗口可能报错AttributeError,则需要将报错的py文件中的np.int改为np.int64,np.float改为np.float64
* 完成上述步骤,重新点击打开AI-Frame.exe即可顺利运行