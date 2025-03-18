install_apps() {
    sudo apt-get install gcc g++ cmake

    # 先安装这个
    sudo apt-get install bzip2 libbz2-dev libsqlite3-dev libreadline-dev libncursesw5-dev
    pyenv uninstall 3.9.18
    pyenv install 3.9.18
    pyenv global 3.9.18  # Or whi
    python -c "import bz2; print(bz2.BZ2Compressor())"

    # @title Download pretrained models 下载预训练模型
    rootDir=~
    mkdir -p $rootDir/GPT-SoVITS/GPT_SoVITS/pretrained_models
    mkdir -p $rootDir/GPT-SoVITS/tools/asr/models
    mkdir -p $rootDir/GPT-SoVITS/tools/uvr5

    cd $rootDir/GPT-SoVITS/GPT_SoVITS/pretrained_models
    # 可能需要梯子，访问不了
    git clone https://huggingface.co/lj1995/GPT-SoVITS

    cd $rootDir/GPT-SoVITS/tools/asr/models
    git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
    git clone https://www.modelscope.cn/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
    git clone https://www.modelscope.cn/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git

    # # @title UVR5 pretrains 安装uvr5模型
    cd $rootDir/GPT-SoVITS/tools/uvr5
    git clone https://huggingface.co/Delik/uvr5_weights
    git config core.sparseCheckout true
    mv $rootDir/GPT-SoVITS/GPT_SoVITS/pretrained_models/GPT-SoVITS/* $rootDir/GPT-SoVITS/GPT_SoVITS/pretrained_models/
}

prepare_dirs() {
    # output_dir = '/data/sovits/output'  # uuid.wav
    # input_dir = '/data/sovits/input'    # voice_id.wav
    # log_dir = '/data/sovits/logs'

    mkdir -p /data/sovits/logs /data/sovits/output /data/sovits/input
}