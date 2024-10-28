conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
cd metrics/PyTorchEMD
python3 setup.py install
cp build/lib.linux-x86_64-cpython-38/emd_cuda.cpython-38-x86_64-linux-gnu.so .
