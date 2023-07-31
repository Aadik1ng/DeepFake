{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ad6127f",
   "metadata": {},
   "outputs": [],
   "source": [
    "!python --version"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00314aa9",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "!python -m pip install scikit-build\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9999a747",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3bd89a33",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install ffmpeg-python\n",
    "!pip install librosa==0.9.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "108f024e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: torchvision==0.3.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (0.3.0)\n",
      "Requirement already satisfied: numpy in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from torchvision==0.3.0) (1.17.1)\n",
      "Requirement already satisfied: six in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from torchvision==0.3.0) (1.16.0)\n",
      "Requirement already satisfied: torch>=1.1.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from torchvision==0.3.0) (1.1.0)\n",
      "Requirement already satisfied: pillow>=4.1.1 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from torchvision==0.3.0) (8.4.0)\n"
     ]
    }
   ],
   "source": [
    "!python -m pip install torchvision==0.3.0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "39aa43f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: librosa==0.7.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 1)) (0.7.0)\n",
      "Requirement already satisfied: numpy==1.17.1 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 2)) (1.17.1)\n",
      "Requirement already satisfied: opencv-contrib-python>=4.2.0.34 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 3)) (4.6.0.66)\n",
      "Requirement already satisfied: opencv-python==4.1.0.25 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 4)) (4.1.0.25)\n",
      "Requirement already satisfied: torch==1.1.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 5)) (1.1.0)\n",
      "Requirement already satisfied: torchvision==0.3.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 6)) (0.3.0)\n",
      "Requirement already satisfied: tqdm==4.45.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 7)) (4.45.0)\n",
      "Requirement already satisfied: numba==0.48 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from -r requirements.txt (line 8)) (0.48.0)\n",
      "Requirement already satisfied: joblib>=0.12 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (1.1.1)\n",
      "Requirement already satisfied: audioread>=2.0.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (3.0.0)\n",
      "Requirement already satisfied: soundfile>=0.9.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (0.12.1)\n",
      "Requirement already satisfied: decorator>=3.0.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (5.1.1)\n",
      "Requirement already satisfied: scipy>=1.0.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (1.5.4)\n",
      "Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (0.24.2)\n",
      "Requirement already satisfied: resampy>=0.2.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (0.3.1)\n",
      "Requirement already satisfied: six>=1.3 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from librosa==0.7.0->-r requirements.txt (line 1)) (1.16.0)\n",
      "Requirement already satisfied: pillow>=4.1.1 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from torchvision==0.3.0->-r requirements.txt (line 6)) (8.4.0)\n",
      "Requirement already satisfied: setuptools in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from numba==0.48->-r requirements.txt (line 8)) (59.6.0)\n",
      "Requirement already satisfied: llvmlite<0.32.0,>=0.31.0dev0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from numba==0.48->-r requirements.txt (line 8)) (0.31.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.7.0->-r requirements.txt (line 1)) (3.1.0)\n",
      "Requirement already satisfied: cffi>=1.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from soundfile>=0.9.0->librosa==0.7.0->-r requirements.txt (line 1)) (1.15.1)\n",
      "Requirement already satisfied: pycparser in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from cffi>=1.0->soundfile>=0.9.0->librosa==0.7.0->-r requirements.txt (line 1)) (2.21)\n"
     ]
    }
   ],
   "source": [
    "!python -m pip install -r requirements.txt \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f83ba1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "!python -m pip install opencv-contrib-python>=4.2.0.34\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fca5e520",
   "metadata": {},
   "outputs": [],
   "source": [
    "!python -m pip install tqdm==4.45.0 numba==0.48"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7990bf67",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "!pip install --upgrade librosa\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c4a45e5a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['.git',\n",
       " '.gitignore',\n",
       " '.ipynb_checkpoints',\n",
       " '4.2.0.34',\n",
       " 'audio.py',\n",
       " 'checkpoints',\n",
       " 'color_syncnet_train.py',\n",
       " 'Data',\n",
       " 'evaluation',\n",
       " 'face_detection',\n",
       " 'filelists',\n",
       " 'hparams.py',\n",
       " 'hq_wav2lip_train.py',\n",
       " 'inference.py',\n",
       " 'models',\n",
       " 'preprocess.py',\n",
       " 'README.md',\n",
       " 'requirements.txt',\n",
       " 'results',\n",
       " 'temp',\n",
       " 'Untitled.ipynb',\n",
       " 'Wav2Lip',\n",
       " 'wav2lip_train.py',\n",
       " '__pycache__']"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "os.listdir()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "3d5740d0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"inference.py\", line 7, in <module>\n",
      "    import torch, face_detection\n",
      "  File \"D:\\Wav2Lip\\Wav2Lip\\Wav2Lip\\lib\\site-packages\\torch\\__init__.py\", line 79, in <module>\n",
      "    from torch._C import *\n",
      "ImportError: DLL load failed: The specified procedure could not be found.\n"
     ]
    }
   ],
   "source": [
    "!python inference.py --checkpoint_path=r\"checkpoints\\wav2lip_gan.pth\" --face=r\"Data\\new.mp4\" --audio=r\"Data\\output10.wav\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c40fa2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvcc --version"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce6013e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e72ca9fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "!python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "98382eb9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting torch==1.1.0\n",
      "  Using cached https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-win_amd64.whl (99.6 MB)\n",
      "Requirement already satisfied: numpy in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from torch==1.1.0) (1.17.1)\n"
     ]
    }
   ],
   "source": [
    "!python -m pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-win_amd64.whl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4404ed8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in links: https://download.pytorch.org/whl/torch_stable.html\n",
      "Requirement already satisfied: torch==1.1.0 in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (1.1.0)\n",
      "Requirement already satisfied: numpy in d:\\wav2lip\\wav2lip\\wav2lip\\lib\\site-packages (from torch==1.1.0) (1.17.1)\n"
     ]
    }
   ],
   "source": [
    "!pip install torch==1.1.0 -f https://download.pytorch.org/whl/torch_stable.html\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18492a4e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2rc1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
