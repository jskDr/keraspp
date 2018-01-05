import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib
import matplotlib.font_manager as fm
# fm.get_fontconfig_fonts()
font_location = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)


def save_history_history(fname, history_history, fold=''):
    np.save(os.path.join(fold, fname), history_history)


def load_history_history(fname, fold=''):
    history_history = np.load(os.path.join(fold, fname)).item(0)
    return history_history


def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('정확도')
    plt.xlabel('에포크')
    plt.legend(['학습 데이터 성능', '검증 데이터 성능'], loc=0)
    # plt.show()


def plot_loss(history, title=None):
    # summarize history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('손실')
    plt.xlabel('에포크')
    plt.legend(['학습 데이터 성능', '검증 데이터 성능'], loc=0)
    # plt.show()


def plot_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_acc(history)
    plt.subplot(1, 2, 2)
    plot_loss(history)

    
def plot_loss_acc(history):
    plot_loss(history, '(a) 손실 추이')
    plt.show()            
    plot_acc(history, '(b) 정확도 추이')
    plt.show()
    
    
def plot_acc_loss(history):
    plot_acc(history, '(a) 정확도 추이')
    plt.show()
    plot_loss(history, '(b) 손실 추이')
    plt.show()            
    