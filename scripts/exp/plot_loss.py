# -*- coding: utf-8 -*-
from __future__ import print_function
from matplotlib import rcParams
FONT_FAMILY='DejaVu Serif'
rcParams["font.family"] = FONT_FAMILY 
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
import utils as u
import os
from scipy.signal import savgol_filter
SMOOTH_CURVE=False
USE_MEAN=True

markers=['.','x','o','v', '+','d','s','h', 'P','p','*']
#markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)
fixed_colors = {
        'S-SGD': '#ff3300',
        'ssgd': '#ff3300',
        'gTopK': '#009900',
        'blue': 'b',
        1.0: 'r',
        0.01: 'C1',
        0.001: 'C2',
        0.002: 'C5',
        0.00025: 'C3',
        0.0001: 'C0',
        0.00005: 'C1',
        0.00001: 'C4',
        'none': 'r',
        'eftopk': 'blue',
        'eftopk-womc': 'yellow',
        'eftopkdecay': 'green',
        'eftopkdecay-womc': 'magenta',
        'eftopkdd': 'black',
        'eftopkdd-womc': 'purple',
        'signum': 'C0',
        'efsignum': 'C1',
        'efsignumdecay': 'C4',
        }

OUTPUTPATH='/tmp/ijcai2019'
LOGHOME='/tmp/logs'

FONTSIZE=14
HOSTNAME='localhost'
num_batches_per_epoch = None
global_max_epochs=150
global_density=0.001
NFIGURES=1;NFPERROW=1
PLOT_NORM=False
if PLOT_NORM:
    #FIGSIZE=(5*NFPERROW,3.1*NFIGURES/NFPERROW)
    FIGSIZE=(5*NFPERROW,3.2*NFIGURES/NFPERROW)
else:
    #FIGSIZE=(5*NFPERROW,2.9*NFIGURES/NFPERROW)
    FIGSIZE=(2*5.2*NFPERROW,2*4.2*NFIGURES/NFPERROW)

fig, group_axs = plt.subplots(NFIGURES/NFPERROW, NFPERROW,figsize=FIGSIZE)
if NFIGURES > 1 and PLOT_NORM:
    ax = None
    group_axtwins = []
    for i in range(NFIGURES/NFPERROW):
        tmp = []
        for a in group_axs[i]:
            tmp.append(a.twinx())
        group_axtwins.append(tmp)
    global_index = 0
else:
    ax = group_axs
    ax1 = ax
    global_index = None
ax2 = None

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG-16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
        'lstmptb': 'LSTM-PTB',
        'lstm': 'LSTM-PTB',
        'lstman4': 'LSTM-AN4'
        }

def get_real_title(title):
    return STANDARD_TITLES.get(title, title)

def seconds_between_datetimestring(a, b):
    a = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    b = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    delta = b - a 
    return delta.days*86400+delta.seconds
sbd = seconds_between_datetimestring

def get_loss(line, isacc=False):
    valid = line.find('val acc: ') > 0 or line.find('val top-1 acc') > 0 if isacc else line.find('avg loss: ') > 0
    if line.find('Epoch') > 0 and valid: 
        items = line.split(' ')
        loss = float(items[-1])
        epoch = int(line.split('INFO Epoch')[1].split(',')[0])
        if line.find('top-5 acc') > 0:
            loss = float(line.split('val top-1 acc:')[-1].split(',')[0]) #*0.01
        t = line.split(' I')[0].split(',')[0]
        t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        return loss, t, epoch
    return None, None, None

def read_losses_from_log(logfile, isacc=False):
    global num_batches_per_epoch
    f = open(logfile)
    loss_dict = {}
    losses = []
    times = []
    average_delays = []
    lrs = []
    i = 0
    time0 = None 
    max_epochs = global_max_epochs
    print('max_epochs: ', max_epochs)
    counter = 0
    for line in f.readlines():
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        valid = line.find('val acc: ') > 0 or line.find('val top-1 acc') > 0 if isacc else line.find('average loss: ') > 0
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        if line.find('Epoch') > 0 and valid:
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            if not time0:
                time0 = t
        if line.find('lr: ') > 0:
            try:
                lr = float(line.split(',')[-2].split('lr: ')[-1])
                lrs.append(lr)
            except:
                pass
        if line.find('average delay: ') > 0:
            delay = int(line.split(':')[-1])
            average_delays.append(delay)
        loss, t, epoch = get_loss(line, isacc)
        if loss is None:
            continue
        if epoch not in loss_dict:
            loss_dict[epoch] = [] 
        loss_dict[epoch].append(loss)
        if loss and t:
            counter += 1
            losses.append(loss)
            times.append(t)
        #if counter > max_epochs:
        #    break
    #print(loss_dict)
    f.close()
    if USE_MEAN:
        max_epoch = max(loss_dict.keys())
        losses = [np.mean(loss_dict.get(i, [0])) for i in range(1, max_epoch+1)] 
        stds = [np.std(loss_dict.get(i, [0])) for i in range(1, max_epoch+1)] 
        run_num = len(loss_dict[max_epoch-1])
        #print('max_epoch: ', max_epoch, ', run_num: ', run_num)
        all_max = []
        for j in range(run_num):
            current_run = [loss_dict.get(i, [0]*run_num)[j] for i in range(1, max_epoch)]
            all_max.append(np.max(current_run))
    else:
        stds = [0] * len(losses)
        all_max = [np.max(losses)]
    losses = losses[:max_epochs]
    stds = stds[:max_epochs]
    if len(times) > 0:
        t0 = time0 if time0 else times[0] #times[0]
        for i in range(0, len(times)):
            delta = times[i]- t0
            times[i] = delta.days*86400+delta.seconds
        times = times[:max_epochs]
    if isacc and not logfile.find('lstm') > 0:
        if losses[-1] > 1.0:
            losses = [l * 0.01 for l in losses] 
    return losses, stds, all_max, times, average_delays, lrs

def read_norm_from_log(logfile):
    f = open(logfile)
    means = []
    stds = []
    for line in f.readlines():
        if line.find('gtopk-dense norm mean') > 0:
            items = line.split(',')
            mean = float(items[-2].split(':')[-1])
            std = float(items[--1].split(':')[-1])
            means.append(mean)
            stds.append(std)
    #print('means: ', means)
    #print('stds: ', stds)
    return means, stds


def plot_sub_boxes(ax, name=None):

    #Plot subboxes
    bbox_to_anchor = (-0.02, -0.08, 1, 0.95)
    if name == 'lstm':
        bbox_to_anchor = (-0.02, -0.28, 1, 0.95)
    subaxes = inset_axes(ax,
        width='50%', 
        height='30%', 
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        loc='upper right')
    plts = ax.lines
    #subaxes.set_ylim(bottom=0.85, top=0.95)
    for line in plts:
        x = line.get_xdata()
        y = line.get_ydata()
        half = len(x)*3//4
        subx = x[half:]
        suby = y[half:]
        subaxes.plot(subx, suby, color=line.get_color(), marker=line.get_marker(), markerfacecolor='none', linewidth=1.5)
    subaxes.set_ylim(bottom=subaxes.get_ylim()[0])


def plot_loss(logfile, label, isacc=False, title='ResNet-20', fixed_color=None):
    if not os.path.isfile(logfile) and logfile.find('scigpu12-0.log') > 0:
        logfile = logfile.replace('scigpu12', 'scigpu13')
    losses, stds, all_max, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    norm_means, norm_stds = read_norm_from_log(logfile)

    #print('times: ', times)
    #print('losses: ', losses)
    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0
    if delay > 0:
        label = label + ' (delay=%d)' % delay
    if isacc:
        ax.set_ylabel('top-1 Validation Accuracy')
    else:
        ax.set_ylabel('training loss')
    ax.set_title(get_real_title(title))
    marker = markeriter.next()
    if fixed_color:
        color = fixed_color
    else:
        color = coloriter.next()

    iterations = np.arange(len(losses)) 
    if SMOOTH_CURVE:
        losses = savgol_filter(losses, 5, 3)
    max_acc = np.mean(all_max) #np.max(losses) if isacc and logfile.find('lstm') < 0 else np.min(losses)
    std = np.std(all_max) #stds[np.argmax(losses)] if isacc and logfile.find('lstm') < 0 else stds[np.argmin(losses)]
    print('Algo: %s mean acc of %d runs: %f +- %f\n' % (label, len(all_max), max_acc, std))
    line = ax.plot(iterations, losses, label=label, marker=marker, markerfacecolor='none', color=color, linewidth=1)
    #line = ax.errorbar(iterations, losses, yerr=stds, label=label, marker=marker, markerfacecolor='none', color=color, linewidth=1)
    ax.fill_between(iterations, np.array(losses) - np.array(stds), np.array(losses)+ np.array(stds), alpha=0.2)
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    ax.set_xlabel('# of epochs')
    if len(lrs) > 0:
        lr_indexes = [0]
        lr = lrs[0]
        for i in range(len(lrs)):
            clr = lrs[i]
            if lr != clr:
                lr_indexes.append(i)
                lr = clr
    u.update_fontsize(ax, FONTSIZE)
    return line, max_acc, std


def plot_with_params(dnn, nworkers, bs, lr, hostname, legend, isacc=False, prefix='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False, force_color=None):
    global global_density
    global_density = density
    postfix='5922'
    color = None
    if prefix.find('allreduce')>=0:
        postfix='0'
    elif prefix.find('single') >= 0:
        postfix = None
    if sparsity:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (prefix, dnn, nworkers, bs, lr, sparsity)
    elif nsupdate:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, dnn, nworkers, bs, lr, nsupdate)
    else:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f' % (prefix, dnn, nworkers, bs, lr)
    if sg is not None:
        logfile += '-sg%.2f' % sg
    if density is not None:
        logfile += '-ds%s' % str(density)
        color = fixed_colors[density]
    else:
        color = fixed_colors['S-SGD']
    if force_color:
        color = force_color
    if postfix is None:
        logfile += '/%s.log' % (hostname)
    else:
        logfile += '/%s-%s.log' % (hostname, postfix)
    print('logfile: ', logfile)
    if force_legend:
        l = legend
    else:
        l = legend+ '(lr=%.4f, bs=%d, %d workers)'%(lr, bs, nworkers)
    line, max_acc, std = plot_loss(logfile, l, isacc=isacc, title=dnn, fixed_color=color) 
    return line, max_acc, std

def plot_convergence(lax=None, network=None, subfig=None):
    global global_index
    global global_max_epochs
    global markeriter
    density = 0.001
    plts = []
    isacc=True
    #isacc=False
    #nworkers_list=[4, 8, 16, 32]
    nworkers_list=[32]
    if network == 'lstm':
        global_max_epochs = 40;bs=20;lr=2.0;nworkers=32
        lr=1.0
        line = plot_with_params(network, 1, bs, lr, 'scigpu11', r'Dense-SGD (P=1,bs=%d,lr=%f)'%(bs, lr), isacc=isacc, prefix='singlegpu-gwarmup-dc1-model-exp', nsupdate=1, force_legend=True, force_color='black')
        lr=2.0
        line = plot_with_params(network, 1, bs, lr, 'scigpu11', r'Dense-SGD (P=1,bs=%d,lr=%f)'%(bs, lr), isacc=isacc, prefix='singlegpu-gwarmup-dc1-model-exp', nsupdate=1, force_legend=True, force_color='black')
    elif network == 'resnet110':
        global_max_epochs = 160;bs=128;lr=0.1
        hostname='scigpu12';bs=32
        hostname='scigpu12';bs=64
        #hostname='scigpu12';bs=128
        markeriter = itertools.cycle(markers)
        update='convergence'
        line = plot_with_params(network, 1, 128, lr, 'scigpu11', r'Dense-SGD (P=1,bs=%d,lr=%f)'%(bs, lr), isacc=isacc, prefix='singlegpu-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='black')
        markeriter = itertools.cycle(markers)
        update='convergence'
        for nworkers in nworkers_list:
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            lr=0.566;update='convergence'
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            lr=3.2;update='convergence'
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')

        #markeriter = itertools.cycle(markers)
        #density=0.001;lr=0.1;bs=128
        #for nworkers in nworkers_list:
        #    #line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'TopK-SGD (P=%d)'%nworkers,isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-gtopkjournal', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
        #    #update='r1'
        #    update='convergence';lr=0.1
        #    line = plot_with_params(network, nworkers, bs, lr, 'scigpu12', r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')
        #    update='convergence';lr=0.566
        #    line = plot_with_params(network, nworkers, bs, lr, 'scigpu12', r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')
        #    update='convergence';lr=3.2
        #    line = plot_with_params(network, nworkers, bs, lr, 'scigpu12', r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')

        markeriter = itertools.cycle(markers)
        hostname='scigpu12';bs=32
        #hostname='scigpu12';bs=64
        #hostname='scigpu13';bs=128
        for nworkers in nworkers_list:
            update='convergence';lr=0.1
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
            update='convergence';lr=0.566
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            update='convergence';lr=3.2
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')

        for nworkers in nworkers_list:
            markeriter = itertools.cycle(markers)
            update='convergence';lr=0.1
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopKDD-SGD+MC (P=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC (P=%d,bs=%d,lr=%f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='blue')

            update='convergence';lr=0.566
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopKDD-SGD+MC (P=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC (P=%d,bs=%d,lr=%f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='blue')
            update='convergence';lr=3.2
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopKDD-SGD+MC (P=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC (P=%d,bs=%d,lr=%f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='blue')

        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            update='convergence';lr=0.1
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')

    elif network == 'resnet20':
        global_max_epochs = 160;bs=128;lr=0.1
        markeriter = itertools.cycle(markers)
        update='convergence'
        line = plot_with_params(network, 1, 128, lr, 'scigpu11', r'Dense-SGD (P=1,BS=128,lr=0.1)', isacc=isacc, prefix='singlegpu-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='black')
        #update='mc'
        markeriter = itertools.cycle(markers)
        hostname='scigpu13';bs=128
        #hostname='scigpu12';bs=64
        for nworkers in nworkers_list:
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            update='convergence'
            lr=0.566
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')

        #markeriter = itertools.cycle(markers)
        #density=0.001;lr=0.1;bs=128
        #for nworkers in nworkers_list:
        #    #line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'TopK-SGD (P=%d)'%nworkers,isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-gtopkjournal', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
        #    #update='r1'
        #    update='convergence';lr=0.1
        #    line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')
        #    update='convergence';lr=0.566
        #    line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')
        #    update='convergence';lr=3.2
        #    line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')

        markeriter = itertools.cycle(markers)
        density=0.001;lr=0.1;
        for nworkers in nworkers_list:
            update='convergence';lr=0.1
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='y')
            update='convergence';lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='y')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='y')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='y')
            #update='mc';lr=0.141
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='y')
            #update='mc-devidenorm';lr=0.141
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopKDD-SGD+MC (P=%d,lr=%.4f,ADD=1,MAX_DELAY=6)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')

        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            update='convergence';lr=0.1
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopKDD-SGD+MC (P=%d,lr=%.4f,ADD=1,MAX_DELAY=6)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC (P=%d,bs=%d,lr=%f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='blue')
            update='convergence';lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC (P=%d,bs=%d,lr=%f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='blue')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopKDD-SGD+MC (P=%d,lr=%.4f,ADD=1,MAX_DELAY=6)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC (P=%d,bs=%d,lr=%f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='blue')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopKDD-SGD+MC (P=%d,lr=%.4f,ADD=1,MAX_DELAY=6)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')

        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            update='convergence';lr=0.1
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')

    elif network == 'vgg16':
        markeriter = itertools.cycle(markers)
        global_max_epochs=160;bs=128;lr=0.1
        update='convergence'
        line = plot_with_params(network, 1, 128, lr, 'scigpu11', r'Dense-SGD (P=1,BS=128,lr=0.1)', isacc=isacc, prefix='singlegpu-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='black')
        update='convergence'
        markeriter = itertools.cycle(markers)
        hostname='scigpu13';bs=128;isacc=True
        hostname='scigpu12';bs=32
        for nworkers in nworkers_list:
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            #lr=3.2;update='convergence'
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu12', r'Dense-SGD (P=%d,lr=%f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            lr=0.566;update='convergence'
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='r')
            lr=3.2;update='convergence'
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')
        lr=0.1
        markeriter = itertools.cycle(markers)
        density=0.001;

        markeriter = itertools.cycle(markers)
        density=0.001;lr=0.1
        """
        for nworkers in nworkers_list:
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'TopK-SGD (P=%d)'%nworkers,isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-gtopkjournal', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
            #update='nmc'
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')
            lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')
            lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')
        """

        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            update='convergence';lr=0.1
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            update='convergence';lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='y')
            #update='convergence';lr=3.2
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu13', r'EFTopK-SGD+MC (P=%d,lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu12', r'EFTopK-SGD (P=%d,lr=%f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='g')

        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            #update='mc-devidenorm1';lr=0.1
            #line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'EFTopKDD-SGD+MC(P=%d,lr=%.4f,ADD=0.3,MAX_DELAY=1.5)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            update='convergence';lr=0.1
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC(P=%d,bs=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr),isacc=isacc,  prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True,force_color='blue')
            #line = plot_with_params(network, nworkers, bs, lr,hostname', r'EFTopKDD-SGD+MC(P=%d,lr=%.4f,ADD=0.6,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            update='convergence';lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC(P=%d,bs=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr),isacc=isacc,  prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True,force_color='blue')
            #line = plot_with_params(network, nworkers, bs, lr,hostname', r'EFTopKDD-SGD+MC(P=%d,lr=%.4f,ADD=0.6,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDD-SGD+MC(P=%d,bs=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr),isacc=isacc,  prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True,force_color='blue')
            #line = plot_with_params(network, nworkers, bs, lr,hostname, r'EFTopKDD-SGD+MC(P=%d,lr=%.4f,ADD=0.6,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopkdd-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            update='convergence';lr=0.1
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=0.566
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, hostname, r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')

    elif network == 'lstman4':
        pass
    elif network == 'resnet50':
        #isacc=False
        markeriter = itertools.cycle(markers)
        global_max_epochs=90;bs=64;lr=0.1
        #update='convergence'
        #line = plot_with_params(network, 1, 128, lr, 'scigpu11', r'Dense-SGD (P=1,BS=128,lr=0.1)', isacc=isacc, prefix='singlegpu-baseline-gwarmup-dc1-gtopkjournal-%s'%update, nsupdate=1, force_legend=True, force_color='black')
        update='convergence'
        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            #'allreduce-baseline-gwarmup-dc1-gtopkjournal-convergence/resnet50-n32-bs64-lr0.1000-ns1'
            #lr=0.1
            #line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'Dense-SGD (P=%d,bs=%d,lr=%.4f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-convergence', nsupdate=1, force_legend=True, force_color='r')
            lr=0.8
            #line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-gtopkjournal-convergence', nsupdate=1, force_legend=True, force_color='r')
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'Dense-SGD (P=%d,bs=%d,lr=%.4f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color='r')

        markeriter = itertools.cycle(markers)
        density=0.001;lr=0.8
        for nworkers in nworkers_list:
            lr=0.1
            #line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopK-SGD+MC (P=%d.lr=%.4f)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-comp-eftopk-baseline-gwarmup-dc1-gtopkjournal-convergence', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
            #line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            lr=0.8
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')
            lr=1.2
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopK-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopk-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='y')

        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            #update='convergence';lr=0.1
            #line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDD-SGD+MC(P=%d,bs=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr),isacc=isacc,  prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True,force_color='blue')
            update='convergence';lr=0.8
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDD-SGD+MC(P=%d,bs=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,bs,lr),isacc=isacc,  prefix='allreduce-mc-comp-eftopkdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True,force_color='blue')
            #version='eftopkddr4'
            #line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDD-SGD+MC(P=%d,bs=%d,lr=%.4f,%s)'%(nworkers,bs,lr,version),isacc=isacc,  prefix='allreduce-mc-comp-%s-gwarmup-dc1-model-exp-thres-512000kbytes'%version, nsupdate=1, density=density, force_legend=True,force_color='blue')

        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            #update='convergence';lr=0.1
            #line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=0.8
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=1.2
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=2.4
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=3.2
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=4.8
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=6.4
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=8.0
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=9.6
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=11.2
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')
            update='convergence';lr=12.8
            line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFTopKDecay-SGD+MC (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-eftopkdecay-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='orange')

        #markeriter = itertools.cycle(markers)
        #density=0.001;lr=0.8
        #for nworkers in nworkers_list:
        #    update='convergence';lr=0.1
        #    line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFDGCDD-SGD+MC(P=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-mc-comp-dgcsamplingdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, sg=None, density=density, force_legend=True,force_color='blue')
        #    update='convergence';lr=0.8
        #    line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFDGCDD-SGD+MC(P=%d,lr=%.4f,ADD=0.2,MAX_DELAY=1.3)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-mc-comp-dgcsamplingdd-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, sg=None, density=density, force_legend=True,force_color='blue')
        #    update='r1';lr=0.8
        #    line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFDGCDD-SGD+MC(P=%d,lr=%.4f,ADD=0.3,MAX_DELAY=1.8)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-mc-comp-dgcsamplingdd-gwarmup-dc1-model-exp-%s-thres-512000kbytes'%update, nsupdate=1, sg=None, density=density, force_legend=True,force_color='blue')
        #    update='r2';lr=0.8
        #    line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFDGCDD-SGD+MC(P=%d,lr=%.4f,ADD=1,MAX_DELAY=6)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-mc-comp-dgcsamplingdd-gwarmup-dc1-model-exp-%s-thres-512000kbytes'%update, nsupdate=1, sg=None, density=density, force_legend=True,force_color='blue')
        #    update='r3';lr=0.8
        #    line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFDGCDD-SGD+MC(P=%d,lr=%.4f,ADD=1,MAX_DELAY=20)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-mc-comp-dgcsamplingdd-gwarmup-dc1-model-exp-%s-thres-512000kbytes'%update, nsupdate=1, sg=None, density=density, force_legend=True,force_color='blue')
        #    update='r4';lr=0.8
        #    line = plot_with_params(network, nworkers, bs, lr, 'gpu9', r'EFDGCDD-SGD+MC(P=%d,lr=%.4f,ADD=1,MAX_DELAY=200)'%(nworkers,lr),isacc=isacc,  prefix='allreduce-mc-comp-dgcsamplingdd-gwarmup-dc1-model-exp-%s-thres-512000kbytes'%update, nsupdate=1, sg=None, density=density, force_legend=True,force_color='blue')
        pass
    elif network in ['resnet152', 'densenet161', 'densenet201']:
        pass
    elif network == 'inceptionv4':
        pass
    elif network == 'alexnetbn':
        nworkers_list = [16]
        markeriter = itertools.cycle(markers)
        global_max_epochs=95;bs=256;lr=0.01
        #update='convergence'
        line = plot_with_params(network, 1, bs, lr, 'scigpu10', r'Dense-SGD (P=1,BS=%d,lr=%.4f)'%(bs,lr), isacc=isacc, prefix='singlegpu-gwarmup-dc1-model-exp', nsupdate=1, force_legend=True, force_color='black')
        update='convergence'
        markeriter = itertools.cycle(markers)
        for nworkers in nworkers_list:
            #'allreduce-baseline-gwarmup-dc1-gtopkjournal-convergence/resnet50-n32-bs64-lr0.1000-ns1'
            lr=0.16
            line = plot_with_params(network, nworkers, bs, lr, 'scigpu11', r'Dense-SGD (P=%d,lr=%.4f)'%(nworkers,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-hvd-thres-512000kbytes', nsupdate=1, force_legend=True, force_color='r', density=1.0)
    return plts

def best_to_best(lax=None, network=None, subfig=None):
    global global_index
    global global_max_epochs
    global markeriter
    isacc=True
    #isacc=False
    density=0.001
    nworkers=32
    compressors = ['none', 'eftopk', 'eftopkdecay']
    if network == 'lstm':
        global_max_epochs = 40;bs=32;lr=22.0;nworkers=32
        pass
    elif network in ['vgg16', 'resnet110', 'resnet20']:
        global_max_epochs = 160;bs=128
        hostname='scigpu12'
        if network == 'vgg16':
            lrs = [0.1, 0.1, 0.566]
        elif network == 'resnet110':
            lrs = [0.1, 0.1, 0.1]
        elif network == 'resnet20':
            lrs = [1.6, 0.8, 1.6]
    markeriter = itertools.cycle(markers)
    for idx, compressor in enumerate(compressors):
        lr = lrs[idx]
        if compressor == 'none':
            _, acc = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-exp-thres-512000kbytes', nsupdate=1, density=1.0, force_legend=True, force_color=fixed_colors[compressor])
        else:
            _, acc = plot_with_params(network, nworkers, bs, lr, hostname, r'%s w/ MC (P=%d,bs=%d,lr=%f)'%(compressor, nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-%s-gwarmup-dc1-model-exp-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color=fixed_colors[compressor])


def plot_convergence_new(lax=None, network=None, subfig=None):
    global global_index
    global global_max_epochs
    global markeriter
    isacc=True
    #isacc=False
    exp='nips2020'
    density = 0.001
    network='vgg16'
    if network in ['resnet20', 'resnet110', 'vgg16']:
        global_max_epochs = 160;
        #hostname='scigpu12';nworkers=32;bs=128
        hostname='gpu9';nworkers=32;bs=128
        #hostname='scigpu11';nworkers=4;bs=128
        #hostname='scigpu12';nworkers=8;bs=128;density=0.01
        #hostname='hsw221';nworkers=16;bs=32
        #lrs=[0.1, 0.2, 0.4, 0.566, 0.8, 1.6, 3.2]
        lrs=[0.0001, 0.001, 0.01, 0.1, 1]
        #lrs=[0.1, 0.566, 3.2]
        #lrs=[0.1] #, 0.566, 3.2]
    elif network in ['lstm', 'lstmwt2']:
        global_max_epochs=40;bs=4;nworkers=32
        hostname='hsw221'
        lrs=[20]
    elif network in ['resnet50']:
        global_max_epochs=90;bs=64;nworkers=32
        hostname='gpu9'
        lrs=[0.1, 0.8, 1.2, 2.4, 3.2, 4.8, 6.4, 9.6, 11.2, 12.8]
        #lrs=[0.8, 1.2, 2.4]
        #lrs=[0.8]
    elif network == 'alexnetbn':
        global_max_epochs=95;bs=256;lr=0.01
        #hostname='scigpu11';nworkers=32
        hostname='gpu9';nworkers=32
    #   lrs=[0.01, 0.32, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 3.2]
    #compressors = ['none'] #, 'eftopk', 'eftopkdecay', 'eftopkdd', 'eftopk-womc', 'eftopkdecay-womc', 'eftopkdd-womc']
    #compressors = ['none', 'eftopk', 'eftopkdecay', 'eftopk-womc', 'eftopkdecay-womc']
    compressors = ['none', 'eftopk', 'eftopkdecay'] #, 'eftopkdd']
    #compressors = ['none', 'eftopk', 'eftopkdecay', 'eftopkdd']
    #compressors = ['none', 'eftopk', 'eftopkdecay', 'eftopkdd', 'signum', 'efsignum', 'efsignumdecay']
    ##compressors = ['eftopk', 'eftopkdecay'] #, 'signum', 'efsignum', 'efsignumdecay']

    markeriter = itertools.cycle(markers)
    bests = []
    for compressor in compressors:
        markeriter = itertools.cycle(markers)
        max_lr = 0
        max_acc = 0; max_std = 0
        for lr in lrs:
            try:
                if compressor == 'none':
                    _, acc, std = plot_with_params(network, nworkers, bs, lr, hostname, r'Dense-SGD (P=%d,bs=%d,lr=%f)'%(nworkers,bs,lr), isacc=isacc, prefix='allreduce-gwarmup-dc1-model-%s-thres-512000kbytes'%exp, nsupdate=1, density=1.0, force_legend=True, force_color=fixed_colors[compressor])
                elif compressor.find('womc') > 0:
                    _, acc, std = plot_with_params(network, nworkers, bs, lr, hostname, r'%s w/o MC (P=%d,bs=%d,lr=%f)'%(compressor, nworkers,bs,lr), isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-%s-thres-512000kbytes'%(compressor.split('-')[0], exp), nsupdate=1, density=density, force_legend=True, force_color=fixed_colors[compressor])
                else:
                    _, acc, std = plot_with_params(network, nworkers, bs, lr, hostname, r'%s (P=%d,bs=%d,lr=%f)'%(compressor, nworkers,bs,lr), isacc=isacc, prefix='allreduce-mc-comp-%s-gwarmup-dc1-model-%s-thres-512000kbytes'%(compressor, exp), nsupdate=1, density=density, force_legend=True, force_color=fixed_colors[compressor])
                if max_acc < acc:
                    max_acc = acc
                    max_std = std
                    max_lr = lr
            except Exception as e:
                print('Exception: %s' % e)
        #print('-'*20)
        #print('--> Compressor: %s, max_acc: %f at lr: %f' % (compressor, max_acc, max_lr))
        #print('-'*20)
        bests.append((compressor, max_acc, max_lr))
    for compressor, max_acc, max_lr in bests:
        print('--> Compressor: %s, max_acc: %f +- %d at lr: %f' % (compressor, max_acc, max_std, max_lr))


def plot_single_convergence():
    global global_index
    global global_max_epochs
    global LOGHOME
    #LOGHOME='/media/sf_Shared_Data/gpuhome/repositories/p2p-dl/logs'
    #LOGHOME='./logs'
    LOGHOME='/Users/lele/shared-server/gaussiank-logs'
    #network='resnet20';
    #network='resnet110';
    #network='vgg16';
    #network='lstm';
    #network='lstmwt2';
    #network='resnet50';
    #network='alexnetbn';
    #plts = plot_convergence(ax, network)
    #ax.legend()
    plts = plot_convergence_new(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #plts = best_to_best(ax, network)
    #plot_sub_boxes(ax)
    plt.subplots_adjust(bottom=0.10, left=0.10, right=0.70, top=0.84, wspace=0.63, hspace=0.55)
    ax.grid(linestyle=':')


if __name__ == '__main__':
    plot_single_convergence()
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())
    #mng.frame.Maximize(True)
    plt.show()
