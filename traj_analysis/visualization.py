import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from functools import partial
from itertools import cycle


def parse_plt_label(name, ncharge):
    if '_minus' in name or '_plus' in name:
        name = name.split('_')[0]
    if ncharge == 0: return '', '', name
    pom = '+' if ncharge>=0 else '-'
    t = '' if ncharge in [1, -1] else '%d'%abs(ncharge)
    name = re.sub(r'(\d+)', r'$_{\1}$', name)
    name = r'%s$^{%s\rm{%s}}$'%(name, t, pom.replace(r"-", u"\u2212"))
    name = name.replace(r'$$', '')
    return pom, t, name


def plot_rxn_changed_mols(df, charge_dict={}, reverse=False):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['mathtext.default'] = 'regular'
    a = 0.8
    base_dict = {'Li': 1, 'PF6': -1}
    _marker = ('o', '^', 's', 'D', '*', 'v', '>', '<')
    assert isinstance(charge_dict, dict)
    charge_dict.update(base_dict)
    df = df.copy()
    n_excess_Li, xlabel = 0, ''
    for col in df.columns:
        if 'origin_' in col and col[7:] in charge_dict:
            nc = charge_dict[col[7:]]
            n_excess_Li += df[col]*nc
            pom, t, l = parse_plt_label(col[7:], nc)
            pom = ' %s '%pom
            t = t+'*' if t!='' else t
            if pom == ' - ':
                xlabel += '%s%sn[%s]'%(pom, t, l)
            else:
                xlabel = '%s%sn[%s]'%(pom, t, l)+xlabel
    if reverse:
        xlabel = xlabel.replace(' + ', ' = ').replace(
                        ' - ', ' + ').replace(' = ', ' - ')
        n_excess_Li = -n_excess_Li
    xlabel_tmp = xlabel.split(' + ', 1)
    xlabel_tmp.append(xlabel_tmp.pop(0))
    xlabel = ''.join(xlabel_tmp)
    xlabel = xlabel.replace(' - ', ' \N{MINUS SIGN} ')
    xticks = n_excess_Li.values
    xl, xr = xticks.min(), xticks.max()
    N_ = xr - xl
    for i in range(2, 5, 1):
        if N_//i in [5, 6, 7, 8]:
            xticks = np.arange(xl, xr+i, i)
            break
    df['excess_Li'] = n_excess_Li
    df.drop(columns=['config_id', 'rxn', 'rdc'], inplace=True)
    gb = df.groupby(['excess_Li']).agg(['mean', partial(np.std, ddof=0)])
    x = gb.index.values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    marker = cycle(_marker) 
    for col in df.columns:
        if 'rxn_' in col:
            l = col[4:]
            nc = charge_dict[l] if l in charge_dict else 0
            pom, t, nl = parse_plt_label(l, nc)
            ax1.errorbar(x, gb[col]['mean'], gb[col]['std'], ls='--', 
                 marker=next(marker), alpha=a, label=nl, capsize=3)
    handles, labels = ax1.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=16)
    ax1.set_ylabel('Avg. mol. count', fontsize=18)
    yticks = [0,1,2,3]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=16)
    
    marker = cycle(_marker) 
    for col in df.columns:
        if 'rdc_' in col:
            l = col[4:]
            nc = charge_dict[l] if l in charge_dict else 0
            pom, t, nl = parse_plt_label(l, nc)
            ax2.errorbar(x, gb[col]['mean'], gb[col]['std'], ls='--', 
                 marker=next(marker), alpha=a, label=nl, capsize=3)
    handles, labels = ax2.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    ax2.legend(handles, labels, fontsize=16, loc='upper center')
    title = 'Reduction' if not reverse else 'Oxidation'
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=16)
    
    plt.tight_layout()
    plt.savefig('Fig_3bc.tiff', dpi=600)
    plt.show()


def plot_rxn_cleaned_timestep_hist(df, bins=25, show_fake_rxn=False, xrange=(0, 250)):
    a = 0.7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    X, L = [], []
    #remove data from last frame
    rxn_idx = [i for i, x, fs in zip(df.idx, df.rxn_cleaned_diff_src, df.rxn_is_last_frame) 
               if x!=set() for y, f in zip(x, fs) if not f]
    X.append(rxn_idx), L.append('rxn')
    if show_fake_rxn:
        #remove data from last frame
        fake_rxn_idx = [i for i, x in zip(df.idx, df.rxn_transient_src) \
                        if x!=set() for y in x]
        X.append(fake_rxn_idx), L.append('transient_rxn')
    ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, zorder=0, stacked=True)
    ax1.set_title('Reaction', fontsize=14)
    ax1.set_xlabel('Time step', fontsize=13)
    ax1.set_ylabel('Frequency', fontsize=13)

    X, L = [], []
    #remove data from last frame
    rdc_idx = [i for i, x, fs in zip(df.idx, df.rdc_cleaned_diff_src, df.rdc_is_last_frame) 
               if x!=set() for y, f in zip(x, fs) if not f]
    X.append(rdc_idx), L.append('rxn')
    if show_fake_rxn:
        #remove data from last frame
        fake_rdc_idx = [i for i, x in zip(df.idx, df.rdc_transient_src) \
                        if x!=set() for y in x]
        X.append(fake_rdc_idx), L.append('transient_rxn')
    ax2.hist(X, bins=bins, range=xrange, label=L, alpha=a, zorder=0, stacked=True)
    ax2.set_title('Reduction', fontsize=14)
    ax2.set_xlabel('Time step', fontsize=13)
    ax2.set_ylabel('Frequency', fontsize=13)
    if show_fake_rxn:
        ax1.legend()
        ax2.legend()
    plt.show()


def plot_rxn_cleaned_timestep_hist_by_mols(df, bins=25, show_fake_rxn=False, 
                                   xrange=(0, 250), charge_dict={}):
    a = 0.7
    base_dict = {'Li': 1, 'PF6': -1}
    assert isinstance(charge_dict, dict)
    charge_dict.update(base_dict)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    mol_name = [x[7:] for x in df.columns if ('origin_' in x) and (x[7:]!='Li')]
    def rxn_mol_cnt(x,  name):
        src = [y.rsplit('_', 1)[0] for y in x.iloc[0]]
        res = {'%s_%s_cnt'%(name,y): src.count(y) for y in mol_name}
        return res

    name = 'rxn_cleaned_diff_src'
    applied_df = df[['rxn_cleaned_diff_src']].apply(rxn_mol_cnt, args=(name,), 
                        axis='columns', result_type='expand')
    df = pd.concat([df, applied_df], axis='columns')
    X, L = [], []
    for mn in mol_name:
        #remove data from last frame
        mn_idx = [i for i, x, fs in zip(df.idx, df['%s_%s_cnt'%(name, mn)], 
            df['rxn_is_last_frame']) if x!=0 for y, f in zip(range(x), fs) if not f]
        nc = charge_dict[mn] if mn in charge_dict else 0
        pom, t, nl = parse_plt_label(mn, nc)
        X.append(mn_idx)
        L.append(nl)
    ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
    ax1.set_xlabel('DFT call', fontsize=18)
    xticks = [0,50,100,150,200,250]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=18)
    yticks = [0,10,20,30,40,50,60,70,80]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=16)

    name = 'rdc_cleaned_diff_src'
    applied_df = df[['rdc_cleaned_diff_src']].apply(rxn_mol_cnt, args=(name,), 
                        axis='columns', result_type='expand')
    df = pd.concat([df, applied_df], axis='columns')
    X, L = [], []
    for mn in mol_name:
        #remove data from last frame
        mn_idx = [i for i, x, fs in zip(df.idx, df['%s_%s_cnt'%(name, mn)], 
            df['rdc_is_last_frame']) if x!=0 for y, f in zip(range(x), fs) if not f]
        nc = charge_dict[mn] if mn in charge_dict else 0
        pom, t, nl = parse_plt_label(mn, nc)
        X.append(mn_idx)
        L.append(nl)
    ax2.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
    ax2.legend(fontsize=16)
    ax2.set_xlabel('DFT call', fontsize=18)
    xticks = [0,50,100,150,200,250]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=16)
    yticks = [0,1,2,3,4,5,6,7]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks, fontsize=16)

    plt.tight_layout()
    plt.savefig('Fig_3de.tiff', dpi=600)
    plt.show()

    if show_fake_rxn:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        name = 'transient_rxn_cleaned_diff_src'
        applied_df = df[['rxn_cleaned_transient_src']].apply(rxn_mol_cnt, args=(name,), 
                            axis='columns', result_type='expand')
        df = pd.concat([df, applied_df], axis='columns')
        X, L = [], []
        for mn in mol_name:
            #remove data from last frame
            mn_idx = [i for i, x in zip(df.idx, df['%s_%s_cnt'%(name, mn)]) \
                      if x!=0 for y in range(x)]
            nc = charge_dict[mn] if mn in charge_dict else 0
            pom, t, nl = parse_plt_label(mn, nc)
            X.append(mn_idx)
            L.append(nl)
        ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
        ax1.legend()
        ax1.set_title('Transient Reaction', fontsize=14)
        ax1.set_xlabel('Time step', fontsize=13)
        ax1.set_ylabel('Frequency', fontsize=13)

        name = 'transient_rdc_cleaned_diff_src'
        applied_df = df[['rdc_cleaned_transient_src']].apply(rxn_mol_cnt, args=(name,), 
                            axis='columns', result_type='expand')
        df = pd.concat([df, applied_df], axis='columns')
        X, L = [], []
        for mn in mol_name:
            #remove data from last frame
            mn_idx = [i for i, x in zip(df.idx, df['%s_%s_cnt'%(name, mn)]) \
                      if x!=0 for y in range(x)]
            nc = charge_dict[mn] if mn in charge_dict else 0
            pom, t, nl = parse_plt_label(mn, nc)
            X.append(mn_idx)
            L.append(nl)
        ax2.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
        ax2.legend()
        ax2.set_title('Transient Reduction', fontsize=14)
        ax2.set_xlabel('Time step', fontsize=13)
        ax2.set_ylabel('Frequency', fontsize=13)
        plt.show()
    return df
