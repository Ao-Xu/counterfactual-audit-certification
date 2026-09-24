"""Evidence-preserving appendix figure repairs. Never executes code/ plotters.

Python 3.10 with numpy, pandas, matplotlib, PyMuPDF, Pillow.
Outputs only approved figures, their repair_data, and the repair QA directory.
"""
from pathlib import Path
import argparse
import base64
import copy
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / '../../idea_1/iclr_2027_comparative_utility'
QA = ROOT / 'build/revision/figure_audit/repairs'
STAGE = QA / 'stage'
BLUE, ORANGE, GREEN, PURPLE, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#666666'
COLORS = [GRAY, BLUE, '#E69F00', ORANGE]
MARKS = ['o', 's', '^', 'D', 'v']
METHODS = ['separate_split', 'separate_shared', 'paired_split', 'paired_shared']
LABELS = ['Separate / split', 'Separate / shared', 'Paired / split', 'Paired / shared']
LEDGER = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def eligible():
    text = (ROOT / 'appendix.tex').read_text(encoding='utf8')
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    return [(m.group(2), float(re.search(r'width=([\d.]+)', m.group(1)).group(1))
             if re.search(r'width=([\d.]+)', m.group(1)) else 1.)
            for m in re.finditer(r'\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}', text)
            if 'oral_planner_v6' not in m.group(2)
            and m.group(2) not in ('figures/intro.pdf', 'figures/audit_evidence.pdf')]


def initialize():
    QA.mkdir(parents=True, exist_ok=True)
    saved = QA / 'initial_manifest.json'
    if saved.exists():
        return json.loads(saved.read_text(encoding='utf8'))
    old = json.loads((ROOT / 'build/revision/figure_audit/inventory.json').read_text(encoding='utf8'))
    lookup = {r['path']: r['id'] for r in old['figures']}
    rows = []
    for path, width in eligible():
        src = ROOT / path
        back = QA / 'originals' / path
        back.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(back))
        if src.with_suffix('.svg').exists():
            shutil.copy2(str(src.with_suffix('.svg')), str(back.with_suffix('.svg')))
        rows.append({'path': path, 'id': lookup.get(path, src.stem), 'width_fraction': width,
                     'before_sha256': sha(src), 'original_pdf': str(back.relative_to(ROOT))})
    protected = ['main.tex', 'appendix.tex', 'figures/intro.pdf']
    protected += [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT/'experiments/oral_planner_v6').glob('*') if p.is_file()]
    obj = {'figures': rows, 'protected_initial_hashes': {p: sha(ROOT/p) for p in protected}}
    saved.write_text(json.dumps(obj, indent=2), encoding='utf8')
    return obj


def source(path, relative=None):
    """Copy exact source bytes, never extrapolate data from graphics."""
    path = Path(path)
    if relative is None:
        relative = path.relative_to(LEGACY).as_posix()
    rel = Path(relative)
    dest = ROOT / rel.parent / 'repair_data' / rel.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or sha(dest) != sha(path):
        shutil.copy2(str(path), str(dest))
    return dest


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .65, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'lines.linewidth': 1.3,
        'savefig.transparent': False, 'axes.unicode_minus': True})


def axes_style(ax):
    ax.grid(axis='y', color='#dddddd', lw=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.65)


def external(ax, ncol=2, y=-.28, **kwargs):
    kwargs.setdefault('columnspacing',1.1)
    return ax.legend(loc='upper center', bbox_to_anchor=(.5,y), ncol=ncol,
                     frameon=False, handlelength=2, **kwargs)


def finish(fig, path, sources, notes, mode='data redraw'):
    assert path in dict(eligible()), 'Figure removed or not authorized: '+path
    dest = STAGE / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Fixed 5.5-in canvas, not bbox_inches=tight: no hidden global font shrink.
    fig.savefig(str(dest), facecolor='white')
    fig.savefig(str(dest.with_suffix('.svg')), facecolor='white')
    plt.close(fig)
    LEDGER[path] = {'mode': mode, 'sources': [
        {'path': str(p.relative_to(ROOT)).replace('\\','/'), 'sha256': sha(p)} for p in sources],
        'notes': notes, 'status': 'staged; visual inspection required'}


def analytic():
    path='figures/comparative_utility/fig_comparative_utility.pdf'
    src=source(ROOT/'code/overleaf/experiments/comparative_utility/generate_comparative_utility.py',
               'figures/comparative_utility/original_formula_source.py')
    d=np.linspace(-1,1,401); m=-.12; alpha=-.58
    theta=np.linspace(0,np.pi,301); k=np.arange(1,33)
    v=.64/128+.36/(128*k)
    fig,axs=plt.subplots(3,1,figsize=(5.5,7.15))
    fig.subplots_adjust(left=.17,right=.96,bottom=.11,top=.96,hspace=1.0)
    a,b,c=axs
    a.fill_between(d,m-abs(alpha)*abs(d),m+abs(alpha)*abs(d),color='#dfe9ef',label='Certificate interval')
    a.plot(d,m+alpha*d,color=BLUE,label='Equality witness')
    a.axhline(m,color=GRAY,ls='--',label='Reference risk difference')
    a.axhline(0,color='#333333',lw=.65)
    a.set(xlim=(-1.04,1.04),ylim=(-.85,.65),xlabel=r'Covered-shift direction $\delta$',
          ylabel=r'Paired risk difference $\Delta$',title='(a) Covered-shift equality (analytic)')
    external(a,ncol=3,y=-.32,fontsize=7.6)
    b.plot(np.degrees(theta),np.sqrt(2-2*np.cos(theta)),color=BLUE,label='Paired radius')
    b.plot(np.degrees(theta),np.full_like(theta,2.),color=ORANGE,ls='--',label='Separate radii')
    b.set(xlim=(-4,184),ylim=(-.05,2.17),xticks=[0,45,90,135,180],
          xlabel='Angle between centered responses (degrees)',ylabel='Shift-radius contribution',
          title='(b) Pairing exploits cancellation (analytic)')
    external(b,ncol=2,y=-.32)
    c.plot(k,v,color=BLUE,label='Two-level variance')
    c.axhline(.64/128,color=ORANGE,ls='--',label='Anchor floor')
    c.scatter([1,4,16,32],v[[0,3,15,31]],color=GREEN,s=20,zorder=4,label='Selected K')
    c.set(xlim=(.2,32.8),ylim=(0,.0083),xticks=[1,4,8,16,32],xlabel='Siblings per anchor K',
          ylabel='Variance of the paired mean',title='(c) More siblings saturate (analytic)')
    external(c,ncol=3,y=-.32)
    for ax in axs: axes_style(ax)
    finish(fig,path,[src],['Same exact analytic formulas; K=1 and K=32 fully visible.',
         'Three original panels kept; separate exterior legends; n=128, sigma_A^2=.64, sigma_S^2=.36.'], 'analytic redraw')


def matched():
    base=ROOT/'code/oral_attribution_v5/results'
    sources=[source(base/f,'experiments/oral_attribution_v5/'+f) for f in
             ['dense_thresholds.json','factorial.csv','oracle_forecast.json','MATCHED_RESULTS_MANIFEST.json']]
    vals=json.loads(sources[0].read_text())['thresholds']
    lookup={(x['geometry'],x['method'],x['target']):x for x in vals}
    # Confirm every one of 64 manuscript threshold triples, not just selected cells.
    tex=(ROOT/'appendix.tex').read_text(encoding='utf8')
    table=tex[tex.index(r'ID & Separate/split'):]
    for target in [.8,.9]:
        target_section=table.split(str(int(target*100))+r'\% power}')[1].split(r'\multicolumn')[0]
        for g in range(8):
            line=re.search(r'(?m)^'+str(g)+r' & (.*)',target_section).group(1)
            triples=[tuple(map(int,x)) for x in re.findall(r'(\d+) \[(\d+), (\d+)\]',line)]
            expected=[tuple(lookup[g,m,target][k] for k in ['estimate','lower','upper']) for m in METHODS]
            if triples != expected:
                raise ValueError('Current TeX differs from reproduced data: geometry {}, target {}'.format(g,target))
    fc=json.loads(sources[2].read_text())['rows']
    fig,axs=plt.subplots(1,2,figsize=(5.5,3.6))
    fig.subplots_adjust(left=.12,right=.98,bottom=.34,top=.87,wspace=.62)
    for j,m in enumerate(METHODS):
        rr=[lookup[g,m,.9] for g in range(8)]
        y=np.array([r['estimate'] for r in rr])/1000
        er=np.array([[r['estimate']-r['lower'],r['upper']-r['estimate']] for r in rr]).T/1000
        axs[0].errorbar(np.arange(8)+(j-1.5)*.16,y,yerr=er,fmt=MARKS[j],color=COLORS[j],ms=3.5,capsize=2,label=LABELS[j])
        for target,mark in [(.8,'o'),(.9,'^')]:
            rows=[r for r in fc if r['method']==m and r['target']==target]
            obs=np.array([lookup[r['geometry'],m,target]['estimate'] for r in rows])/1000
            pred=np.array([r['oracle_forecast'] for r in rows])/1000
            axs[1].scatter(obs,pred,color=COLORS[j],marker=mark,s=18,facecolors='none' if target==.8 else COLORS[j])
    axs[0].set(xlabel='Frozen geometry ID',ylabel='90% threshold (thousands)',xticks=range(8),title='(a) Matched factorial')
    axs[1].plot([2,14],[2,14],ls=':',color=GRAY,lw=.8)
    axs[1].set(xlabel='Observed threshold\n(thousands)',ylabel='Oracle forecast (thousands)',title='(b) Forecast check')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
    fig.text(.5,.025,'Forecast markers: open circle 80%; filled triangle 90%',ha='center',fontsize=8)
    for ax in axs: axes_style(ax)
    finish(fig,'experiments/oral_attribution_v5/factorial_forecast.pdf',sources,
        ['All 64 threshold triples exactly match current manuscript table.',
         'Fresh frozen-confirmation provenance retained; no historical raw recovery asserted.'])
    factors=pd.read_csv(sources[1]); factors=factors[factors.target==.9].sort_values('geometry')
    fig,axs=plt.subplots(1,2,figsize=(5.5,3.2));fig.subplots_adjust(left=.12,right=.98,top=.87,bottom=.29,wspace=.57)
    for name,col,marker,color in [('Pairing','shapley_pairing_factor','o',ORANGE),('Reuse','shapley_reuse_factor','s',BLUE)]:
        axs[0].plot(factors.geometry,factors[col],ls='none',marker=marker,color=color,label=name)
    axs[0].set(xlabel='Geometry ID',ylabel='Symmetric budget factor',xticks=range(8),title='(a) Pairing and reuse')
    external(axs[0],y=-.37)
    y=factors.log_interaction.to_numpy()
    axs[1].errorbar(factors.geometry,y,yerr=[y-factors.log_interaction_lower.to_numpy(),factors.log_interaction_upper.to_numpy()-y],fmt='o',color=GRAY,capsize=2)
    axs[1].axhline(0,color=GRAY,lw=.7)
    axs[1].set(xlabel='Geometry ID',ylabel='Log interaction',xticks=range(8),title='(b) Interaction')
    for ax in axs: axes_style(ax)
    finish(fig,'experiments/oral_attribution_v5/path_independent.pdf',sources,
           ['Unchanged factorial estimates and threshold confidence sets; no connecting lines across categorical IDs.'])
    fig,ax=plt.subplots(figsize=(5.5,2.9));fig.subplots_adjust(left=.13,right=.97,top=.95,bottom=.27)
    contrasts=[('H-way penalty','generic_fixed_Hledger','C_generic_fixed'),('Ledger: 2 vs 1','D_generic_fixed_matched','C_generic_fixed'),('Loose / tight\nrange','generic_fixed_loose_range','D_generic_fixed_matched'),('Marginal / joint\nvariance','linear_marginal_EB','generic_fixed_loose_range')]
    for i,(name,num,den) in enumerate(contrasts):
        y=np.array([lookup[g,num,.9]['estimate']/lookup[g,den,.9]['estimate'] for g in range(8)])
        ax.scatter(np.full(8,i)+np.linspace(-.08,.08,8),y,s=21,facecolors='none',edgecolors=BLUE)
        ax.plot([i-.13,i+.13],[np.median(y)]*2,color=ORANGE,lw=2)
    ax.axhline(1,color=GRAY,ls=':',lw=.8);ax.set(xticks=range(4),xticklabels=[x[0] for x in contrasts],ylabel='90% anchor-budget ratio')
    axes_style(ax)
    finish(fig,'experiments/oral_attribution_v5/ablation.pdf',sources,
           ['All eight cells per contrast retained; dots are geometry ratios and short lines their median.'])


def qwen_cluster():
    src=source(LEGACY/'experiments/llm_cfpt/results/strict_cluster_scaling.csv')
    df=pd.read_csv(src)
    x=np.column_stack([1/df.n_eval,1/(df.n_eval*df.k_eval)])
    a,b=np.linalg.lstsq(x,df.variance.to_numpy(),rcond=None)[0]
    assert np.allclose(x @ [a,b],df.two_level_fit,rtol=1e-12,atol=1e-15)
    r2=lambda fit:1-np.sum((df.variance.to_numpy()-fit)**2)/np.sum((df.variance-df.variance.mean())**2)
    fig,axs=plt.subplots(2,2,figsize=(5.5,5.7));fig.subplots_adjust(left=.14,right=.97,top=.91,bottom=.15,hspace=.86,wspace=.55)
    for n,c,m in [(16,BLUE,'o'),(32,GREEN,'s'),(64,ORANGE,'^')]:
        r=df[df.n_eval==n].sort_values('k_eval')
        axs[0,0].errorbar(r.k_eval,r.variance*1000,yerr=r['std']*1000,color=c,marker=m,capsize=2,ms=4,label='n='+str(n))
    axs[0,0].set(xlabel='Fresh siblings K',ylabel='Risk variance (×10⁻³)',xticks=[1,2,4],title='(a) Sibling scaling')
    external(axs[0,0],ncol=3,y=-.37,fontsize=7.5,columnspacing=.65)
    ns=np.array([16,32,64]);axs[0,1].bar(range(3),1000*a/ns,color=BLUE,label='a/n')
    axs[0,1].bar(range(3),1000*b/(4*ns),bottom=1000*a/ns,color=ORANGE,hatch='///',label='b/(nK)')
    axs[0,1].set(xticks=range(3),xticklabels=ns,xlabel='Fresh anchors n',ylabel='Fitted variance (×10⁻³)',title='(b) Components, K=4')
    external(axs[0,1],y=-.37)
    for ax,col,lab,color,letter in [(axs[1,0],'two_level_fit','a/n + b/(nK)',BLUE,'c'),(axs[1,1],'naive_fit','c/(nK)',PURPLE,'d')]:
        xx=df[col].to_numpy()*1000;yy=df.variance.to_numpy()*1000
        ax.scatter(xx,yy,color=color,s=24)
        lim=[min(xx.min(),yy.min())*.88,max(xx.max(),yy.max())*1.07]
        ax.plot(lim,lim,ls='--',lw=.8,color=GRAY)
        ax.set(xlim=lim,ylim=lim,xlabel=lab+' fitted (×10⁻³)',ylabel='Bootstrap variance (×10⁻³)',title='('+letter+') '+('Two-level' if letter=='c' else 'Naive')+' fit; R²={:.4f}'.format(r2(df[col].to_numpy())))
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axs.flat:axes_style(ax)
    fig.text(.5,.972,'Fresh-evaluation variance diagnostic',ha='center',fontsize=10)
    fig.text(.5,.025,'Error bars: SD across 24 models; 1,200 bootstrap draws per model/cell',ha='center',fontsize=7.8)
    finish(fig,'experiments/llm_cfpt/figures/fig_c5_cluster_scaling.pdf',[src],
        ['Exact nine CSV cells; existing fits checked against original least-squares formula.',
         'Only display units scaled by 1000; sparse ticks; SD not CI; all four panels retained.'])


def hans():
    src=source(LEGACY/'experiments/hans_suite/outputs/results.csv')
    df=pd.read_csv(src)
    order=['factual','natural_cf','hans_balanced','aligned_corrupt','reversed_corrupt']
    colors=[GRAY,'#56B4E9',GREEN,ORANGE,BLUE]
    names=['Factual','Natural\nCF','HANS\nbalanced','Aligned\nerror','Reversed\nerror']
    fullnames=['Factual','Natural CF','HANS balanced','Aligned error','Reversed error']
    groups=['lexical_overlap','subsequence','constituent']
    sub=df[df.split.str.startswith('hans_') & (df.split!='hans_all')].copy()
    sub['group']=sub.split.str.replace('hans_','',regex=False)
    fig=plt.figure(figsize=(5.5,6.15)); gs=fig.add_gridspec(2,2,hspace=.67,wspace=.52)
    axs=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    fig.subplots_adjust(left=.14,right=.97,top=.89,bottom=.12)
    for j,cond in enumerate(order):
        s=sub[sub.condition==cond]; gr=s.groupby('group',sort=False).risk
        mean=gr.mean().reindex(groups);e=gr.std().reindex(groups).fillna(0)/np.sqrt(s.seed.nunique())
        axs[0].errorbar(range(3),mean,yerr=1.96*e,marker=MARKS[j],color=colors[j],ls='none',capsize=2.5,ms=4,label=fullnames[j])
    axs[0].set(xticks=range(3),xticklabels=['Lexical','Subsequence','Constituent'],ylabel='Brier risk (bounded)',title='(a) Heuristic-group risk')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,fontsize=8)
    for ax,metric,letter,label in [(axs[1],'risk','b','Mean HANS Brier risk'),(axs[2],'nll','c','Mean HANS NLL')]:
        g=sub.groupby('condition')[metric].agg(['mean','std','count']).reindex(order)
        err=1.96*g['std'].fillna(0)/np.sqrt(g['count'])
        ax.barh(range(5),g['mean'],xerr=err,capsize=2,color=colors,height=.72)
        ax.set(yticks=range(5),yticklabels=names if metric=='risk' else ['']*5,xlabel=label,
               title='('+letter+') '+('Bounded risk' if metric=='risk' else 'NLL diagnostic'))
        ax.invert_yaxis()
    for ax in axs:axes_style(ax)
    finish(fig,'experiments/hans_suite/outputs/fig_hans_main.pdf',[src],
        ['Three original panels retained in wide-top + two-bottom layout (the supplied HANS PDF has three, not four panels).',
         'Original seed/group aggregation and 1.96*SE bars unchanged; statistical aggregation caveat remains for panels b/c.',
         'Categorical heuristic groups shown as points without artificial continuous interpolation.'])
    fig,axs=plt.subplots(2,1,figsize=(5.5,6.2));fig.subplots_adjust(left=.15,right=.96,top=.96,bottom=.12,hspace=.75)
    x=np.arange(5);w=.35
    for split,label,col,off,hatch in [('hans_all','HANS',GREEN,-w/2,''),('anli_dev_r123','ANLI dev',PURPLE,w/2,'///')]:
        g=df[df.split==split].groupby('condition').brier.agg(['mean','std','count']).reindex(order)
        axs[0].bar(x+off,g['mean'],w,yerr=1.96*g['std'].fillna(0)/np.sqrt(g['count']),color=col,hatch=hatch,capsize=2,label=label)
    axs[0].set(xticks=x,xticklabels=names,ylabel='Brier risk (bounded)',title='(a) Covered versus external shift')
    external(axs[0],y=-.35)
    piv=df[df.split=='anli_dev_r123'].pivot(index='seed',columns='condition',values='brier')
    gaps=[(piv[c]-piv.factual).to_numpy() for c in order[1:]]
    axs[1].bar(range(4),[v.mean() for v in gaps],yerr=[1.96*v.std(ddof=1)/np.sqrt(len(v)) for v in gaps],color=colors[1:],capsize=2)
    axs[1].axhline(0,color=GRAY,lw=.8)
    axs[1].set(xticks=range(4),xticklabels=names[1:],ylabel='ANLI Brier − factual',title='(b) External-shift gap')
    for ax in axs:axes_style(ax)
    finish(fig,'experiments/hans_suite/outputs/fig_hans_external.pdf',[src],
        ['All five conditions, original five training seeds and original mean/1.96*SE bars retained. Negative external results unchanged.'])


def qwen_paired():
    src=source(LEGACY/'experiments/comparative_utility/qwen_paired_results/qwen_paired_summary.csv')
    df=pd.read_csv(src);fig,axs=plt.subplots(1,2,figsize=(5.5,3.25));fig.subplots_adjust(left=.15,right=.97,top=.86,bottom=.29,wspace=.63)
    for cond,color,marker in [('Wlow','#E69F00','o'),('Wstar',BLUE,'s')]:
        cr=df[(df.condition==cond)&(df.split=='calibration')].sort_values('pool');tr=df[(df.condition==cond)&(df.split=='test')].sort_values('pool')
        assert list(cr.pool)==list(tr.pool)
        axs[0].scatter(cr.unadjusted_U_rho,tr.delta_hat,color=color,marker=marker,s=26,label=cond,facecolors='none')
        axs[1].plot(tr.pool,tr.sigma_a_hat,color=color,marker=marker,ls='none',label=cond,markerfacecolor='none')
    axs[0].axvline(0,color=GRAY,ls='--',lw=.7);axs[0].axhline(0,color=GRAY,ls=':',lw=.7)
    axs[0].set(xlabel='Calibration upper endpoint\n(unadjusted)',ylabel='Independent-test risk difference',title='(a) Audit versus test')
    axs[1].set(xlabel='Training pool ID',ylabel='Test anchor variance',xticks=range(8),title='(b) Anchor component')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.04),ncol=2,frameon=False)
    for ax in axs:axes_style(ax)
    finish(fig,'experiments/comparative_utility/qwen_paired_results/fig_qwen_paired_audit.pdf',[src],
           ['Same pool-matched points; zero endpoint kept; no line across independent pool IDs; calibration identity explicit.'])


def p0():
    sources=[source(LEGACY/'experiments/shift_granularity/p0_outputs'/f) for f in ['p0_aggregate.csv','p0_all_records.csv','p0_manifest.json']]
    agg=pd.read_csv(sources[0]);raw=pd.read_csv(sources[1])
    fig=plt.figure(figsize=(5.5,5.9));gs=fig.add_gridspec(2,2,hspace=.88,wspace=.6)
    axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    fig.subplots_adjust(left=.15,right=.96,top=.94,bottom=.15)
    for j,(cl,label,color,mark) in enumerate([('global','Global',BLUE,'o'),('group','Label-group',ORANGE,'s'),('task','Per-anchor',GREEN,'^')]):
        rr=agg[(agg['class']==cl)&np.isclose(agg.rho,.25)]
        means=rr.groupby(['size','task','condition'])[['estimate','certificate']].mean()
        axs[0].scatter(means.estimate,means.certificate,s=34,marker=mark,edgecolors=color,facecolors='none',lw=1,label=label)
        axs[1].bar(j,means.certificate.mean(),color=color,hatch=['','///','..'][j])
    lim=axs[0].get_xlim();axs[0].plot(lim,lim,ls=':',color=GRAY,lw=.8)
    axs[0].set(xlabel='Observed reference effect',ylabel='Certificate at ρ=0.25',title='(a) Certificate slack')
    axs[0].xaxis.set_major_locator(MaxNLocator(3));axs[0].yaxis.set_major_locator(MaxNLocator(4))
    axs[1].set(xticks=range(3),xticklabels=['Global','Label\ngroup','Per\nanchor'],ylabel='Mean certificate',title='(b) Shift-class comparison')
    external(axs[0],ncol=1,y=-.35,fontsize=8)
    # Original primary nuisance direction, exactly as declared in the manifest.
    tags=json.loads(sources[2].read_text())['nuisance_tags'];tag=tags[0]
    for i,(cond,label,color,mark) in enumerate([('cf_0','η=0%',BLUE,'o'),('cf_20','η=20%',ORANGE,'s'),('cf_40','η=40%',GREEN,'^')]):
        rr=raw[(raw.condition==cond)&(raw['class']=='global')&(raw['style']==tag)]
        series=rr.groupby('delta').actual_shift.mean().sort_index()
        axs[2].plot(series.index,series.to_numpy(),color=color,marker=mark,ls=['-','--',':'][i],ms=4,label=label)
    axs[2].axhline(0,color=GRAY,ls=':',lw=.7)
    axs[2].set(xlabel='Prespecified mixture shift δ',ylabel='Observed effect',title='(c) Held-out shift path')
    external(axs[2],ncol=3,y=-.34)
    for ax in axs:axes_style(ax)
    finish(fig,'experiments/shift_granularity/p0_outputs/fig_p0_shift_granularity.pdf',sources,
        ['Same condition/seed aggregation and prespecified direction as original.',
         'All three nearly coincident classes retained with open shapes; eta explicitly expressed in percent.'])


def boundaries():
    sources=[source(LEGACY/'experiments/shift_granularity/boundary_outputs'/f) for f in ['qwen_corruption_boundary.csv','support_escape_witness.csv','boundary_manifest.json']]
    df=pd.read_csv(sources[0]);witness=pd.read_csv(sources[1])
    fig,axs=plt.subplots(2,1,figsize=(5.5,6.3));fig.subplots_adjust(left=.15,right=.96,top=.95,bottom=.17,hspace=.8)
    for size,color in [('0p5b',BLUE),('1p5b',ORANGE)]:
        for task,mark,ls in [('attributes','o','-'),('relations','s','--')]:
            rr=df[(df['size']==size)&(df.task==task)&(df.condition!='factual')]
            ys=[];los=[];his=[]
            for eta in [0,.2,.4]:
                vals=rr[np.isclose(rr.eta,eta)].delta_vs_factual.to_numpy()
                boot=vals[np.random.default_rng(100+len(vals)).integers(len(vals),size=(2000,len(vals)))].mean(axis=1)
                ys.append(vals.mean());los.append(np.quantile(boot,.025));his.append(np.quantile(boot,.975))
            # Same deterministic seed-bootstrap statistic and draw count as original.
            axs[0].errorbar([0,.2,.4],ys,yerr=[np.array(ys)-los,np.array(his)-ys],color=color,marker=mark,ls=ls,capsize=2,label=size.replace('p','.')+', '+task)
    axs[0].axhline(0,color=GRAY,ls=':',lw=.7)
    axs[0].set(xticks=[0,.2,.4],xlabel='Screened rewrite corruption rate η',ylabel='Test loss change vs. factual',title='(a) Semantic corruption: locked Qwen predictions')
    external(axs[0],ncol=2,y=-.34)
    for j,response in enumerate([-1,-.5,0,.5,1]):
        r=witness[np.isclose(witness.off_support_response,response)].sort_values('gamma')
        axs[1].plot(r.gamma,r.actual_gap,color=[BLUE,'#56B4E9',GRAY,'#E69F00',ORANGE][j],ls=['-','--',':','-.','-'][j],marker=MARKS[j],markevery=2,ms=3,label='Response '+str(response))
    axs[1].plot([0,1],[0,0],color='#222222',ls=(0,(2,2)),lw=1.1,label='Reference certificate')
    axs[1].set(xlabel='Mass on unseen style γ',ylabel='Actual paired loss gap',title='(b) Support escape: exact analytic witness')
    external(axs[1],ncol=3,y=-.34,fontsize=7.8)
    for ax in axs:axes_style(ax)
    finish(fig,'experiments/shift_granularity/boundary_outputs/fig_failure_boundaries.pdf',sources,
        ['Same three-seed bootstrap bands, 2000 resamples and deterministic seed as supplied script; no retraining.',
         'Empirical and analytic panels explicitly distinguished; original zero-response witness retained.'])


def dashboard():
    paths=['confirmatory_v1/lora_metrics.json','decision_results/decision_summary.csv',
           'crossed_consistency_v4/results.json','transfer_tightness/metrics.json']
    sources=[source(LEGACY/'experiments/llm_cfpt'/p) for p in paths]
    directions=json.loads(sources[0].read_text())['rows'];dec=pd.read_csv(sources[1])
    inter=json.loads(sources[2].read_text());slack=json.loads(sources[3].read_text())['qwen']
    fig,axs=plt.subplots(2,2,figsize=(5.5,6.7));fig.subplots_adjust(left=.14,right=.97,top=.94,bottom=.205,hspace=1.05,wspace=.6)
    ax=axs[0,0];by={}
    for r in directions:by.setdefault(r['seed'],{})[r['condition']]=r['action_risk']
    rng=np.random.default_rng(719)
    for i,(name,col) in enumerate([('clean',BLUE),('random',GRAY),('oppose',GREEN),('reinforce',ORANGE)]):
        v=np.array([by[s][name]-by[s]['factual'] for s in sorted(by)])
        ax.scatter(i+rng.uniform(-.085,.085,len(v)),v,s=16,color=col,alpha=.7)
        half=2.365*v.std(ddof=1)/np.sqrt(len(v))
        ax.errorbar(i,v.mean(),yerr=half,color='#222222',fmt='o',ms=4,mfc='white',capsize=2)
    ax.axhline(0,color=GRAY,ls='--',lw=.7)
    ax.set(xticks=range(4),xticklabels=['Clean','Random','Oppose','Reinforce'],ylabel='Action-risk change vs. factual',title='(a) Matched error rates')
    ax.tick_params(axis='x',rotation=30)
    ax.text(.5,-.37,'Corruptions: 20%; 8 paired pools\nMean and 95% t interval',transform=ax.transAxes,ha='center',va='top',fontsize=7.8)
    ax=axs[0,1]
    for j,(method,label,col) in enumerate([('direct','Direct target-law','#222222'),('full','Directional',BLUE),('quadratic','Quadratic-only','#E69F00'),('rate_only','Rate-only',GRAY)]):
        rr=dec[dec.method==method].sort_values('budget')
        ax.plot(rr.budget,rr.mae,color=col,marker=MARKS[j],ms=3,ls=['-','--','-.',':'][j],label=label)
    ax.set(xscale='log',xticks=[50,100,200,400,800],xticklabels=['50','100','200','400','800'],xlabel='Clean calibration labels',ylabel='Risk-difference MAE',title='(b) Decision prediction')
    external(ax,ncol=1,y=-.31,fontsize=8)
    ax=axs[1,0]
    specs=[('KL ×100',inter['contrasts']['calibration']['Wstar-W0']['KL'],100,BLUE),
           ('G ×1000',inter['contrasts']['test']['Wstar-W0']['G'],1000,GREEN),
           ('Risk ×100',inter['contrasts']['test']['Wstar-U0']['Rnu'],100,ORANGE)]
    for y,(name,record,scale,col) in enumerate(specs):
        mean=record['difference']*scale;upper=record['upper95']*scale
        ax.plot([mean,upper],[y,y],color=col,lw=1.3)
        ax.plot(mean,y,'o',color=col,mfc='white',ms=4)
        ax.plot(upper,y,'|',color=col,ms=9,mew=1.3)
    ax.axvline(0,color=GRAY,ls='--',lw=.7)
    # Risk-only NI margin, do not draw a universal vertical threshold for all metrics.
    ax.plot(1,2,marker='D',mfc='none',mec=ORANGE,ms=4)
    ax.set(yticks=range(3),yticklabels=[s[0] for s in specs],ylim=(-.5,2.5),xlim=(-7.6,2),
           xlabel='Scaled contrast (negative improves)',title='(c) Intervention endpoints')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.text(.5,-.37,'Point: estimate; cap: 95% upper\nDiamond: risk NI margin = 0.01',transform=ax.transAxes,ha='center',va='top',fontsize=7.5)
    ax=axs[1,1];xx=np.arange(len(slack));w=.25
    for j,(key,label,col,hatch) in enumerate([('cancellation','Sign coherence',BLUE,''),('heterogeneity','Magnitude homogeneity','#E69F00','///'),('utilization','Overall utilization',GREEN,'..')]):
        ax.bar(xx+(j-1)*w,[r[key] for r in slack],width=w,color=col,hatch=hatch,label=label)
    ax.set(xticks=xx,xticklabels=[str(r['alpha']) for r in slack],ylim=(0,1),xlabel='Head perturbation α',ylabel='Fraction of bound used',title='(d) Transfer slack')
    external(ax,ncol=1,y=-.32,fontsize=7.8)
    for ax in axs.flat:axes_style(ax)
    finish(fig,'experiments/llm_cfpt/real_llm_evidence/fig_real_llm_evidence.pdf',sources,
        ['Original four diagnostics retained; all legends and explanations outside plotting areas.',
         'Corrected source-plot bug: missing lower95 had been replaced by an already-scaled mean and scaled a second time.',
         'Source provides one-sided upper95, so plot now shows estimate-to-upper only, never fabricated lower endpoints.',
         'NI margin attached only to reference-risk row; different metric scaling stays explicit.'])


def render_staged():
    rows=initialize()['figures'];checks=[]
    for row in rows:
        if row['path'] not in dict(eligible()):continue
        p=STAGE/row['path']
        if not p.exists():continue
        d=fitz.open(str(p));page=d[0]
        dest=QA/'renders'/(row['id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
        page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(str(dest))
        # Final width fraction exactly as currently declared in appendix.
        scale=396*dict(eligible())[row['path']]/page.rect.width
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
        small=[{'text':s['text'],'pt':round(s['size']*scale,2)} for s in spans if s['size']*scale<7-.01]
        checks.append({'path':row['path'],'id':row['id'],'page_size_pt':list(page.rect)[2:],
                       'placement_scale':scale,'small_spans':small,'sha256':sha(p),
                       'minimum_font_pt':min(s['size']*scale for s in spans),
                       'svg_text_nodes':len(re.findall('<text',p.with_suffix('.svg').read_text(encoding='utf8'))),
                       'visual_status':'needs explicit rendered-image inspection'})
        d.close()
    (QA/'staged_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf8')
    print('RENDERED',len(checks),'figures; see staged_checks.json',flush=True)


SVGNS='http://www.w3.org/2000/svg'
ET.register_namespace('',SVGNS)
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')


def tag(name):return '{'+SVGNS+'}'+name


def svg_document(defs,groups,width,height):
    root=ET.Element(tag('svg'),{'width':str(width)+'pt','height':str(height)+'pt',
         'viewBox':'0 0 {} {}'.format(width,height),'version':'1.1'})
    root.append(copy.deepcopy(defs))
    for group in groups:root.append(copy.deepcopy(group))
    return root


def svg_pdf(root):
    svg=fitz.open(stream=ET.tostring(root,encoding='utf8'),filetype='svg')
    data=svg.convert_to_pdf();svg.close()
    return data


def graphics_bbox(defs,group,width,height):
    # Graphics layout bounds only. No plotted x/y values are read or inferred.
    doc=fitz.open(stream=svg_pdf(svg_document(defs,[group],width,height)),filetype='pdf')
    boxes=[fitz.Rect(box) for kind,box in doc[0].get_bboxlog() if kind!='ignore-text']
    rect=fitz.Rect()
    for box in boxes:rect|=box
    doc.close()
    return rect


def panel_bbox(defs,group,width,height):
    # Some SVG consumers do not apply clipping when reporting drawing bounds.
    # Only axes background + text are used for layout: never un-clipped data paths.
    safe=ET.Element(tag('g'))
    first=next((n for n in group if n.tag==tag('path') and n.get('fill')=='#ffffff'),None)
    if first is not None:safe.append(copy.deepcopy(first))
    def text_only(node):
        if node.tag==tag('text'):return copy.deepcopy(node)
        if node.tag!=tag('g'):return None
        result=ET.Element(node.tag,dict(node.attrib))
        for child in node:
            added=text_only(child)
            if added is not None:result.append(added)
        return result if len(result) else None
    for n in group:
        textnode=text_only(n)
        if textnode is not None:safe.append(textnode)
    box=graphics_bbox(defs,safe,width,height)
    return fitz.Rect(box.x0-3,box.y0-3,box.x1+3,box.y1+3)


def axis_rect(group,height):
    first=next(n for n in group if n.tag==tag('path') and n.get('fill')=='#ffffff')
    coords=list(map(float,re.findall(r'[\d.]+',first.get('d'))))
    x0,y0,x1,y1=coords[:4]
    return fitz.Rect(x0,height-y1,x1,height-y0)


def panel_extents(panels,boxes,height):
    rects=[axis_rect(p,height) for p in panels]
    left=max(r.x0-b.x0 for r,b in zip(rects,boxes))
    right=max(b.x1-r.x1 for r,b in zip(rects,boxes))
    top=max(r.y0-b.y0 for r,b in zip(rects,boxes))
    bottom=max(b.y1-r.y1 for r,b in zip(rects,boxes))
    return rects,left,right,top,bottom


def chrome_pdf(svg_path,pdf_path,width,height):
    """Chrome's vector print path preserves SVG clip paths, images, and live text."""
    printable=QA/'print'/(pdf_path.stem+'.html');printable.parent.mkdir(parents=True,exist_ok=True)
    markup=svg_path.read_text(encoding='utf8').split('?>')[-1]
    html='<!doctype html><meta charset="utf-8"><style>@page{size:'+str(width)+'pt '+str(height)+'pt;margin:0}html,body{margin:0;padding:0;width:'+str(width)+'pt;height:'+str(height)+'pt}svg{display:block}</style>'+markup
    printable.write_text(html,encoding='utf8')
    chrome=Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
    profile=QA/'chrome_profile';profile.mkdir(exist_ok=True)
    args=[str(chrome),'--headless','--disable-gpu','--disable-background-networking',
          '--disable-extensions','--no-first-run','--no-default-browser-check',
          '--no-pdf-header-footer','--print-to-pdf-no-header',
          '--user-data-dir='+str(profile),'--print-to-pdf='+str(pdf_path),printable.as_uri()]
    run=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if run.returncode or not pdf_path.exists():
        raise RuntimeError('Chrome PDF failed: '+run.stderr.decode('utf8',errors='replace')[-1000:])
    doc=fitz.open(str(pdf_path));assert len(doc)==1, 'Unexpected printed page count';doc.close()


def text_clusters(group):
    # Mathematical labels can be split across adjacent text nodes sharing a baseline.
    clusters=[];last=None
    for node in list(group):
        if node.tag!=tag('text'):
            last=None;continue
        transform=node.get('transform','')
        if last is None or transform!=last[0]:
            last=(transform,[]);clusters.append(last[1])
        last[1].append(node)
    return clusters


def boost_cluster(cluster,factor,anchor='center'):
    positions=[]
    for node in cluster:
        for span in node:
            positions.extend(float(x) for x in span.get('x','0').split())
    center=(min(positions) if anchor=='left' else (min(positions)+max(positions))/2) if positions else 0
    for node in cluster:
        node.set('font-size',str(float(node.get('font-size'))*factor))
        for span in node:
            if span.get('x'):
                span.set('x',' '.join('{:.7g}'.format(center+(float(x)-center)*factor) for x in span.get('x').split()))
            if span.get('y'):
                span.set('y',' '.join('{:.7g}'.format(float(y)*factor) for y in span.get('y').split()))


def geometry_signature(group):
    # Path geometry and image bytes/geometry are invariant under this repair.
    values=[]
    for n in group.iter():
        if n.tag==tag('path'):values.append(('path',n.get('d'),n.get('transform')))
        elif n.tag==tag('image'):values.append(('image',tuple(sorted(n.attrib.items()))))
    return hashlib.sha256(repr(values).encode('utf8')).hexdigest(),len(values)


VECTOR_SPECS={
 'F06':{'legends':[(54,61)],'note':'Power axis zoom: 0.5–1.0; original bands retained.'},
 'F17':{'legends':[(222,235)]},
 'F18':{'legends':[],'order':[0,3,1,4,2,5]},
 'F19':{'legends':[(106,111),(209,214)],'dedup':True,
        'note':'Distinct budget units: anchors / paired evaluations.'},
 'F21':{'legends':[(63,84)]},
 'F22':{'legends':[(115,122)],'note':'Left: measured power. Right: analytic population standard error.'},
 'F07':{'legends':[(39,55)]},
 'F16':{'legends':[(51,62)]},
 'F28':{'legends':[(59,75)]},
 'F29':{'legends':[(133,154)]},
 'F23':{'legends':[(48,61)],'note':'Discrete policies/sizes; lines are visual guides.'},
}


def embed_dejavu(defs,root):
    from matplotlib.font_manager import FontProperties,findfont
    variants=set()
    for n in root.iter():
        family=n.get('font-family','')
        if family.startswith('DejaVu'):
            variants.add((family,n.get('font-style','normal'),n.get('font-weight','normal')))
    if not variants:return
    rules=[]
    for family,fontstyle,weight in sorted(variants):
        font=Path(findfont(FontProperties(family=family,style=fontstyle,weight=weight)))
        encoded=base64.b64encode(font.read_bytes()).decode('ascii')
        rules.append("@font-face{font-family:'%s';font-style:%s;font-weight:%s;src:url(data:font/ttf;base64,%s) format('truetype')}"%(family,fontstyle,weight,encoded))
    style_node=ET.SubElement(defs,tag('style'));style_node.text='\n'.join(rules)


def redundant_lines(root,fid):
    palettes={
      'F06':{'#777777':'4,2','#c44036':'1,2'},
      'F17':{'#4077a5':'5,2','#46815a':'1,2'},
      'F21':{'#416a98':'5,2','#696969':'1,2','#8b5c91':'5,2,1,2','#ce8d35':'8,2'},
      'F29':{'#9467bd':'5,2','#1f77b4':'1,2','#ff7f0e':'5,2,1,2','#2ca02c':'8,2'},
      'F23':{'#416a98':'5,2','#696969':'1,2'},
      'F19':{'#4077a5':'5,2'},
    }
    palette=palettes.get(fid,{})
    for n in root.iter(tag('path')):
        color=n.get('stroke');data=n.get('d','')
        if color in palette and n.get('fill')=='none' and 'C' not in data and float(n.get('stroke-width','0'))>=1.2:
            n.set('stroke-dasharray',palette[color])


def vector_reflow():
    """No digitization: relocate existing SVG groups, with exact data path strings."""
    for row in initialize()['figures']:
        fid=row['id'];path=row['path']
        if fid not in VECTOR_SPECS or path not in dict(eligible()):continue
        spec=VECTOR_SPECS[fid];src=QA/'originals'/path
        doc=fitz.open(str(src));width=doc[0].rect.width;height=doc[0].rect.height
        root=ET.fromstring(doc[0].get_svg_image(text_as_path=False));doc.close()
        nodes=list(root);defs=nodes[0]
        assert defs.tag==tag('defs')
        embed_dejavu(defs,root)
        redundant_lines(root,fid)
        axes=[]
        for i,n in enumerate(nodes):
            if n.tag==tag('path') and n.get('fill')=='#ffffff' and re.fullmatch(r'M[\d.]+ [\d.]+H[\d.]+V[\d.]+H[\d.]+Z',n.get('d','')):
                if i>1:axes.append(i)
        assert axes, fid+' no original axes recognized'
        excluded={i for lo,hi in spec['legends'] for i in range(lo,hi)}
        panels=[];signatures=[]
        for start,stop in zip(axes,axes[1:]+[len(nodes)]):
            group=ET.Element(tag('g'),{'id':'original-axis-'+str(len(panels))})
            for i in range(start,stop):
                if i not in excluded:group.append(copy.deepcopy(nodes[i]))
            panels.append(group);signatures.append(geometry_signature(group))
        # Heatmap shared-axis tick labels are copied as graphic text labels, not data.
        if fid=='F18':
            for col in range(3):
                g=ET.SubElement(panels[col],tag('g'),{'transform':'translate({} {})'.format(179.99*col,-166.82812)})
                for idx in [135,138,141,144,145,146]:g.append(copy.deepcopy(nodes[idx]))
                if col:
                    for dest,indices in [(col,[14,17,20,23,24,25]),(col+3,[149,152,155,158,159,160])]:
                        g=ET.SubElement(panels[dest],tag('g'),{'transform':'translate({} 0)'.format(179.99*col)})
                        for idx in indices:g.append(copy.deepcopy(nodes[idx]))
        legend_entries=[]
        for k,(start,stop) in enumerate(spec['legends']):
            if k and spec.get('dedup'):continue
            entry=ET.Element(tag('g'))
            # First node is the legend's rounded white background. It is not data.
            for i in range(start+1,stop):
                entry.append(copy.deepcopy(nodes[i]))
                if nodes[i].tag==tag('text'):
                    legend_entries.append(entry);entry=ET.Element(tag('g'))
            assert not len(entry),fid+' incomplete legend entry'
        # Fit panels first, then increase labels if final manuscript placement needs it.
        ncol=1 if len(panels)==1 else 2
        cellwidth=(396-24-12*(ncol-1))/ncol
        scale=1.
        for iteration in range(4):
            boxes=[panel_bbox(defs,g,width,height) for g in panels]
            rects,left,right,top,bottom=panel_extents(panels,boxes,height)
            scale=cellwidth/(left+max(r.width for r in rects)+right)
            changed=False
            for g in panels:
                for cluster in text_clusters(g):
                    fs=max(float(n.get('font-size')) for n in cluster)
                    smallest=min(float(n.get('font-size')) for n in cluster)
                    ratio=max(7.7/(fs*scale*row['width_fraction']),
                              7.15/(smallest*scale*row['width_fraction']))
                    if ratio>1.002:boost_cluster(cluster,ratio);changed=True
            if not changed:break
        boxes=[panel_bbox(defs,g,width,height) for g in panels]
        rects,left,right,top,bottom=panel_extents(panels,boxes,height)
        scale=cellwidth/(left+max(r.width for r in rects)+right)
        rowheight=(top+max(r.height for r in rects)+bottom)*scale
        for entry in legend_entries:
            for cluster in text_clusters(entry):
                fs=max(float(n.get('font-size')) for n in cluster)
                boost_cluster(cluster,(8.2/row['width_fraction'])/fs,anchor='left')
        legend_boxes=[graphics_bbox(defs,g,width,height) for g in legend_entries]
        legend_cols=2 if len(legend_entries)>3 else max(1,len(legend_entries))
        max_leg=max([b.width for b in legend_boxes]+[0])
        if max_leg*legend_cols+14*(legend_cols-1)>370:legend_cols=2 if max_leg*2+14<370 else 1
        legrowheight=max([b.height for b in legend_boxes]+[0])+6
        nrows=int(np.ceil(len(panels)/ncol));gap=20
        plot_height=nrows*rowheight+(nrows-1)*gap
        legend_height=int(np.ceil(len(legend_entries)/max(1,legend_cols)))*legrowheight
        outheight=12+plot_height+14+legend_height+8+(20 if spec.get('note') else 0)
        output=svg_document(defs,[],396,outheight)
        order=spec.get('order',list(range(len(panels))))
        for dest,idx in enumerate(order):
            box=boxes[idx];rect=rects[idx]
            x=12+(dest%ncol)*(cellwidth+12)+scale*(left-(rect.x0-box.x0))
            y=12+(dest//ncol)*(rowheight+gap)+scale*(top-(rect.y0-box.y0))
            wrapper=ET.SubElement(output,tag('g'),{'id':'panel-'+str(idx),
                'transform':'translate({:.7g} {:.7g}) scale({:.7g}) translate({:.7g} {:.7g})'.format(x,y,scale,-box.x0,-box.y0)})
            wrapper.append(panels[idx])
            assert geometry_signature(panels[idx])==signatures[idx],fid+' data geometry changed'
        for i,(g,b) in enumerate(zip(legend_entries,legend_boxes)):
            x=12+(i%legend_cols)*(372/legend_cols);y=12+plot_height+14+(i//legend_cols)*legrowheight
            wrap=ET.SubElement(output,tag('g'),{'id':'external-legend-'+str(i),
                 'transform':'translate({:.7g} {:.7g})'.format(x-b.x0,y-b.y0)})
            wrap.append(g)
        if spec.get('note'):
            node=ET.SubElement(output,tag('text'),{'x':'198','y':str(outheight-8),'text-anchor':'middle',
                 'font-size':str(7.8/row['width_fraction']),'font-family':'Arial','fill':'#333333'})
            node.text=spec['note']
        dest=STAGE/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.with_suffix('.svg').write_bytes(ET.tostring(output,encoding='utf8',xml_declaration=True))
        chrome_pdf(dest.with_suffix('.svg'),dest,396,outheight)
        LEDGER[path]={'mode':'lossless data-vector reflow; empirical data unavailable',
            'sources':[{'path':str(src.relative_to(ROOT)).replace('\\','/'),'sha256':sha(src)}],
            'notes':['No empirical coordinates digitized or reconstructed.',
                     'All original data path d/transform strings and embedded heatmap images retained unchanged.',
                     'Legend graphics removed from plotting layer and moved to an exterior strip; text expanded independently.',
                     'Source SVG has live text and native vector groups; existing heatmaps remain embedded originals.'],
            'geometry_signatures':signatures,'n_panels':len(panels),
            'original_legend_ranges':spec['legends'],'final_font_target_pt':7.7,
            'status':'staged; manual rendered-image inspection required'}
        print('REFLOWED',fid,'panels',len(panels),'height',round(outheight,1),flush=True)


def qa_bundle():
    """Build review artifacts; this function never marks a visual PASS."""
    rows=[r for r in initialize()['figures'] if r['path'] in dict(eligible())]
    assert all((STAGE/r['path']).exists() for r in rows)
    sheets=QA/'contact_sheets';sheets.mkdir(exist_ok=True)
    graydir=QA/'grayscale';graydir.mkdir(exist_ok=True)
    olddir=QA/'before_renders';olddir.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    proof=fitz.open();cards=[];captions=[]
    tex=(ROOT/'appendix.tex').read_text(encoding='utf8')
    def braced(text,start):
        depth=1;pos=start
        while pos<len(text) and depth:
            if text[pos]=='{' and (pos==0 or text[pos-1]!='\\'):depth+=1
            if text[pos]=='}' and (pos==0 or text[pos-1]!='\\'):depth-=1
            pos+=1
        return text[start:pos-1]
    for r in rows:
        fid=r['id'];rel=r['path'];doc=fitz.open(str(STAGE/rel));page=doc[0]
        Image.open(QA/'renders'/(fid+'.png')).convert('L').save(graydir/(fid+'.png'))
        with fitz.open(str(QA/'originals'/rel)) as old:
            old[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(str(olddir/(fid+'.png')))
        p=proof.new_page(width=612,height=792)
        p.insert_text((36,26),fid+' | current appendix placement | 5.5-inch text block',fontsize=10)
        p.insert_textbox(fitz.Rect(36,34,576,62),rel,fontsize=8)
        w=396*dict(eligible())[rel];h=page.rect.height*w/page.rect.width
        p.show_pdf_page(fitz.Rect((612-w)/2,76,(612+w)/2,76+h),doc,0)
        p.insert_text((36,768),'Standalone size proof, not a LaTeX float-placement approval.',fontsize=8)
        doc.close()
        position=tex.index('{'+rel+'}')
        begin=tex.rfind(r'\begin{figure',0,position)
        end=tex.index(r'\end{figure',position)
        block=tex[begin:end];cm=re.search(r'\\caption(?:\[[^\]]*\])?\{',block)
        caption=braced(block,cm.end()) if cm else ''
        lm=re.search(r'\\label\{([^}]+)\}',block)
        captions.append({'id':fid,'path':rel,'include_line':tex[:position].count('\n')+1,
                         'label':lm.group(1) if lm else None,'caption_tex':caption,
                         'width_fraction':dict(eligible())[rel]})
        cards.append('<section><h2>'+fid+' '+html.escape(rel)+'</h2>'
            '<p><a href="stage/'+rel+'">PDF</a> | <a href="stage/'+rel[:-4]+'.svg">editable SVG</a>'
            ' | <a href="originals/'+rel+'">original PDF backup</a></p>'
            '<div class="pair"><figure><img src="before_renders/'+fid+'.png"><figcaption>Before</figcaption></figure>'
            '<figure><img src="renders/'+fid+'.png"><figcaption>Repaired</figcaption></figure></div>'
            '<details><summary>Current unchanged TeX caption</summary><pre style="white-space:pre-wrap">'+html.escape(caption)+'</pre></details></section>')
    proof.save(str(QA/'actual_size_proof.pdf'));proof.close()
    for start in range(0,len(rows),4):
        for gray in [False,True]:
            canvas=Image.new('RGB',(1240,1840),'white');draw=ImageDraw.Draw(canvas)
            for slot,r in enumerate(rows[start:start+4]):
                x=(slot%2)*620;y=(slot//2)*920
                draw.text((x+12,y+10),r['id']+'  '+Path(r['path']).name,fill='black',font=font)
                im=Image.open((graydir if gray else QA/'renders')/(r['id']+'.png')).convert('RGB')
                im.thumbnail((594,858))
                canvas.paste(im,(x+12,y+48))
            canvas.save(sheets/('{}_{:02d}.png'.format('gray' if gray else 'color',start//4+1)))
    index='<!doctype html><meta charset="utf-8"><title>Appendix figure repair QA</title>'
    index+='<style>body{font:15px Arial;margin:24px}section{border-top:1px solid #aaa;padding:15px 0}.pair{display:flex;align-items:flex-start;gap:20px}figure{width:48%;margin:0}img{max-width:100%;height:auto}h2{font-size:16px}</style>'
    index+='<h1>22 retained appendix figures: before / repaired</h1><p>Actual rendered PDFs. Historical audit IDs are not final manuscript figure numbers.</p>'
    index+='<p><a href="actual_size_proof.pdf">Actual-size proof PDF</a> | <a href="staged_checks.json">Font/size checks</a> | <a href="repair_manifest.json">Provenance ledger</a></p>'
    (QA/'index.html').write_text(index+''.join(cards),encoding='utf8')
    (QA/'current_caption_inventory.json').write_text(json.dumps(captions,indent=2,ensure_ascii=False),encoding='utf8')
    print('QA_BUNDLE',len(rows),'figures; grayscale and actual-size proofs ready',flush=True)


def approve_reviewed(ids):
    """Explicit human/model visual review acknowledgement bound to artifact bytes."""
    checks=json.loads((QA/'staged_checks.json').read_text(encoding='utf8'))
    assert set(ids)=={r['id'] for r in checks},'Review must cover every current figure'
    assert {r['path'] for r in checks}==set(dict(eligible()))
    for r in checks:
        assert sha(STAGE/r['path'])==r['sha256'],'Stage changed after rendering'
        assert not r['small_spans'],'Font gate failed: '+r['id']
        r['svg_sha256']=sha((STAGE/r['path']).with_suffix('.svg'))
        r['visual_status']='PASS: rendered color figure manually inspected; no severe clipping/legend/text defect'
        if r['id'] in ('F07','F16','F28'):
            r['grayscale_status']='Partial: color-only identity remains in original scatter glyphs; original data-vector geometry preserved'
        else:
            r['grayscale_status']='Inspected; line/marker/hatch or spatial category cues retained'
    record={'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'figures':checks,
            'scope':'Standalone figures at actual appendix width; not a compiled manuscript layout approval'}
    (QA/'reviewed_artifacts.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8')


def protected_now():
    paths=['main.tex','appendix.tex','figures/intro.pdf','figures/audit_evidence.pdf']
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/oral_planner_v6').rglob('*') if p.is_file()]
    return {p:sha(ROOT/p) for p in paths if (ROOT/p).is_file()}


def install_reviewed():
    """Replace only current, unchanged canonical targets with reviewed artifacts."""
    review=json.loads((QA/'reviewed_artifacts.json').read_text(encoding='utf8'))
    rows=review['figures'];current=dict(eligible())
    assert set(current)=={r['path'] for r in rows},'Appendix references changed: recheck scope'
    original={r['path']:r for r in initialize()['figures']}
    protected=protected_now()
    # Validate every target before performing any replacement.
    for r in rows:
        rel=Path(r['path']);dest=(ROOT/rel).resolve()
        assert rel.parts[0] in ('experiments','figures') and '..' not in rel.parts
        assert ROOT.resolve() in dest.parents and 'oral_planner_v6' not in rel.parts
        assert current[r['path']]==original[r['path']]['width_fraction'],'Placement changed: rerender'
        assert sha(STAGE/rel)==r['sha256']
        assert sha((STAGE/rel).with_suffix('.svg'))==r['svg_sha256']
        assert sha(dest) in (original[r['path']]['before_sha256'],r['sha256']),'Canonical changed by another writer: '+r['path']
        assert sha(QA/'originals'/rel)==original[r['path']]['before_sha256']
        if dest.with_suffix('.svg').exists():
            oldsvg=(QA/'originals'/rel).with_suffix('.svg')
            acceptable=[r['svg_sha256']]+([sha(oldsvg)] if oldsvg.exists() else [])
            assert sha(dest.with_suffix('.svg')) in acceptable,'Unowned SVG exists: '+str(rel)
    installed=[]
    for r in rows:
        dest=ROOT/r['path']
        for ext in ('.pdf','.svg'):
            shutil.copy2(str((STAGE/r['path']).with_suffix(ext)),str(dest.with_suffix(ext)))
        assert sha(dest)==r['sha256'] and sha(dest.with_suffix('.svg'))==r['svg_sha256']
        installed.append({'path':r['path'],'id':r['id'],'before_sha256':original[r['path']]['before_sha256'],
                          'installed_pdf_sha256':sha(dest),'installed_svg_sha256':sha(dest.with_suffix('.svg'))})
    after=protected_now()
    assert protected==after,'Protected file changed during install; inspect external concurrent writer'
    (QA/'installed_manifest.json').write_text(json.dumps({'installed_at_utc':datetime.now(timezone.utc).isoformat(),
        'figures':installed,'protected_before_install':protected,'protected_after_install':after,
        'canonical_tex_written':False,'code_directory_written':False},indent=2),encoding='utf8')
    ledgerfile=QA/'repair_manifest.json'
    ledger=json.loads(ledgerfile.read_text(encoding='utf8'))
    for r in rows:
        ledger[r['path']].update({'status':r['visual_status'],'grayscale_status':r['grayscale_status'],
            'minimum_font_pt_at_current_placement':r['minimum_font_pt'],'installed_pdf_sha256':r['sha256'],
            'installed_svg_sha256':r['svg_sha256']})
    ledgerfile.write_text(json.dumps(ledger,indent=2,ensure_ascii=False),encoding='utf8')
    print('INSTALLED',len(installed),'PDF + SVG pairs; protected files unchanged during install',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',default='analytic,matched,cluster,hans,qwen_paired')
    parser.add_argument('--approve-reviewed',help='Comma-separated IDs, only after explicit visual inspection; no auto-approval')
    args=parser.parse_args()
    initialize();style()
    registry={'analytic':analytic,'matched':matched,'cluster':qwen_cluster,'hans':hans,'qwen_paired':qwen_paired,
              'p0':p0,'boundaries':boundaries,'dashboard':dashboard,'render':render_staged,'vectors':vector_reflow,
              'bundle':qa_bundle,'install':install_reviewed}
    for name in filter(None,args.only.split(',')):
        registry[name]()
        print('STAGED',name,flush=True)
    ledgerfile=QA/'repair_manifest.json'
    previous=json.loads(ledgerfile.read_text(encoding='utf8')) if ledgerfile.exists() else {}
    previous.update(LEDGER);ledgerfile.write_text(json.dumps(previous,indent=2,ensure_ascii=False),encoding='utf8')
    if args.approve_reviewed:approve_reviewed(args.approve_reviewed.split(','))


if __name__=='__main__':main()
