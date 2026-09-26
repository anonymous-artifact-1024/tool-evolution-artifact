"""Redraw frozen preexperiment tables; no model calls or statistical recomputation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

MODELS = ['Qwen3-14B', 'qwen3-max-2026-01-23']
LABELS = ['Qwen3-14B', 'Qwen3-Max']
COLORS = ['#237C83', '#7962A6']
MARKERS = ['o', 'D']
CONDS = ['A', 'B1', 'C1', 'D1', 'B2', 'C2', 'D2']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    names = ['table_1_condition_summary.csv', 'table_2_predeclared_contrasts.csv', 'table_4_service_block_costs.csv']
    tables = []
    for name in names:
        with (a.source_dir / name).open(encoding='utf-8-sig', newline='') as f:
            tables.append(list(csv.DictReader(f)))
    conditions, contrasts, costs = tables
    assert len(conditions) == 14 and len(contrasts) == 20 and len(costs) == 60
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial', 'DejaVu Sans'],
        'font.size':8.5, 'axes.labelsize':8.5, 'axes.titlesize':9.5,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
        'text.color':'#25313B', 'axes.labelcolor':'#25313B',
        'xtick.color':'#52606B', 'ytick.color':'#52606B',
        'axes.edgecolor':'#B8C0C7', 'axes.linewidth':0.6,
        'axes.spines.top':False, 'axes.spines.right':False,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'savefig.dpi':400, 'axes.axisbelow':True})
    def clean(ax):
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=7)
        ax.tick_params(axis='x', length=3)
        ax.grid(axis='x', color='#E9EDF0', linewidth=.55)
    def save(fig, name):
        for ext in ['pdf', 'png', 'svg']:
            fig.savefig(a.output_dir / (name+'.'+ext), facecolor='white')
        plt.close(fig)
    def legend(fig, y=.985):
        fig.legend([Line2D([], [], color=c, marker=m, lw=1.2, markersize=4)
                    for c,m in zip(COLORS,MARKERS)], LABELS,
                   loc='upper center', bbox_to_anchor=(.57,y), ncol=2,
                   frameon=False, columnspacing=2.4, handlelength=1.8)

    # Compact performance matrix: this figure answers "where are the high/low cells?"
    matrix=np.array([[float(next(r for r in conditions if r['model']==m and r['condition']==c)['mrr'])
                      for c in CONDS] for m in MODELS])
    fig,ax=plt.subplots(figsize=(7.2,2.55))
    cmap=LinearSegmentedColormap.from_list('saner_mrr',['#F4F7F6','#D8EBE8','#83BDB6','#237C83'])
    mesh=ax.pcolormesh(np.arange(8),np.arange(3),matrix,cmap=cmap,vmin=.30,vmax=.65,
                       edgecolors='white',linewidth=5,shading='flat')
    for row in range(2):
        baseline=matrix[row,0]
        for col in range(7):
            value=matrix[row,col]; delta=value-baseline
            color='white' if value>.55 else '#24313B'
            ax.text(col+.5,row+.44,f'{value:.3f}',ha='center',va='center',fontsize=11,
                    fontweight='bold',color=color)
            if col:
                ax.text(col+.5,row+.70,f'{delta:+.3f} vs A',ha='center',va='center',fontsize=6.8,
                        color=color,alpha=.9)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(7)+.5,['A\nBaseline','B1\nRefactor','C1\nExtend','D1\nBoth',
                                   'B2\nRefactor','C2\nExtend','D2\nBoth'])
    ax.set_yticks([.5,1.5],LABELS)
    ax.tick_params(length=0,pad=7)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.text(2.5,-.18,'SEARCH TOOLS',ha='center',va='bottom',fontsize=7.5,fontweight='bold',color='#5D6972')
    ax.text(5.5,-.18,'READ TOOLS',ha='center',va='bottom',fontsize=7.5,fontweight='bold',color='#5D6972')
    cax=fig.add_axes([.948,.35,.012,.38])
    cb=fig.colorbar(mesh,cax=cax,orientation='vertical',ticks=[.30,.45,.60])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7,length=2,pad=2)
    cb.set_label('MRR',fontsize=7,labelpad=3)
    fig.subplots_adjust(left=.13,right=.93,bottom=.30,top=.81)
    fig.text(.13,.055,'Cell: mean MRR; second line: difference from A. 20 tasks × 3 repetitions. Descriptive CIs remain in Table 1.',fontsize=7.3)
    save(fig,'figure_1_condition_performance')

    # One forest plot; color encodes model and marker shape encodes tool family.
    fig,ax=plt.subplots(figsize=(7.2,3.65))
    clean(ax); ax.axvspan(-.02,.02,color='#E8ECEF',zorder=0); ax.axvline(0,color='#85929B',lw=.8)
    ys=[4.8,3.8,2.45,1.45,.1]
    eqs={1:[f'C1 - A',f'D1 - B1',f'B1 - A',f'D1 - C1','(D1-C1) - (B1-A)'],
         2:[f'C2 - A',f'D2 - B2',f'B2 - A',f'D2 - C2','(D2-C2) - (B2-A)']}
    offsets={(0,1):.24,(1,1):.08,(0,2):-.08,(1,2):-.24}
    tool_markers={1:'o',2:'s'}
    for idx,model in enumerate(MODELS):
        for instance in (1,2):
            rows=[next(r for r in contrasts if r['model']==model and int(r['instance'])==instance and r['contrast']==eq)
                  for eq in eqs[instance]]
            for y,r in zip(ys,rows):
                pos=y+offsets[(idx,instance)]
                ax.plot([float(r['interval_lower']),float(r['interval_upper'])],[pos,pos],color=COLORS[idx],lw=1.15,alpha=.9)
                ax.plot(float(r['estimate']),pos,marker=tool_markers[instance],color=COLORS[idx],ms=4.4,zorder=3)
    for y in [3.18,.78]: ax.axhline(y,color='#DDE3E7',lw=.7)
    ax.set(xlim=(-.32,.32),ylim=(-.55,5.35),xlabel='MRR difference')
    ax.set_xticks([-.3,-.2,-.1,0,.1,.2,.3])
    ax.set_yticks(ys,[r'RQ1     $C_i-A$',r'            $D_i-B_i$',
                      r'RQ2     $B_i-A$',r'            $D_i-C_i$',
                      r'RQ3     $(D_i-C_i)-(B_i-A)$'])
    handles=[Line2D([],[],color=COLORS[0],lw=1.4,label=LABELS[0]),
             Line2D([],[],color=COLORS[1],lw=1.4,label=LABELS[1]),
             Line2D([],[],color='#59656E',marker='o',lw=0,label='Search ($i=1$)',markersize=4),
             Line2D([],[],color='#59656E',marker='s',lw=0,label='Read ($i=2$)',markersize=4)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.59,.975),ncol=4,frameon=False,
               handlelength=1.6,columnspacing=1.7)
    fig.subplots_adjust(left=.32,right=.97,bottom=.21,top=.82)
    fig.text(.32,.045,'Joint 95% max-t intervals · Gray band: ±0.02 · All 20 contrasts inconclusive.',fontsize=7.3)
    save(fig,'figure_2_predeclared_contrasts')

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.45),gridspec_kw={'width_ratios':[1,1]})
    ax=axes[0]; clean(ax)
    for j,c in enumerate(CONDS):
        y=6-j
        vals=[float(next(r for r in conditions if r['model']==m and r['condition']==c)['mean_generations']) for m in MODELS]
        ax.plot(vals,[y,y],color='#CDD5DB',lw=1.6)
        for k,v in enumerate(vals):
            ax.plot(v,y,MARKERS[k],color=COLORS[k],ms=4.5)
            ax.text(v+(-.65 if k==0 else .65),y,f'{v:.1f}',ha='right' if k==0 else 'left',va='center',fontsize=7.3,color=COLORS[k])
    ax.set(yticks=range(6,-1,-1),yticklabels=CONDS,xlim=(0,29),ylim=(-.6,6.6),xlabel='Mean generations per episode')
    ax.set_xticks([0,10,20,30])
    ax.set_title('(a)  Execution length',loc='left',fontweight='bold',pad=12)
    ax=axes[1]
    values=np.sort([float(r['nominal_cost_cny']) for r in costs])
    ax.step(np.r_[0,values],np.r_[0,np.arange(1,61)/60],where='post',color=COLORS[1],lw=1.5)
    ax.fill_between(np.r_[0,values],np.r_[0,np.arange(1,61)/60],step='post',color=COLORS[1],alpha=.055)
    ax.axvline(values.mean(),color=COLORS[1],ls=(0,(3,3)),lw=.85)
    ax.text(.96,.09,f'60 blocks\nMean  {values.mean():.2f} CNY\nMedian  {np.median(values):.2f} CNY',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,linespacing=1.6)
    ax.set(xlim=(0,16),ylim=(0,1.04),xlabel='Service cost per block (CNY)',ylabel='Cumulative fraction of blocks')
    ax.set_xticks([0,4,8,12,16]); ax.set_yticks([0,.25,.5,.75,1])
    ax.grid(color='#E9EDF0',lw=.55)
    ax.set_title('(b)  Service cost distribution',loc='left',fontweight='bold',pad=12)
    legend(fig)
    fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.79,wspace=.4)
    fig.text(.07,.06,'Seven conditions per block. Nominal logged-token cost; discounts and historical preflight excluded.',fontsize=7.3)
    save(fig,'figure_3_execution_and_cost')

    captions='''# Publication figure notes

These figures visualize the completed preexperiment only; they are not formal RQ1–RQ3 findings.

Figure 1. File-level MRR across seven conditions. Each cell shows the mean over 20 tasks after averaging three repetitions per task; the smaller line reports its descriptive difference from the shared baseline A. Search and read families are separated. B: interface refactoring; C: functional extension; D: both. Color encodes the same absolute MRR scale for both models. Descriptive task-level bootstrap intervals remain in Table 1 so the matrix can focus on the condition pattern.

Figure 2. Twenty predeclared MRR contrasts in one shared forest plot. RQ1 concerns extension, RQ2 refactoring, and RQ3 interaction. Color identifies model; circles identify search tools (i=1) and squares identify read tools (i=2). Intervals are the original joint 95% max-t bootstrap intervals across all 20 contrasts. The shaded region denotes ±0.02; all contrasts remain inconclusive.

Figure 3. (a) Mean generation counts per episode; connectors aid model comparison and are not confidence intervals. (b) Empirical cumulative distribution of nominal cost for 60 service blocks, each containing seven conditions. The dashed line marks the mean. Cost uses logged provider usage at 2.5 CNY per million input tokens and 10 CNY per million output tokens; discounts and historical paid preflight are excluded. Blocks are displayed descriptively and are not claimed to be independent tasks.

Style reference: NeurIPS formatting instructions, sections 4.3–4.4 and 6 (legibility, grayscale usability, captions outside artwork, embedded fonts): https://media.neurips.cc/Conferences/NeurIPS2022/Styles/neurips_2022.pdf . This is a design reference, not the SANER submission specification.

All figures use the same model colors and marker shapes. PDF and SVG are vector outputs; PNG is exported at 400 dpi. Canvas width is 7.2 inches for a double-column figure; adjust to the final venue template. No statistical values were recomputed. Original files remain available in the parent directory.
'''
    (a.output_dir/'figure_captions.md').write_text(captions,encoding='utf-8')
    manifest={'source_sha256':{n:hashlib.sha256((a.source_dir/n).read_bytes()).hexdigest() for n in names},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'statistical_recomputation':False, 'matplotlib_version':matplotlib.__version__,
              'outputs':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output_dir.iterdir() if f.suffix in ['.pdf','.png','.svg']}}
    (a.output_dir/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(str(a.output_dir))

if __name__=='__main__': main()
