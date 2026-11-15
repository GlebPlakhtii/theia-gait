"""Export descriptive participant plots from existing, quality-gated results."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

FOCUS = [('trunk_frontal_rom95', 'Frontal trunk motion'),
         ('hip_line_rom95_proxy', 'Hip-line motion proxy'),
         ('shoulder_line_rom95', 'Shoulder-line motion')]
SIDE_FOCUS = [('knee_rom95_mean', 'Knee motion'),
              ('knee_rom95_asymmetry', 'Left/right knee ROM difference'),
              ('trunk_side_rom95', 'Side-view trunk motion')]
GROUPS = [('with_amputation', 'ProGait · with amputation', '#b96b2b', 'D'),
          ('without_amputation', 'Reference · without amputation', '#147d75', 'o')]
MODE = 'guarded_known_subject_sensitivity'


def short_id(value):
    return value.replace('participant:', '').replace('progait_', 'PG').replace('ljmu360_', '')


def label_scatter(ax, points):
    """Place participant labels without hiding nearby points or other labels."""
    fig=ax.figure;fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bounds=ax.get_window_extent(renderer);occupied=[]
    if ax.get_legend() is not None:occupied.append(ax.get_legend().get_window_extent(renderer))
    marker_boxes=[Bbox.from_bounds(px-8,py-8,16,16) for px,py in
                  ax.transData.transform([(p['trunk_rom95_deg'],p['shoulder_rom95_deg']) for p in points])]
    directions=[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]
    for point in points:
        xy=(point['trunk_rom95_deg'],point['shoulder_rom95_deg'])
        text=ax.annotate(point['label'],xy,xytext=(8,8),textcoords='offset points',fontsize=9)
        best=None
        for radius in [8,16,24,36,52,70]:
            for sx,sy in directions:
                dx,dy=radius*sx,radius*sy;text.set_position((dx,dy))
                text.set_ha('left' if dx>0 else 'right' if dx<0 else 'center')
                text.set_va('bottom' if dy>=0 else 'top')
                box=text.get_window_extent(renderer).expanded(1.12,1.18)
                if not (bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1)):continue
                collisions=sum(box.overlaps(other) for other in occupied+marker_boxes)
                score=(collisions,abs(dx)+abs(dy))
                if best is None or score<best[0]:best=(score,dx,dy,box)
            if best and best[0][0]==0:break
        if best:
            _,dx,dy,box=best;text.set_position((dx,dy))
            text.set_ha('left' if dx>0 else 'right' if dx<0 else 'center');text.set_va('bottom' if dy>=0 else 'top')
            occupied.append(box)
            if max(abs(dx),abs(dy))>8:
                ax.annotate('',xy,xytext=(dx,dy),textcoords='offset points',
                            arrowprops=dict(arrowstyle='-',color='#788895',lw=.6,shrinkA=4,shrinkB=7))


def build_review_figures(report, catalog, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    results = {r['outcome']['id']: r for r in report['results']
               if r['mode'] == MODE and r['sensitivity'] == 'base'
               and r['stratum']['environment'] == 'overground'}
    methods = {r['outcome']['id']: r for r in report['paired_pose_method_summary']}
    files = []
    def save(fig, stem):
        for ext in ['png', 'svg']:
            target = destination / f'{stem}.{ext}'
            temporary = target.with_suffix('.tmp')
            fig.savefig(temporary, format=ext, dpi=180)
            temporary.replace(target)
        plt.close(fig)
        files.append(stem)

    # Repeated clips are already collapsed by trial and participant upstream.
    distribution_outcomes = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for ax, (key, title) in zip(axes.ravel(), FOCUS + SIDE_FOCUS):
        item = results[key]
        distribution_outcomes.append(dict(outcome=key, view=item['stratum']['view'],
            unit=item['outcome']['unit'], groups={g: item['groups'][g] for g, *_ in GROUPS}))
        labels = []
        for i, (group, label, color, marker) in enumerate(GROUPS):
            g = item['groups'][group]
            units = g['units']; n = len(units)
            labels.append(f'{"With" if i == 0 else "Reference"}\nn = {n} {"person" if n == 1 else "people"}')
            jitter = np.linspace(-.12, .12, n) if n > 1 else [0]
            for j, (unit, dx) in enumerate(zip(units, jitter)):
                ax.scatter(i + dx, unit['value'], s=50, marker=marker, color=color,
                           edgecolors='white', linewidths=.7, zorder=4)
            if n > 1:
                ax.plot([i-.23, i-.23], [g['q25'], g['q75']], lw=5, color=color, alpha=.55)
                ax.plot([i-.29, i-.17], [g['median'], g['median']], lw=2.5, color=color)
            if not n:
                ax.text(i, .5, 'Unavailable', ha='center', transform=ax.get_xaxis_transform())
        ax.set_xticks([0, 1], labels); ax.set_xlim(-.4, 1.55)
        ax.set_ylim(bottom=0); ax.margins(y=.16)
        view = 'front/rear view' if key in dict(FOCUS) else 'side view'
        ax.set_title(title+'\n'+view, fontsize=11)
        ax.set_ylabel('Absolute ROM95 difference (degrees)' if key == 'knee_rom95_asymmetry'
                      else 'Projected ROM95 (degrees)')
        ax.grid(axis='y', alpha=.16)
    fig.suptitle('Participant distributions · overground walking · same Sapiens2-1B model', fontsize=14)
    fig.text(.5, .015, 'One dot = one participant. Views stay separate; the same people can appear in multiple panels.\nBars = group IQR; short line = median. Guarding differs; no clinical threshold.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95), h_pad=2.5); save(fig, 'participant_distributions')

    # Join participants by identity; never pair different people or pool frames.
    points = []
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    for group, label, color, marker in GROUPS:
        xs = {u['unit_id']: u for u in results['trunk_frontal_rom95']['groups'][group]['units']}
        ys = {u['unit_id']: u for u in results['shoulder_line_rom95']['groups'][group]['units']}
        joined = sorted(xs.keys() & ys.keys())
        for j, uid in enumerate(joined):
            x,y = xs[uid]['value'],ys[uid]['value']
            ax.scatter(x, y, c=color, marker=marker, s=95, edgecolors='white', linewidths=.8,
                       label=f'{label} (n={len(joined)})' if j == 0 else None, zorder=3)
            points.append(dict(unit_id=uid, label=short_id(uid), cohort=group,
                trunk_rom95_deg=x, shoulder_rom95_deg=y,
                trunk_clips=xs[uid]['clips'], shoulder_clips=ys[uid]['clips'],
                common_clips=sorted(set(xs[uid]['clips']) & set(ys[uid]['clips']))))
    ax.set_xlabel('Frontal trunk ROM95 (degrees)'); ax.set_ylabel('Shoulder-line ROM95 (degrees)')
    ax.set_title('Trunk motion versus shoulder motion · front/rear view', fontsize=14, pad=16)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.margins(.18)
    ax.grid(alpha=.16)
    if points: ax.legend(loc='upper left', frameon=False, fontsize=9)
    else: ax.text(.5,.5,'No eligible paired participant observations yet',transform=ax.transAxes,ha='center')
    fig.text(.5,.02,'Each point is one participant. No regression or diagnostic boundary is fitted. Assistance and camera conditions differ.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1))
    if points:label_scatter(ax,points)
    save(fig,'trunk_shoulder_scatter')

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (key, title) in enumerate(FOCUS):
        group_delta = results[key]['median_difference_with_minus_without']
        method_delta = methods[key]['participant_summary']['median']
        for offset, value, color, label in [(-.13,group_delta,'#b96b2b','Group medians: with − reference'),
                                            (.13,method_delta,'#506bb3','Same-video method: Sapiens − publisher')]:
            if value is None: continue
            ax.hlines(i+offset, 0, value, color=color, lw=2)
            ax.scatter(value, i+offset, c=color, s=65, label=label if i == 0 else None, zorder=4)
            ax.annotate(f'{value:+.2f}°',(value,i+offset),xytext=(6 if value>=0 else -6,0),
                        textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=10)
        n = methods[key]['participant_summary']['participant_count'] or 0
        ax.text(.99, i+.29, f'method pairs: {n} participant(s)',transform=ax.get_yaxis_transform(),
                ha='right',va='center',fontsize=8,color='#5e6d79')
    ax.set_yticks(range(3),[title for _,title in FOCUS]); ax.invert_yaxis()
    ax.axvline(0,color='#9ba9b4',lw=1); ax.set_xlabel('Signed difference in projected ROM95 (degrees)')
    ax.set_title('Group differences alongside pose-method sensitivity',fontsize=14,pad=15)
    ax.margins(x=.23,y=.3); ax.grid(axis='x',alpha=.14)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.23),frameon=False,ncol=2,fontsize=9)
    fig.text(.5,.015,'Method disagreement is not a calibrated error bound. These are descriptive values; no confidence intervals or significance tests.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1)); save(fig,'method_sensitivity')

    counts=Counter((c['source'],c['status'])for c in catalog['clips'])
    sources=sorted({c['source']for c in catalog['clips']})
    totals=[sum(c['source']==s for c in catalog['clips'])for s in sources]
    done=[counts[s,'observations_extracted']for s in sources]
    labels=[s+(' · publisher poses' if s=='progait' else ' · Sapiens2-1B')for s in sources]
    coverage_rows=[dict(source=s,method='publisher' if s=='progait' else 'sapiens2_1b',
                       extracted=d,total=t,subset_rerun=False)for s,d,t in zip(sources,done,totals)]
    reruns=report.get('harmonization_status',[])
    if reruns:
        labels.append('progait · Sapiens2-1B rerun subset')
        done.append(sum(x['status']=='observations_extracted' for x in reruns));totals.append(len(reruns))
        coverage_rows.append(dict(source='progait',method='sapiens2_1b',extracted=done[-1],total=totals[-1],subset_rerun=True))
    percentages=np.array(done)/np.array(totals)*100
    fig,(ax,eligible_ax)=plt.subplots(1,2,figsize=(14,5.7),gridspec_kw={'width_ratios':[1.25,1]})
    ys=np.arange(len(labels))
    ax.barh(ys,100,color='#e7edf0',height=.55,label='Pending extraction')
    ax.barh(ys,percentages,color='#147d75',height=.55,label='Observations extracted')
    ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_xlim(0,122)
    ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
    for y,d,t in zip(ys,done,totals):ax.text(102,y,f'{d} / {t}',va='center',fontsize=10)
    ax.set_xlabel('Share of clips with extracted observations');ax.set_title('Processing coverage by source',fontsize=13,pad=15)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.19),frameon=False,ncol=2,fontsize=9)
    eligibility_rows=[]
    coverage_focus=[('trunk_frontal_rom95','Trunk · front/rear'),('shoulder_line_rom95','Shoulders · front/rear'),
                    ('knee_rom95_mean','Knees · side'),('trunk_side_rom95','Trunk · side')]
    positions=np.arange(len(coverage_focus))
    for j,(group,label,color,marker) in enumerate(GROUPS):
        values=[results[key]['groups'][group]['participant_count'] or 0 for key,_ in coverage_focus]
        bars=eligible_ax.barh(positions+(j-.5)*.32,values,height=.28,color=color,label='With amputation' if j==0 else 'Reference')
        eligible_ax.bar_label(bars,padding=3,fontsize=9)
    for key,label in coverage_focus:
        eligibility_rows.append(dict(outcome=key,view=results[key]['stratum']['view'],
            groups={group:dict(participants=results[key]['groups'][group]['participant_count'],
                              clips=results[key]['groups'][group]['clip_count'])for group,*_ in GROUPS}))
    eligible_ax.set_yticks(positions,[label for _,label in coverage_focus]);eligible_ax.invert_yaxis()
    maximum=max([results[key]['groups'][group]['participant_count'] or 0 for key,_ in coverage_focus for group,*_ in GROUPS]+[1])
    eligible_ax.set_xlim(0,maximum+2);eligible_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    eligible_ax.set_xlabel('Known participants passing measurement QC')
    eligible_ax.set_title('Eligible participant coverage',fontsize=13,pad=15);eligible_ax.grid(axis='x',alpha=.14)
    eligible_ax.legend(loc='upper left',bbox_to_anchor=(0,-.19),frameon=False,ncol=2,fontsize=9)
    fig.text(.5,.01,'Left: processing coverage; reruns are a subset of the 596 clips. Right: base QC, known-participant overground comparison.\nCounts are per measurement and must not be added across rows. Additional-input identities remain unknown. No correct-form certification.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1),w_pad=3);save(fig,'source_coverage')
    payload=dict(status=report['status'],comparison_updated_at=report['updated_at'],
        generated_at=datetime.now(timezone.utc).isoformat(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Base QC; overground; guarded known-subject sensitivity; projected geometry only.',
        scatter_points=points,distribution_outcomes=distribution_outcomes,coverage_rows=coverage_rows,
        eligible_participant_coverage=eligibility_rows,files=files,formats=['png','svg'],
        no_regression=True,no_clinical_threshold=True)
    manifest=destination/'review_figures.json';temporary=manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False));temporary.replace(manifest)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);args=parser.parse_args()
    payload=build_review_figures(json.loads((args.source/'comparison_results.json').read_text()),
        json.loads((args.source/'catalog.json').read_text()),args.source/'figures')
    print(json.dumps({'figures':payload['files'],'scatter_participants':len(payload['scatter_points'])}))
