"""Add the documented Tiangong-2 case alongside the existing pose replay."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'results/gait_selected_paper_20261115'
path=WEB/'index.html'
html=path.read_text()
case='''<!-- TIANGONG CASE START -->
<article id="tiangong-case" class="case-study" aria-labelledby="tiangong-title">
  <div class="eyebrow">IN-ORBIT CASE STUDY / TIANGONG-2 · 2016</div>
  <h3 id="tiangong-title">Learning to run in microgravity.</h3>
  <p class="case-intro">A documented treadmill adaptation difficulty, with the crew’s account and accompanying footage.</p>
  <div class="case-grid">
    <figure>
      <div class="case-video-wrap"><video id="tiangong-video" controls playsinline preload="metadata" poster="case_studies/tiangong2/edited_poster.jpg" src="case_studies/tiangong2/edited_us_flags.mp4" aria-label="Tiangong-2 treadmill footage, edited U.S. flag version"></video></div>
      <div class="case-switch" role="group" aria-label="Choose original or edited footage">
        <button id="tiangong-edited" aria-pressed="true">U.S. flags · edited</button>
        <button id="tiangong-original" aria-pressed="false">Original footage</button>
      </div>
      <figcaption id="tiangong-caption" aria-live="polite">Edited visuals: wall flags replaced with U.S. flags. The actual mission is Tiangong-2, China. Original audio retained.</figcaption>
      <p class="fine">14 seconds · 480 × 360 source · 25 fps<br>CCTV broadcast excerpt: 11:29–11:43 · published 22 January 2018</p>
    </figure>
    <div class="case-evidence">
      <span class="pill">CREW-REPORTED DIFFICULTY</span>
      <h4>What the source confirms</h4>
      <p>Jing Haipeng and Chen Dong reported that their first treadmill attempts did not allow sustained running. Jing later described a head and posture adjustment that helped him run.</p>
      <p class="fine"><a href="https://www.xinhuanet.com/mil/2016-12/07/c_1120075273.htm" target="_blank" rel="noopener">Crew account · Xinhua, 7 December 2016 ↗</a></p>
      <div class="case-caveat"><b>What remains unverified</b><p>The edited broadcast does not establish which shots show the unsuccessful attempts. This clip has no confirmed knee-fault label, pose overlay, or model score.</p></div>
      <div class="case-links">
        <a href="https://tv.cctv.com/2018/01/22/VIDE0WQWZgwYh8iTl63g57iW180122.shtml" target="_blank" rel="noopener">Original CCTV broadcast ↗</a>
        <a href="https://www.xinhuanet.com/politics/2016-11/08/c_1119874899.htm" target="_blank" rel="noopener">Chen Dong’s in-orbit account ↗</a>
        <a href="case_studies/tiangong2/source_review.json">Source and edit record ↗</a>
      </div>
    </div>
  </div>
</article>
<!-- TIANGONG CASE END -->'''
if '<!-- TIANGONG CASE START -->' in html:
    start=html.index('<!-- TIANGONG CASE START -->')
    end=html.index('<!-- TIANGONG CASE END -->')+len('<!-- TIANGONG CASE END -->')
    html=html[:start]+case+html[end:]
else:
    marker='</section>\n<section id="analysis">'
    assert marker in html
    html=html.replace(marker,case+marker,1)
anchor='<a class="button" href="#tiangong-case">Watch Tiangong-2 case ↘</a>'
if anchor not in html:
    needle='<div class="space-shortcut">'
    assert needle in html
    html=html.replace(needle,needle+anchor,1)
path.write_text(html)
record=WEB/'case_studies/tiangong2/source_review.json'
if record.exists():
    review=json.loads(record.read_text())
    review['assessment']['demo_modified']=True
    review['assessment']['demo_role']='Documented contextual case, original footage and explicitly labeled flag edit; independent of pose replay.'
    record.write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n')
print('Updated',path)
