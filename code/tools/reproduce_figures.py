"""Run preserved figure scripts in a NEW isolated output project.

main, intro and costs work from code.zip alone. Appendix data redraws additionally
use --appendix-tex from the matching LaTeX package to verify the figure scope
and matched numerical table. No source data, manuscript or original PDF is
modified. Original scripts remain byte-identical in tools/figure_sources/.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--which',default='main,intro,costs',help='Comma-separated main,intro,costs,appendix')
    parser.add_argument('--appendix-tex',type=Path,default=CODE.parent/'appendix.tex')
    parser.add_argument('--appendix-groups',default='analytic,matched,cluster,hans,qwen_paired,p0,boundaries,dashboard,vectors')
    parser.add_argument('--font-regular',type=Path)
    parser.add_argument('--font-bold',type=Path)
    parser.add_argument('--chrome',type=Path,help='Chrome/Chromium/Edge executable for historical vector-only PDF reflow')
    args=parser.parse_args()
    selected=args.which.split(',')
    if set(selected)-{'main','intro','costs','appendix'}:
        parser.error('Unknown figure group')
    appendix_groups=args.appendix_groups.split(',')
    allowed={'analytic','matched','cluster','hans','qwen_paired','p0','boundaries','dashboard','vectors'}
    if set(appendix_groups)-allowed:
        parser.error('Unknown appendix group')
    if 'appendix' in selected and not args.appendix_tex.is_file():
        parser.error('Appendix data redraws need --appendix-tex from the matching LaTeX package')
    if args.chrome is not None and not args.chrome.is_file():
        parser.error('--chrome must name an existing executable')
    out=args.output_dir.resolve()
    if out==CODE or CODE.is_relative_to(out) or out.is_relative_to(CODE/'oral_planner_v6') or out.is_relative_to(CODE/'oral_attribution_v5'):
        parser.error('Choose a new output directory outside scientific source/result directories')
    figure_assets=json.loads((CODE/'FIGURE_ASSETS.json').read_text(encoding='utf-8'))
    for r in figure_assets['files']:
        if sha(CODE/r['path'])!=r['sha256']:
            raise ValueError('Figure snapshot hash mismatch: '+r['path'])
    out.mkdir(parents=True,exist_ok=False)
    for sub in ('research','figures','build/revision'):
        (out/sub).mkdir(parents=True,exist_ok=True)
    receipt={'inputs':{},'original_script_sha256':{},'selected':selected,'adaptations':[]}

    def copy(src,rel):
        dst=out/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        receipt['inputs'][rel]=sha(src)
        return dst

    for name in ('plot_main_evidence.py','layout_intro_pdf.py','plot_planner_costs.py','figure_repairs.py'):
        src=CODE/'tools/figure_sources'/name
        copy(src,'research/'+name)
        receipt['original_script_sha256'][name]=sha(src)
    if 'main' in selected:
        for rel in ('oral_planner_v6/results/summary.json','oral_attribution_v5/results/dense_thresholds.csv'):
            copy(CODE/rel,'code/'+rel)
        runpy.run_path(str(out/'research/plot_main_evidence.py'),run_name='__main__')
    if 'costs' in selected:
        rel='oral_planner_v6/results/summary.json'
        copy(CODE/rel,'code/'+rel)
        runpy.run_path(str(out/'research/plot_planner_costs.py'),run_name='__main__')
    if 'intro' in selected:
        copy(CODE/'figure_inputs/intro.pdf','figures/intro.pdf')
        import matplotlib
        fontroot=Path(matplotlib.get_data_path())/'fonts/ttf'
        winregular=Path('C:/Windows/Fonts/times.ttf')
        winbold=Path('C:/Windows/Fonts/timesbd.ttf')
        regular=args.font_regular or (winregular if winregular.is_file() else fontroot/'DejaVuSerif.ttf')
        bold=args.font_bold or (winbold if winbold.is_file() else fontroot/'DejaVuSerif-Bold.ttf')
        if not regular.is_file() or not bold.is_file():
            raise FileNotFoundError('Supply --font-regular and --font-bold as readable TTF files')
        script=out/'research/layout_intro_pdf.py'
        text=script.read_text(encoding='utf-8')
        # Only font-file literals are redirected in memory. Crop coordinates,
        # vector operations, data interiors and the archived source do not change.
        for old,new in [('C:/Windows/Fonts/times.ttf',regular),('C:/Windows/Fonts/timesbd.ttf',bold)]:
            if old not in text:
                raise ValueError('Font adapter requires review after source update')
            text=text.replace(old,new.resolve().as_posix())
        receipt['adaptations'].append({'intro_fonts':{'regular':regular.name,'bold':bold.name},
            'font_sha256':{'regular':sha(regular),'bold':sha(bold)},
            'scope':'Font path substitution only; fallback typography can differ. No proprietary font is bundled.'})
        exec(compile(text,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
    if 'appendix' in selected:
        copy(args.appendix_tex,'appendix.tex')
        for name in ('dense_thresholds.json','factorial.csv','oracle_forecast.json','MATCHED_RESULTS_MANIFEST.json'):
            rel='oral_attribution_v5/results/'+name
            copy(CODE/rel,'code/'+rel)
        rel='overleaf/experiments/comparative_utility/generate_comparative_utility.py'
        copy(CODE/rel,'code/'+rel)
        spec=importlib.util.spec_from_file_location('portable_figure_repairs',out/'research/figure_repairs.py')
        mod=importlib.util.module_from_spec(spec)
        body=(out/'research/figure_repairs.py').read_text(encoding='utf-8')
        if 'vectors' in appendix_groups:
            candidates=[args.chrome]
            candidates.extend(Path(p) for name in ('google-chrome','chromium','chromium-browser','chrome','msedge')
                              if (p:=shutil.which(name)))
            candidates.extend([Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),
                               Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'),
                               Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')])
            browser=next((p.resolve() for p in candidates if p is not None and p.is_file()),None)
            if browser is None:
                raise FileNotFoundError('Vector-only PDF reflow requires --chrome PATH; data redraws can omit the vectors group')
            old="Path('C:/Program Files/Google/Chrome/Application/chrome.exe')"
            if body.count(old)!=1:
                raise ValueError('Browser path adapter requires review after source update')
            body=body.replace(old,'Path('+repr(browser.as_posix())+')')
            receipt['adaptations'].append({'vector_pdf_browser':browser.name,'browser_sha256':sha(browser),
                'scope':'Only browser executable path substituted; original clipping-preserving headless print arguments unchanged.'})
        exec(compile(body,str(out/'research/figure_repairs.py'),'exec'),mod.__dict__)
        mod.LEGACY=CODE/'overleaf'
        mod.QA=out/'qa'
        mod.QA.mkdir()
        copy(CODE/'figure_inputs/metadata/initial_manifest.json','qa/initial_manifest.json')
        # The approved r3 HANS renderer verifies an already-frozen repair CSV
        # against its prior ledger. Recreate that input layout, not its statistics.
        copy(CODE/'figure_inputs/metadata/repair_manifest.json','qa/repair_manifest.json')
        if 'hans' in appendix_groups:
            hans_input=CODE/'overleaf/experiments/hans_suite/outputs/results.csv'
            ledger=json.loads((mod.QA/'repair_manifest.json').read_text(encoding='utf-8'))
            for name in ('fig_hans_main.pdf','fig_hans_external.pdf'):
                row=ledger['experiments/hans_suite/outputs/'+name]
                if row['sources'][0]['sha256']!=sha(hans_input):
                    raise ValueError('Frozen HANS input does not match the approved repair ledger')
            copy(hans_input,'experiments/hans_suite/outputs/repair_data/results.csv')
        initial=json.loads((mod.QA/'initial_manifest.json').read_text(encoding='utf-8'))
        for row in initial['figures']:
            copy(CODE/'figure_inputs/originals'/row['path'],'qa/originals/'+row['path'])
        mod.STAGE=out
        mod.style()
        registry={'analytic':mod.analytic,'matched':mod.matched,'cluster':mod.qwen_cluster,'hans':mod.hans,
                  'qwen_paired':mod.qwen_paired,'p0':mod.p0,'boundaries':mod.boundaries,'dashboard':mod.dashboard,
                  'vectors':mod.vector_reflow}
        for name in appendix_groups:
            registry[name]()
            print('RENDERED appendix',name,flush=True)
        (mod.QA/'repair_manifest.json').write_text(json.dumps(mod.LEDGER,indent=2,ensure_ascii=False),encoding='utf-8')
        receipt['adaptations'].append({'appendix_legacy_root':'code/overleaf','output_root':'.',
            'scope':'Data/analytic redraws and explicitly labelled vector-only reflow; original publication workflow not invoked. Vector-only PDFs are not empirical raw data.'})
    receipt['output_sha256']={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
                              if p.is_file() and p.suffix in {'.pdf','.svg','.png'}
                              and p.relative_to(out).as_posix() not in receipt['inputs']}
    receipt['status']='completed; visual acceptance is separate'
    (out/'FIGURE_RUN.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'graphics':len(receipt['output_sha256']),'status':receipt['status']}))


if __name__=='__main__':
    main()
