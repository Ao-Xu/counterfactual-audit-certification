"""Lossless chart-panel layout from the user PDF; no recovered empirical data.

The four plot interiors are copied, not regenerated. Their actual sampled
points are unmodified. A narrow open strip removes the interpolation segment
between n=500 and n=700 in the KL panel, where the statistic changes definition.
Tick positions and labels are transcribed from the supplied axes. The original
PDF is preserved as the immutable source, not overwritten.
"""
from pathlib import Path
import fitz
import json
import hashlib

root=Path(__file__).resolve().parents[1]
source=root/'figures/intro.pdf'
doc=fitz.open(str(source))
out=fitz.open();p=out.new_page(width=396,height=206)
p.insert_font(fontname='TNR',fontfile='C:/Windows/Fonts/times.ttf')
p.insert_font(fontname='TNRB',fontfile='C:/Windows/Fonts/timesbd.ttf')
def txt(x,y,text,size=7,center=False,bold=False,color=(.1,.1,.1)):
    fontfile='C:/Windows/Fonts/timesbd.ttf' if bold else 'C:/Windows/Fonts/times.ttf'
    font=fitz.Font(fontfile=fontfile)
    if center:x-=font.text_length(text,fontsize=size)/2
    p.insert_text((x,y),text,fontsize=size,fontname='TNRB' if bold else 'TNR',color=color)

# Re-typeset the same workflow as an editable vector schematic. The small
# labels in the supplied illustration were not legible at paper width.
blue=(0,.36,.56); orange=(.70,.30,.02)
def box(rect, lines, col):
    p.draw_rect(fitz.Rect(rect),color=col,fill=tuple(.95+.05*c for c in col),width=.7)
    cx=(rect[0]+rect[2])/2;cy=(rect[1]+rect[3])/2
    for k,line in enumerate(lines):
        txt(cx,cy+2.3+(k-(len(lines)-1)/2)*9,line,7,True,color=col)
def arrow(points,col=(.3,.3,.3)):
    for start,end in zip(points[:-1],points[1:]):p.draw_line(start,end,color=col,width=.8)
    import math
    x,y=points[-1];x0,y0=points[-2];a=math.atan2(y-y0,x-x0)
    for da in [-.55,.55]:p.draw_line((x,y),(x-3.2*math.cos(a+da),y-3.2*math.sin(a+da)),color=col,width=.8)
txt(50,20,'Counterfactual',8.3,True,True)
txt(50,31,'post-training',8.3,True,True)
box((5,48,98,70),['Factual training data'],blue)
box((53,90,100,115),['Generate','CF edits'],orange)
box((53,135,100,160),['Factual +','CF data'],orange)
box((2,176,48,201),['Factual','model'],blue)
box((53,176,100,201),['CFPT','model'],orange)
arrow([(25,70),(25,176)],blue)
arrow([(76,70),(76,90)],orange)
arrow([(76,115),(76,135)],orange)
arrow([(76,160),(76,176)],orange)
txt(25,122,'Fine-tune',7,True,color=blue)
# White backing keeps the edge label distinct from the arrow beneath it.
p.draw_rect(fitz.Rect(5,115,46,125),color=None,fill=(1,1,1))
txt(25,122,'Fine-tune',7,True,color=blue)

panels=[
 ('Brier risk',(473,74,675,190),(131,22,250,82),[(85.5,'.072'),(112.5,'.066'),(139.5,'.060'),(167,'.054')]),
 ('Classification error',(735,74,936,190),(276,22,386,82),[(82,'8.8%'),(107.5,'8.0%'),(133.5,'7.2%'),(159.5,'6.4%'),(186,'5.6%')]),
 ('Predictive KL (nats)',(473,250,675,366),(131,123,250,183),[(265.5,'.24'),(291,'.18'),(316,'.12'),(341.5,'.06')]),
 ('Audited label disagreement',(735,250,936,366),(276,123,386,183),[(253.5,'48%'),(278,'42%'),(302,'36%'),(326,'30%'),(350.5,'24%')])]
for i,(title,cliprect,rect,yticks) in enumerate(panels):
    clip=fitz.Rect(cliprect);dest=fitz.Rect(rect)
    p.show_pdf_page(dest,doc,0,clip=clip,keep_proportion=False)
    p.draw_rect(dest,color=(.3,.3,.3),width=.35)
    txt((dest.x0+dest.x1)/2,dest.y0-7,title,7.8,True,True)
    for yy,label in yticks:
        y=dest.y0+(yy-clip.y0)*dest.height/clip.height
        txt(dest.x0-3-fitz.Font(fontfile='C:/Windows/Fonts/times.ttf').text_length(label,fontsize=7),y+2,label,7)
    # Original sample positions, not interpolated loss values.
    xx0=479 if i%2==0 else 740.5;xx1=669 if i%2==0 else 931
    for n in [100,500,900,1300,1900]:
        xx=xx0+(n-100)*(xx1-xx0)/1800
        x=dest.x0+(xx-clip.x0)*dest.width/clip.width
        txt(x,dest.y1+8,str(n),7,True)
    txt((dest.x0+dest.x1)/2,dest.y1+17,'Generated training examples, n',6.7,True)
    if i==2:
        x=dest.x0+(531-clip.x0)*dest.width/clip.width
        # No data were sampled near n=600: only cross-definition connectors
        # and their interpolated fill are removed. Keep both endpoints.
        p.draw_rect(fitz.Rect(x-2,dest.y0+.35,x+2,dest.y1-.35),color=None,fill=(1,1,1),overlay=True)
        p.draw_line((x-2,dest.y0+.35),(x-2,dest.y1-.35),color=(.5,.5,.5),width=.35,dashes='[2 2]')
        p.draw_line((x+2,dest.y0+.35),(x+2,dest.y1-.35),color=(.5,.5,.5),width=.35,dashes='[2 2]')
        txt(dest.x0+16,dest.y0-1,'Original–CF',5.5,True,color=(.3,.3,.3))
        txt(dest.x1-20,dest.y0-1,'CF–CF',6,True,color=(.3,.3,.3))
out.save(str(root/'figures/intro_audited.pdf'),garbage=4,deflate=True)
(root/'figures/intro_audited.svg').write_text(p.get_svg_image(text_as_path=False),encoding='utf-8')
p.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(str(root/'build/revision/intro_audited.png'))
(root/'research/intro_figure_provenance.json').write_text(json.dumps({
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'operation':'PDF panel copying, vector axes, and re-typeset equivalent workflow; no empirical curve reconstruction',
 'interiors':panels,'KL_connector_gap_original_pdf_x':[527.6,534.4],
 'limitations':'User-provided plot interiors are rasterized in the source PDF; raw plot arrays were not supplied. The source PDF remains unchanged.'},indent=2),encoding='utf-8')
print('Wrote intro_audited.pdf from unchanged source panels.')
