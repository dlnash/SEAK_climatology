# Standard Python modules
from PIL import Image

for i in range(6):
    infname = '../figs/final_figs/fig{0}.png'.format(str(i))
    outfname = '../figs/final_figs_pdf/fig{0}.pdf'.format(str(i))

    image_1 = Image.open(infname)
    im_1 = image_1.convert('RGB')
    im_1.save(outfname)