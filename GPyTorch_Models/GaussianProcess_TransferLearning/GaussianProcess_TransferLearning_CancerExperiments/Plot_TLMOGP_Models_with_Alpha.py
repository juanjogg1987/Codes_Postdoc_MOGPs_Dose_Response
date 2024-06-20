import io
import fitz
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

which_out = 0
plt.close('all')
alpha = 0.1#1.0
vect = np.random.randint(48, size=(20))
#for i in vect:#range(10):
fig, axs = plt.subplots(6, 8,figsize=(25,17))
#axs[0, 0].plot(x, y)
for i in range(0,48):
    try:
        file = '/home/juanjo/Downloads/TargetCancer5/Drug_[1051 1179 1190]/CellLine0_CID683667/Test_plot/'+str(i)+'/plot'+str(which_out)+'.pdf'
        #file = '/home/juanjo/Downloads/TargetCancer5/Drug_[1051 1022 1818 1511]/CellLine0_CID683667/Test_plot/'+str(i)+'/plot' + str(which_out) + '.pdf'
        #file = '/home/juanjo/Downloads/TargetCancer5_Old2/Drug_[1051 1022 1818 1511]_Without0.1Noise/CellLine0_CID683667/Test_plot/'+str(i)+'/plot' + str(which_out) + '.pdf'
        #file = '/home/juanjo/Downloads/TargetCancer5_Old2/Drug_[1051 1179 1190]_9Domains_plus0.1Noise/CellLine0_CID683667/Test_plot/'+str(i)+'/plot' + str(which_out) + '.pdf'
        pdf_file = fitz.open(file)

        # in case there is a need to loop through multiple PDF pages
        for page_number in range(len(pdf_file)):
            page = pdf_file[page_number]
            rgb = page.get_pixmap()
            pil_image = Image.open(io.BytesIO(rgb.tobytes()))

        plt.figure(0)
        #alpha = alpha * 0.5
        plt.imshow(pil_image.convert('RGB'),alpha=alpha)
        print(f"{i//8} and {i%8}")
        axs[i//8 ,i%8].imshow(pil_image.convert('RGB'))
    except:
        print(f"No FOLDER {i}")