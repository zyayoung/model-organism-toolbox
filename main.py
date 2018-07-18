import os
import glob
from nematoda import Nematoda

nematoda = Nematoda(
    r'D:\Projects\model_organism_helper\videos\Nematoda\capture-0004.avi',
    resize_ratio=.5,
    display_scale=.5
)
nematoda.config()
nematoda.process(online=True, output_filename=r'D:\Projects\model_organism_helper\videos\Nematoda\output\capture-0004.avi')
