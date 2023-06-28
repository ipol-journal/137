#!/usr/bin/env python3

import subprocess
import argparse
import PIL.Image
import math

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("pattern", type=int)
args = ap.parse_args()

# Demosaic image
subprocess.run(['demosaick', '-p', str(args.pattern), '-c', 'input_0_sel.png'])
subprocess.run(['dmbilinear', '-p', str(args.pattern), 'mosaiced.png', 'bilinear.png'])
subprocess.run(['dmzhangwu', '-p', str(args.pattern), 'mosaiced.png', 'zhangwu.png'])

# crop images
files = ['bilinear', 'mosaiced.png', 'zhangwu.png']
for filename in files:
    im = PIL.Image.open(filename + '.png')
    (sx, sy) = im.size
    # 2 is the border left out in estimation
    img = im.crop((2, 2, sx-2, sy-2))
    img.save(filename + '.png')

# Compute image differences
subprocess.run(['imdiff', 'ccropped.png', 'demosaiced.png', 'diffdemosaiced.png']),
subprocess.run(['imdiff', 'ccropped.png', 'bilinear.png', 'diffbilinear.png']),
subprocess.run(['imdiff', 'ccropped.png', 'zhangwu.png', 'diffzhangwu.png'])

# Compute image rmse
with open('rmse_nn.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'ccropped.png', 'demosaiced.png'], stdout=stdout, stderr=stdout),

with open('rmse_bilinear.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'ccropped.png', 'bilinear.png'], stdout=stdout, stderr=stdout),

with open('rmse_zhangwu.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'ccropped.png', 'zhangwu.png'], stdout=stdout, stderr=stdout),

# Resize for visualization (new size of the smallest dimension = 200)
(sizeX, sizeY) = PIL.Image.open('input_0.png').size
zoomfactor = max(1, int(math.ceil(200.0/min(sizeX, sizeY))))
(sizeX, sizeY) = (zoomfactor*sizeX, zoomfactor*sizeY)

for filename in ['input_0', 'mosaiced', 'demosaiced', 'bilinear', 'diffdemosaiced', 'diffbilinear', 'ccropped',
    'zhangwu', 'diffzhangwu']:
    im = PIL.Image.open(filename + '.png')
    im.resize((sizeX, sizeY), method='nearest')
    im.save(filename + '_zoom.png')
   