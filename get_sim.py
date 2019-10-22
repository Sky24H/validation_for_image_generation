import argparse
import pytorch_ssim
from FUNCTIONS import compute

parser = argparse.ArgumentParser()

parser.add_argument('--path_a', dest='a', type=str, required=True)
parser.add_argument('--path_b', dest='b', type=str, required=True)
parser.add_argument('--output', dest='o', type=str, required=True)

opt = parser.parse_args()

print('-------Results of HSV&RGB are distances (the lower the better)-------')
print('-----------------Others are similarities (otherwise)-----------------')
print('')
#print('result of PSNR:  ' + '%.6f' % psnr(opt.a,opt.b))
#print('result of FSIM:  ' + '%.6f' % fsim(opt.a,opt.b))

compute(opt.a,opt.b,opt.o)
#resr, resh, resm, resp, resf = compute(opt.a,opt.b)
#print('result of RGB_distance: ' + '%.6f' % resr)
#print('result of HSV_distance: ' + '%.6f' % resh)
#print('result of MSSSIM: ' + '%.6f' % resm)
#print('result of PSNR:   ' + '%.6f' % resp)
#print('result of FSIM:   ' + '%.6f' % resf)
print('Successfully completed, please check the results in '+opt.o+'.xls')
