import cv2
import numpy as np
import os
from PIL import Image
import cupy as cp
import sys
from cupy.fft import fft2, ifft2, ifftshift
import pytorch_ssim
import torch
from torch.autograd import Variable
from xlwt import Workbook

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = pytorch_ssim.MSSSIM()





def compute(a, b, output,):
    resm1, resr1, resh1, resp1, resf1, resm2, resr2, resh2, resp2, resf2 = [], [], [], [], [], [], [], [], [], []
    count = 1
    wb = Workbook()
    ws = wb.add_sheet('results')

    ws.write(0, 0, 'Image')
    ws.write(0, 1, 'RGB_distance')
    ws.write(0, 2, 'HSV_distance')
    ws.write(0, 3, 'MSSSIM')
    ws.write(0, 4, 'PSNR')
    ws.write(0, 5, 'FSIM')
    for _,_,files_a in os.walk(a):
        for name_a in files_a:
            imgr1 = Image.open(a+name_a)
            imr1 = cp.asarray(imgr1, dtype=cp.float)

            for _,_,files_b in os.walk(b):
                for name_b in files_b:
                    if (b+name_b) == (a+name_a):
                        pass
                    else:
                        imgr2 = Image.open(b+name_b)
                        if imgr2.size != imgr1.size:
                            imgr2 = imgr2.resize((imgr1.size[0],imgr1.size[1]))
                        imr2 = cp.asarray(imgr2, dtype=cp.float)

                        resf1.append(FSITM(cp.array(imgr1.convert("L"), dtype=cp.float), cp.array(imgr2.convert("L"), dtype=cp.float)))
                        resp1.append(10* cp.log(255**2 / (cp.mean(cp.square(imr1 - imr2)))))
                        resm1.append(m((torch.from_numpy(np.rollaxis(np.asarray(imgr1, dtype=np.float), 2)).float().unsqueeze(0)/255.0).cuda(), (torch.from_numpy(np.rollaxis(np.asarray(imgr2, dtype=np.float), 2)).float().unsqueeze(0)/255.0).cuda()).item())
                        resr1.append(np.mean(np.sqrt(cp.linalg.norm(imr1 - imr2, axis=1))))
                        resh1.append(np.mean(np.sqrt(cp.linalg.norm(cp.asarray(imgr1.convert("HSV"), dtype=cp.float) - cp.asarray(imgr2.convert("HSV"), dtype=cp.float), axis=1))))

            ws.write(count, 0, a+name_a)
            ws.write(count, 1, np.nanmean(resr1))
            ws.write(count, 2, np.nanmean(resh1))
            ws.write(count, 3, np.nanmean(resm1))
            ws.write(count, 4, np.nanmean(resp1))
            ws.write(count, 5, np.nanmean(resf1))

            print('epoch ' + str(count))
            count += 1
            resm1, resr1, resh1, resp1, resf1 = [], [], [], [], []




    ws.write(count+1, 0, 'Average')
    ws.write(count+1, 1, '=AVERAGE(B2:B'+str(count)+')')
    ws.write(count+1, 2, '=AVERAGE(C2:C'+str(count)+')')
    ws.write(count+1, 3, '=AVERAGE(D2:D'+str(count)+')')
    ws.write(count+1, 4, '=AVERAGE(E2:E'+str(count)+')')
    ws.write(count+1, 5, '=AVERAGE(F2:F'+str(count)+')')
    wb.save(output+'.xls')

'''
            print(a+name_a)
            resr2.append(np.nanmean(resr1))
            print(resr2[count-1])
            resh2.append(np.nanmean(resh1))
            print(resh2[count-1])
            resm2.append(np.nanmean(resm1))
            print(resm2[count-1])
            resp2.append(np.nanmean(resp1))
            print(resp2[count-1])
            resf2.append(np.nanmean(resf1))
            print(resf2[count-1])'''

#    return np.mean(resr2), np.mean(resh2), np.mean(resm2), np.mean(resp2), np.mean(resf2)


def FSITM(HDR, LDR, alpha = None):

    NumPixels = LDR.size

    if alpha is None:
        r = cp.floor(NumPixels / (2. ** 18))
        if r > 1.:
            alpha = 1. - (1. / r)
        else:
            alpha = 0.

    minNonzero = cp.min(HDR[HDR > 0])
    LogH = cp.log(cp.maximum(HDR, minNonzero))

    # float is needed for further calculation
    LogH = cp.around((LogH - LogH.min()) * 255. /
                    (LogH.max() - LogH.min())).astype(cp.float)

    if alpha > 0.:
        PhaseHDR_CH = phasecong100(HDR, 2, 2, 8, 8)
        PhaseLDR_CH8 = phasecong100(LDR, 2, 2, 8, 8)
    else:  # so, if image size is smaller than 512x512?
        PhaseHDR_CH = 0
        PhaseLDR_CH8 = 0

    PhaseLogH = phasecong100(LogH, 2, 2, 2, 2)
    PhaseH = alpha * PhaseHDR_CH + (1 - alpha) * PhaseLogH

    PhaseLDR_CH2 = phasecong100(LDR, 2, 2, 2, 2)
    PhaseL = alpha * PhaseLDR_CH8 + (1 - alpha) * PhaseLDR_CH2
    Q = cp.sum(cp.logical_or(cp.logical_and(PhaseL <= 0, PhaseH <= 0),
               cp.logical_and(PhaseL > 0, PhaseH > 0))) / NumPixels
    return Q

def phasecong100(im, nscale=2,
                 norient=2,
                 minWavelength=7,
                 mult=2,
                 sigmaOnf=0.65):

    rows, cols = im.shape
    imagefft = fft2(im)
    zero = cp.zeros(shape=(rows, cols))

    EO = dict()
    EnergyV = cp.zeros((rows, cols, 3))

    x_range = cp.linspace(-0.5, 0.5, num=cols, endpoint=True)
    y_range = cp.linspace(-0.5, 0.5, num=rows, endpoint=True)

    x, y = cp.meshgrid(x_range, y_range)
    radius = cp.sqrt(x ** 2 + y ** 2)

    theta = cp.arctan2(- y, x)

    radius = ifftshift(radius)

    theta = ifftshift(theta)

    radius[0, 0] = 1.

    sintheta = cp.sin(theta)
    costheta = cp.cos(theta)

    lp = lowpass_filter((rows, cols), 0.45, 15)

    logGabor = []
    for s in range(1, nscale + 1):
        wavelength = minWavelength * mult ** (s - 1.)
        fo = 1.0 / wavelength
        logGabor.append(cp.exp(
            (- (cp.log(radius / fo)) ** 2) / (2 * cp.log(sigmaOnf) ** 2)
        ))
        logGabor[-1] *= lp
        logGabor[-1][0, 0] = 0

    # The main loop...
    for o in range(1, norient + 1):
        angl = (o - 1.) * cp.pi / norient
        ds = sintheta * cp.cos(angl) - costheta * cp.sin(angl)
        dc = costheta * cp.cos(angl) + sintheta * cp.sin(angl)
        dtheta = cp.abs(cp.arctan2(ds, dc))
        dtheta = cp.minimum(dtheta * norient / 2., cp.pi)
        spread = (cp.cos(dtheta) + 1.) / 2.
        sumE_ThisOrient = zero.copy()
        sumO_ThisOrient = zero.copy()
        for s in range(0, nscale):
            filter_ = logGabor[s] * spread
            EO[(s, o)] = ifft2(imagefft * filter_)
            sumE_ThisOrient = sumE_ThisOrient + cp.real(EO[(s, o)])
            sumO_ThisOrient = sumO_ThisOrient + cp.imag(EO[(s, o)])
        EnergyV[:, :, 0] = EnergyV[:, :, 0] + sumE_ThisOrient
        EnergyV[:, :, 1] = EnergyV[:, :, 1] + cp.cos(angl) * sumO_ThisOrient
        EnergyV[:, :, 2] = EnergyV[:, :, 2] + cp.sin(angl) * sumO_ThisOrient
    OddV = cp.sqrt(EnergyV[:, :, 0] ** 2 + EnergyV[:, :, 1] ** 2)
    featType = cp.arctan2(EnergyV[:, :, 0], OddV)
    return featType

def lowpass_filter(size=None, cutoff=None, n=None):
    """ Butterworth filter
    Examples:
    >>> lowpass_filter(3,0.5,2)
    array([[1. , 0.5, 0.5],
           [0.5, 0.2, 0.2],
           [0.5, 0.2, 0.2]])
    """
    if type(size) == int:
        rows = cols = size
    else:
        rows, cols = size
    x_range = cp.linspace(-0.5, 0.5, num=cols)
    y_range = cp.linspace(-0.5, 0.5, num=rows)
    x, y = cp.meshgrid(x_range, y_range)
    radius = cp.sqrt(x ** 2 + y ** 2)
    f = ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))
    return f
