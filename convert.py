import nibabel as  nib
import numpy as np
import imageio
from pathlib import Path


def greyscale(data, clipmax=512, clipmin=-512):
    data = np.clip(data, clipmin, clipmax)
    mx = np.max(data)
    mn = np.min(data)
    ndata = (data - mn) / max(mx - mn, 1e-3)
    return (ndata*255).astype(np.uint8)

# Loads the image and segmentation from the specified case folder
def loadcase(path):
    vol = nib.load(str(path / "imaging.nii.gz"))
    seg = nib.load(str(path  / "segmentation.nii.gz"))
    return vol, seg

def case2png(case):
    case_name = "case_%05d" % (case)
    path = "dataset/" + case_name
    vol, seg = loadcase(Path(path))
    vol_data = vol.get_fdata()
    seg_data = seg.get_fdata()
    num_slices = vol_data.shape[0]
    for i in range(num_slices):
        vol_slice = vol_data[i,:,:]
        seg_slice = seg_data[i,:,:]
        imageio.imwrite("image/%s_%03d.jpg" % (case_name, i), greyscale(vol_slice))
        imageio.imwrite("seg/%s_%03d_seg.jpg" % (case_name, i), greyscale(seg_slice))

if __name__ == "__main__":
    for i in range(589):
        try:
            print("Converting case %05d" % (i))
            case2png(i)
        except:
            continue
