import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from butter_filter import butter_highpass_filter


def Rout(th):
    return 36.0 + 8.0*np.cos(2*np.radians(th))


def Rmid(th):
    return 30.0 + 4.0*np.cos(np.radians(th) - 1.5*np.pi)


def add_lines():
    th = np.degrees(np.linspace(0.0, 2.0*np.pi, 200))
    plt.plot(th, 1.03*Rout(th), 'r--')
    plt.plot(th, 1.03*Rmid(th), 'b--')
    plt.plot(th, 0.5*Rout(th), 'r-.')
    plt.plot(th, 0.7*Rmid(th), 'b-.')
    plt.plot(th, 50.0*np.ones_like(th), 'g-')


def make_knot_mask(Th, R):
    mask = ((R < np.maximum(1.03*Rout(Th), 1.03*Rmid(Th))) &
            (R > np.minimum(0.5**Rout(Th), 0.7*Rmid(Th))))
    return mask


def make_spoke_mask(Th, R):
    mask = ((R > np.maximum(1.03*Rout(Th), 1.03*Rmid(Th))) &
            (R < 50.0))
    return mask


def make_grids(hdr):
    th_pts = hdr["CRVAL1"] + (1.0 + np.arange(hdr["NAXIS1"]) -
                              hdr["CRPIX1"]) * hdr["CDELT1"]
    r_pts = hdr["CRVAL2"] + (1.0 + np.arange(hdr["NAXIS2"]) -
                             hdr["CRPIX2"]) * hdr["CDELT2"]
    return th_pts, r_pts


h2_hdu = pyfits.open("data/H2-crop-remap-nearest.fits")[0]
oiii_hdu = pyfits.open("data/OIII-remap-nearest.fits")[0]

theta, radius = make_grids(h2_hdu.header)
Theta, Radius = np.meshgrid(theta, radius)

bbox = [theta.min() - 0.05, theta.max() + 0.05,
        radius.min() - 0.05, radius.max() + 0.05]

# Every 30 degrees
th_ticks_30 = np.linspace(0.0, 360.0, 13)
# Every 5 degrees
th_ticks_5 = np.linspace(0.0, 360.0, 73)

plt.subplot(211)
plt.imshow(h2_hdu.data, vmin=-2.8, vmax=100.0,
           extent=bbox,
           cmap=plt.cm.gray_r, aspect="auto",
           origin="lower", interpolation="nearest")
add_lines()
plt.xticks(th_ticks_30)
plt.ylabel("radius")
plt.title("H_2 and [O III]")
plt.grid()
plt.axis(bbox)

plt.subplot(212)
plt.imshow(np.log10(oiii_hdu.data), vmin=-2.0, vmax=np.log10(3.0),
           extent=bbox,
           cmap=plt.cm.gray_r, aspect="auto",
           origin="lower", interpolation="nearest")
add_lines()
plt.xlabel("theta")
plt.xticks(th_ticks_30)
plt.ylabel("radius")
plt.grid()
plt.axis(bbox)

plt.savefig("polar.pdf")

plt.clf()
kmask = make_knot_mask(Theta, Radius)
smask = make_spoke_mask(Theta, Radius)

sbright = np.sum(h2_hdu.data*smask, axis=0)/np.sum(smask, axis=0)
kbright = np.sum(h2_hdu.data*kmask, axis=0)/np.sum(kmask, axis=0)
# kbright = np.max(h2_hdu.data*kmask, axis=0)

sbright /= sbright.mean()
kbright /= kbright.mean()

# Plot the original data
plt.subplot(211)
plt.plot(theta, kbright, label="knots")
plt.plot(theta, sbright, label="spokes")
plt.xticks(th_ticks_5)
plt.minorticks_on()
plt.xlabel("theta")
plt.ylabel("brightness")
plt.grid(axis='y')
plt.grid(which='minor', axis='x', alpha=0.3, linestyle='-', linewidth=0.1)
plt.grid(which='major', axis='x', alpha=0.6, linestyle='-', linewidth=0.1)
plt.legend()
plt.axis("tight")
plt.xlim(0.0, 360.0)
plt.ylim(0.0, 3.0)

# Filter out the low frequencies
fs = 10.0  # sampling rate in 1/deg
# smooth structures with sizes in degrees larger than this
smooth_scale = 4.0
lowcut = 1./smooth_scale
sbright = butter_highpass_filter(sbright, lowcut, fs)
kbright = butter_highpass_filter(kbright, lowcut, fs)

print kbright.min(), kbright.mean(), kbright.max(), kbright.std()
print sbright.min(), sbright.mean(), sbright.max(), sbright.std()

# normalize by std
sbright /= 2*sbright.std()
kbright /= kbright.std()

plt.subplot(212)
plt.plot(theta, kbright, label="knots")
plt.plot(theta, sbright, label="spokes")
plt.fill_between(theta, kbright*sbright,
                 where=kbright*sbright > 0.0,
                 color="r", alpha=0.5, label="product")
plt.fill_between(theta, kbright*sbright,
                 where=kbright*sbright < 0.0,
                 color="m", alpha=0.7, label="product")
plt.xticks(th_ticks_5)
plt.minorticks_on()
plt.xlabel("theta")
plt.ylabel("brightness")
plt.grid(axis='y')
plt.grid(which='minor', axis='x', alpha=0.3, linestyle='-', linewidth=0.1)
plt.grid(which='major', axis='x', alpha=0.6, linestyle='-', linewidth=0.1)
plt.legend()
plt.axis("tight")
plt.xlim(0.0, 360.0)
plt.ylim(-2.0, 2.0)
plt.gcf().set_size_inches((50, 12))
plt.savefig("knot-spoke.pdf")

