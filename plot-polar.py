from __future__ import print_function
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
import scipy.stats
import bottleneck as bn
from gauss_filter import gauss_highpass_filter, gauss_lowpass_filter


def Rout(th):
    """Principal outer boundary of knotty ring"""
    return 36.0 + 8.0*np.cos(2*np.radians(th))


def Rmid(th):
    """Subsidiary outer boundary of knotty ring"""
    return 30.0 + 4.0*np.cos(np.radians(th) - 1.5*np.pi)


def Rin1(th):
    """Inner boundary of knots in quadrants 1 and 2"""
    return 25.0 + 11.0*np.cos(np.radians(th) - 1.5*np.pi)


def Rin2(th):
    """Inner boundary of knots in quadrants 3 and 4"""
    return 26.0 + 9.0*np.cos(np.radians(th) - 0.5*np.pi)


def Rin(th):
    """Combined inner boundary"""
    return np.minimum(Rin1(th), Rin2(th))


def Rkmax(th):
    """Maximum radius of knot region"""
    return np.maximum(1.05*Rout(th), 1.03*Rmid(th))


def Rkmin(th):
    """Minimum radius of knot region"""
    return np.minimum(22.0, Rin(th))


def Rsmax(th):
    """Maximum radius of spoke region"""
    return 10.0 + Rkmax(th)


def Rsmin(th):
    """Minimum radius of spoke region"""
    return Rkmax(th)


def add_lines():
    """Draw boundaries of knot and spoke regions"""
    th = np.degrees(np.linspace(0.0, 2.0*np.pi, 200))
    plt.plot(th, Rkmax(th), 'r-')
    plt.plot(th, Rkmin(th), 'b-')
    plt.plot(th, Rsmax(th), 'g-')


def make_knot_mask(Th, R):
    return (R < Rkmax(Th)) & (R > Rkmin(Th))


def make_spoke_mask(Th, R):
    return (R < Rsmax(Th)) & (R > Rsmin(Th))


def make_grids(hdr):
    th_pts = hdr["CRVAL1"] + (1.0 + np.arange(hdr["NAXIS1"]) -
                              hdr["CRPIX1"]) * hdr["CDELT1"]
    r_pts = hdr["CRVAL2"] + (1.0 + np.arange(hdr["NAXIS2"]) -
                             hdr["CRPIX2"]) * hdr["CDELT2"]
    return th_pts, r_pts


h2_hdu = pyfits.open("data/H2-remap-nearest.fits")[0]
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


# Filter out the low frequencies
fs = 10.0  # sampling rate in 1/deg
# smooth structures with sizes in degrees larger than this
smooth_scale = 5.0

# plot filtered images
plt.clf()
plt.subplot(211)
h2_filtered = gauss_highpass_filter(
    np.log10(5.0 + h2_hdu.data), smooth_scale, fs
)
plt.imshow(h2_filtered, vmin=-0.2, vmax=0.25,
           extent=bbox,
           cmap=plt.cm.gray_r, aspect="auto",
           origin="lower", interpolation="nearest")
add_lines()
plt.xticks(th_ticks_30)
plt.ylabel("radius")
s = "H_2 and [O III] high-pass filtered at {} degrees"
plt.title(s.format(int(smooth_scale)))
plt.grid()
plt.axis(bbox)

plt.subplot(212)
oiii_filtered = gauss_highpass_filter(
    np.log10(0.01 + oiii_hdu.data), smooth_scale, fs
)
plt.imshow(oiii_filtered, vmin=-0.3, vmax=0.4,
           extent=bbox,
           cmap=plt.cm.gray_r, aspect="auto",
           origin="lower", interpolation="nearest")
add_lines()
plt.xlabel("theta")
plt.xticks(th_ticks_30)
plt.ylabel("radius")
plt.grid()
plt.axis(bbox)

plt.savefig("polar-filtered.pdf")


plt.clf()
kmask = make_knot_mask(Theta, Radius)
smask = make_spoke_mask(Theta, Radius)


def masked_median_by_column(data, mask):
    """Calculate the masked median of each column of data

    Only consider elements where mask is true
    """
    return np.ma.median(
        np.ma.array(data, mask=~mask),
        axis=0
    ).filled(0.0)


def masked_mean_by_column(data, mask):
    """Calculate the masked median of each column of data

    Only consider elements where mask is true
    """
    return np.sum(data*mask, axis=0)/np.sum(mask, axis=0)


def masked_sum_by_column(data, mask):
    """Calculate the masked sum of each column of data

    Only consider elements where mask is true
    """
    return np.sum(data*mask, axis=0)


def masked_max_by_column(data, mask):
    """Calculate the masked maximum of each column of data

    Only consider elements where mask is true
    """
    return np.max(data*mask, axis=0)


def masked_centile_by_column(data, mask, centile=90):
    """Calculate the masked centile of each column of data

    Only consider elements where mask is true
    """
    return scipy.stats.mstats.mquantiles(
        np.ma.array(data, mask=~mask),
        prob=[centile/100.],
        axis=0
    )[0].filled(0.0)


# sbright = np.sum(h2_hdu.data*smask, axis=0)/np.sum(smask, axis=0)
# # kbright = np.sum(h2_hdu.data*kmask, axis=0)/np.sum(kmask, axis=0)
# kbright = np.max(h2_hdu.data*kmask, axis=0)

h2data = np.log10(5.0 + h2_hdu.data)
sbright = masked_mean_by_column(h2data, smask & np.isfinite(h2data))
kbright = masked_centile_by_column(h2data, kmask & np.isfinite(h2data))

sbright /= bn.nanmean(sbright)
kbright /= bn.nanmean(kbright)

# fill in any gaps in the data
sbright[~np.isfinite(sbright)] = 1.0
kbright[~np.isfinite(kbright)] = 1.0

# Plot the original data
plt.subplot(211)
plt.plot(theta, kbright, label="knots")
plt.plot(theta, sbright, label="spokes")
plt.fill_between(theta, kbright,
                 gauss_lowpass_filter(kbright, smooth_scale, fs),
                 alpha=0.3, color="b")
plt.fill_between(theta, sbright,
                 gauss_lowpass_filter(sbright, smooth_scale, fs),
                 alpha=0.3, color="g")
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
plt.title("Normalized average brightness profiles versus angle")
# plt.ylim(0.0, 3.0)

sbright = gauss_highpass_filter(sbright, smooth_scale, fs)
kbright = gauss_highpass_filter(kbright, smooth_scale, fs)

print(bn.nanmin(kbright), bn.nanmean(kbright),
      bn.nanmax(kbright), bn.nanstd(kbright))
print(bn.nanmin(sbright), bn.nanmean(sbright),
      bn.nanmax(sbright), bn.nanstd(sbright))

# normalize by std
sbright /= bn.nanstd(sbright)
kbright /= bn.nanstd(kbright)

# positive and negative correlations
corr = kbright*sbright
pmask = corr > 0.0
nmask = ~pmask

print()
print(*["------"]*6, sep="\t")
print("Octant", "Th_1", "Th_2", "Pos", "Neg", "Diff", sep="\t")
print(*["------"]*6, sep="\t")
for ioctant in range(8):
    joctant = (theta/45.0).astype(int)
    thmask = joctant == ioctant
    print(ioctant, 45*ioctant, 45*(ioctant+1),
          int(corr[pmask & thmask].sum()),
          int(corr[nmask & thmask].sum()),
          int(corr[thmask].sum()),
          sep="\t")
print(*["------"]*6, sep="\t")
print("all", 0, 360,
      int(corr[pmask].sum()),
      int(corr[nmask].sum()),
      int(corr.sum()),
      sep="\t")
print(*["------"]*6, sep="\t")

plt.subplot(212)
plt.plot(theta, kbright, label="knots")
plt.plot(theta, sbright, label="spokes")
plt.fill_between(theta, corr, where=pmask,
                 color="r", alpha=0.5, label="positive")
plt.fill_between(theta, corr, where=nmask,
                 color="m", alpha=0.7, label="negative")
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
plt.ylim(-4.0, 4.0)
plt.title("Filtered brightness profiles versus angle")
plt.gcf().set_size_inches((50, 12))
plt.savefig("knot-spoke.pdf")
