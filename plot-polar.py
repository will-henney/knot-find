import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np


def add_lines():
    th = np.linspace(0.0, 2.0*np.pi, 200)
    thd = np.degrees(th)
    Rout = 36.0 + 8.0*np.cos(2*th)
    Rmid = 30.0 + 4.0*np.cos(th - 1.5*np.pi)
    # R1 = np.minimum(
    #     55.0 - 23.0*np.sin(th),
    #     50.0 + 15.0*np.sin(th),
    # )
    plt.plot(thd, 1.03*Rout, 'r--')
    plt.plot(thd, 1.03*Rmid, 'b--')
    plt.plot(thd, 0.5*Rout, 'r-.')
    plt.plot(thd, 0.7*Rmid, 'b-.')
    plt.plot(thd, 50.0*np.ones_like(thd), 'g-')


h2_hdu = pyfits.open("data/H2-crop-remap-nearest.fits")[0]
oiii_hdu = pyfits.open("data/OIII-remap-nearest.fits")[0]

bbox = [-0.05, 359.95, -0.05, 89.05]

plt.subplot(211)
plt.imshow(h2_hdu.data, vmin=-2.8, vmax=100.0,
           extent=bbox,
           cmap=plt.cm.gray_r, aspect="auto",
           origin="lower", interpolation="nearest")
add_lines()
plt.xticks(np.linspace(0.0, 360.0, 9))
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
plt.xticks(np.linspace(0.0, 360.0, 9))
plt.ylabel("radius")
plt.grid()
plt.axis(bbox)

plt.savefig("polar.pdf")
