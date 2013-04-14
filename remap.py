"""
Remap a Cartesian image to polar coordinates
"""
import argparse
import numpy as np
import astropy.io.fits as pyfits
from scipy.interpolate import griddata



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Remap a Cartesian image to polar coordinates"""
    )
    parser.add_argument(
        "filename", type=str,
        help="""Name of image file"""
    )
    parser.add_argument(
        "--center", type=int, nargs=2, metavar=("XC", "YC"),
        default=(1995, 1897),
        help="""Center of image (in pixel units, 1-based)"""
    )
    parser.add_argument(
        "--rmax", type=float, default=89.0,
        help="""Maximum radius to include in output image"""
    )
    parser.add_argument(
        "--PA", type=float, default=295.0,
        help="""PA corresponding to theta=0"""
    )
    parser.add_argument(
        "--method", type=str, default="nearest",
        choices=("nearest", "linear", "cubic"),
        help="""Interpolation mehod"""
    )

    cmd_args = parser.parse_args()

    hdu = pyfits.open(cmd_args.filename)[0]

    # Read in WCS info - convert to arcsec
    cd11 = 3600.0 * hdu.header["CD1_1"]
    cd12 = 3600.0 * hdu.header["CD1_2"]
    cd21 = 3600.0 * hdu.header["CD2_1"]
    cd22 = 3600.0 * hdu.header["CD2_2"]

    # We ignore CRPIX and use the center given on the command line
    i0, j0 = cmd_args.center
    ny, nx = hdu.data.shape
    
    # Construct 2D grids of 1-based pixel coords
    I, J = np.meshgrid(1 + np.arange(nx), 1 + np.arange(ny))
    
    # Cartesian grids in arcsec, where +Y points north
    X = cd11*(I - i0) + cd21*(J - j0)
    Y = cd12*(I - i0) + cd22*(J - j0)

    # Convert to circular coords
    R = np.sqrt(X**2 + Y**2)
    th = (np.degrees(np.arctan2(-X, Y)) - cmd_args.PA)  % 360.0

    print "R stats (min, mean, max)", R.min(), R.mean(), R.max()
    print "theta stats (min, mean, max)", th.min(), th.mean(), th.max()

    # Construct regular grid in R-th
    Rmax, thmax = cmd_args.rmax, 360.0
    dR, dth = 0.1, 0.1
    Rpts = np.arange(0.0, Rmax+dR, dR)
    thpts = np.arange(0.0, thmax, dth)
    Rgrid, thgrid = np.meshgrid(Rpts, thpts)
    print "Grid size: ", Rgrid.shape

    # Interpolate image onto new grid
    imgrid = griddata( (th.ravel(), R.ravel()), hdu.data.ravel(), 
                       (thgrid, Rgrid), 
                       cmd_args.method)

    # Save to a new FITS file
    newhdu = pyfits.PrimaryHDU(imgrid.T)
    newhdu.header.update(
        WCSNAME="(theta, R)",
        CRPIX1=1.0, CRPIX2=1.0, CRVAL1=0.0, CRVAL2=0.0, 
        CDELT1=dth, CDELT2=dR,
        CUNIT1="deg", CUNIT2="arcsec",
        CTYPE1="theta", CTYPE2="Radius"
        )
    stem, suff = cmd_args.filename.split(".")
    newhdu.writeto(
        "{}-{}-{}.{}".format(stem, "remap", cmd_args.method, suff),
        clobber=True
    )
