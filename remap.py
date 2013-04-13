"""
Remap a Cartesian image to polar coordinates
"""
import pyfits
import argparse



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
        default=(1995, 1896),
        help="""Center of image (in pixel units, 1-based)"""
    )

    cmd_args = parser.parse_args()

    hdu = pyfits.open(cmd_args.filename)["SCI"]

