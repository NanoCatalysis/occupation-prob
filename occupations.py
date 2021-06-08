#!/usr/bin/python

# MIT License
#
# Copyright (c) 2021 Luis Gálvez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard library imports
import argparse

#Third party imports
from ase.io import read

# Local imports
from functions import read_properties, calc_prob, plot_prob

def main():
    # Command line options
    parser = argparse.ArgumentParser(description='Caculates the occupation probabilities of the isomers given in an Extended XYZ file.')
    parser.add_argument('in_file', help='Path to Extended XYZ file contaning the list of isomers')
    parser.add_argument('max_temp', help='Maximum temperature', type=float)
    parser.add_argument('--n_temp', help='Number of divisions in the temperature range (default 1000)', type=int)
    parser.add_argument('--out_file', help='Path to output image file (default "Probabilities.pdf")')
    parser.add_argument('--size', help='Width and height of the output image, in inches (default 8.0 6.0)', type=float, nargs='+')
    args = parser.parse_args()
    
    filename = args.in_file # Input file
    max_temp = args.max_temp # Maximum temperature

    # Temperature range divisions    
    if args.n_temp:
        n_temp = args.n_temp
    else:
        n_temp = 1000
    
    # Output file
    if args.out_file:
        outfile = args.out_file
    else:
        outfile = 'Probabilities.pdf'

    # Output image size
    if args.size:
        size = tuple(args.size)
        print(size)
    else:
        size = (8.0, 6.0)
    
    # Reads the input Extended XYZ file
    isomers = read(filename, index=':')
    
    # Extracts the required properties
    energies, symm_order, spin_m, freq, moments = read_properties(isomers)
    
    # Converts energy units from eV to a.u.
    energies /= 13.6056980659
    
    # Calculates the occupation probabilities for the temperature range given
    temp, prob = calc_prob(energies, symm_order, spin_m, freq, moments, max_temp, n_temp)

    # Plots the occupation probabilities and saves the results in the output file
    plot_prob(temp, prob, outfile, size)

if __name__ == '__main__':
    main()
