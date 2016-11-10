
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Declare expected arguments
    parser.add_argument('--file', help='file to crop', type=str, required=True)
    parser.add_argument('--time', help='length of the experiment', required=True, type=int)
    parser.add_argument('--out', help='output file name')

    # Parse arguments
    args = parser.parse_args()

    # Open file
    with open(args.file, 'r') as f:
        lines = f.readlines()

    if not args.out:
        outfile = args.file.split('.')[0] + '_cropped.txt'
    else:
        outfile = args.out

    # Crop
    with open(outfile, 'w') as f:
        for line in lines[:args.time]:
            f.write(line)