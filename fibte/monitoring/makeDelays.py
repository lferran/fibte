import os
import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import six
from matplotlib import colors

from fibte.monitoring import algo_styles

color_list = list(six.iteritems(colors.cnames))

class DelaysComparator(object):
    def __init__(self, algo_list=[], parent_folder='', to_plot='all', plot_name=''):
        # What to plot?
        self.to_plot = to_plot

        # Results folder
        self.delay_dir = os.path.join(os.path.dirname(__file__), 'results/delay/')

        # Extra folder where algo folders reside under
        self.parent_folder = parent_folder

        # List of files for different algorithm measurements
        self.algo_list = algo_list
        self.algos = [a for a in self.algo_list]
        self.algo_dirs = [os.path.join(self.parent_folder, a) for a in self.algos]
        self.algo_to_dir = {v: self.algo_dirs[i] for i, v in enumerate(self.algos)}

        # Load them
        self.algo_to_delays = self.load_measurements()
        self.color_index = 0
        self.plot_name = plot_name

    def extractAlgoNameFromFilename(self, filename):
        if 'ecmp' in filename.lower():
            return 'ECMP'

        elif 'elephant-dag-shifter' in filename.lower():
            if 'True' in filename:
                return 'Elephant-DAG-Shifter-Sample'
            else:
                return 'Elephant-DAG-Shifter-Best'

        elif 'mice-dag-shifter' in filename.lower():
            return 'Mice-DAG-Shifter-Best'

        elif 'full-dag-shifter' in filename.lower():
            if 'True' in filename:
                return 'Full-DAG-Shifter-Sample'
            else:
                return 'Full-DAG-Shifter-Best'

    def load_measurements(self):
        """
        Given a list of filenames, return a dictionary keyed by algorithm.
        Each value is an ordered list of tuples (time, measurement).
        Each measurement is a dictionary with the correspondign edges

        :param file_list:
        :return:
        """
        algo_to_measurements = {}

        for algo, algodir in self.algo_to_dir.iteritems():

            # Extract measurements
            delays = self.read_delays(algodir)

            # Add it to dict
            algo_to_measurements[algo] = delays

        return algo_to_measurements

    def read_delays(self, folder):
        """Returns a dictionary keyed by measuring time, containing
        the link loads readouts for the aggregation core links at
        specified times.
        """
        flows_to_delay = {}
        for flowfile in os.listdir(folder):
            fields = flowfile.split('_')
            if len(fields) == 5:
                (ftype, src, sport, dst, dport) = fields
                round = 0
            elif len(fields) == 6:
                (ftype, src, sport, dst, dport, round) = fields
            else:
                raise ValueError

            flow = (src, sport, dst, dport, round)
            with open(os.path.join(folder, flowfile), 'r') as f:
                expected = float(f.readline().strip('\n').split(' ')[1])
                starttime_ms = float(f.readline().strip('\n'))
                try:
                    endtime_ms = float(f.readline().strip('\n'))
                except:
                    print("Found flow that couldn't be completely sent!")
                    continue
                else:
                    flows_to_delay[flow] = {'type': ftype, 'expected': expected, 'measured': (endtime_ms - starttime_ms)/1000.0}

        return flows_to_delay

    def get_algo_style(self, algo, index=None):
        try:
            color = algo_styles[algo]['color']
            linestyle = algo_styles[algo]['linestyle']
            linewidth = algo_styles[algo]['linewidth']
            return color, linestyle, linewidth, None
        except KeyError:
            if index == None:
                index = self.color_index
                color = color_list[self.color_index][1]
                linestyle = '-'
                linewidth = 2
                self.color_index += 1
                return color, linestyle, linewidth, index
            else:
                color = color_list[index][1]
                linestyle = '-'
                linewidth = 2
                return color, linestyle, linewidth, index

    def plot_delay_distribution_comparison(self, ideal=True):
        """"""
        if self.to_plot != 'all' and self.to_plot != 'together':
            if self.to_plot == 'elephant':
                ftype = 'elep'
            else:
                ftype = 'mice'

            fig = plt.figure()
            #fig.suptitle("CDF of flow completion times", fontsize=20)
            plt.xlabel("Completion time (s) [{0}s]".format(self.to_plot), fontsize=16)
            plt.ylabel("Percentage of flows", fontsize=16)
            plt.ylim([0, 1.05])

            if ideal:
                # Print expected first
                algo = 'ideal'
                delays = self.algo_to_delays[self.algos[0]]
                expecteds = [vs['expected'] for f, vs in delays.iteritems() if vs['type'] == ftype]

                # Returns values corresponding to each bin and bins
                values, bins = np.histogram(expecteds, bins=10000)

                # Compute CDF
                cumulative = np.cumsum(values, dtype=float)

                # Normalize wrt number of flows
                cumulative /= float(len(expecteds))

                # Plot it!
                label = 'Ideal'
                color, linestyle, linewidth, _ = self.get_algo_style(label)

                plt.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

            for algo in self.algos:
                delays = self.algo_to_delays[algo]
                measured = [vs['measured'] for f, vs in delays.iteritems() if vs['type'] == ftype]

                # Compute histogram
                values, bins = np.histogram(measured, bins=10000)

                # Compute CDF of the histogram heights
                cumulative = np.cumsum(values, dtype=float)

                # Normalize wrt number of flows
                cumulative /= float(len(measured))

                # Plot it!
                label = algo
                color, linestyle, linewidth, _ = self.get_algo_style(algo)
                plt.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

            # Write legend and plot
            plt.legend(loc='best')

            # Set grid on
            plt.grid(True)
            plt.show()

        elif self.to_plot == 'all':
            types = ['elephant', 'mice']
            algo_to_cols = {}

            f, axarr = plt.subplots(2)#, sharex=True)
            for index, ax in enumerate(axarr):
                fftype = types[index]
                if fftype == 'elephant':
                    ftype = 'elep'
                else:
                    ftype = 'mice'
                ax.set_xlabel("Completion time (s) [{0}s]".format(fftype), fontsize=16)
                ax.set_ylabel("Percentage of flows", fontsize=16)
                ax.set_ylim([0, 1.05])
                if ideal:
                    # Print expected first
                    algo = 'ideal'
                    delays = self.algo_to_delays[self.algos[0]]
                    expecteds = [vs['expected'] for f, vs in delays.iteritems() if vs['type'] == ftype]

                    if not expecteds:
                        continue

                    # Returns values corresponding to each bin and bins
                    values, bins = np.histogram(expecteds, bins=10000)

                    # Compute CDF
                    cumulative = np.cumsum(values, dtype=float)

                    # Normalize wrt number of flows
                    cumulative /= float(len(expecteds))

                    # Plot it!
                    label = 'Ideal'
                    index = algo_to_cols.get(label, None)
                    if index == None:
                        color, linestyle, linewidth, index = self.get_algo_style(label)
                        algo_to_cols[label] = index
                    else:
                        color, linestyle, linewidth, _ = self.get_algo_style(label, index)

                    ax.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

                for algo in self.algos:
                    delays = self.algo_to_delays[algo]
                    measured = [vs['measured'] for f, vs in delays.iteritems() if vs['type'] == ftype]

                    # Compute histogram
                    values, bins = np.histogram(measured, bins=10000)

                    # Compute CDF of the histogram heights
                    cumulative = np.cumsum(values, dtype=float)

                    # Normalize wrt number of flows
                    cumulative /= float(len(measured))

                    # Plot it!
                    label = algo
                    index = algo_to_cols.get(label, None)
                    if index == None:
                        color, linestyle, linewidth, index = self.get_algo_style(label)
                        algo_to_cols[label] = index
                    else:
                        color, linestyle, linewidth, _ = self.get_algo_style(label, index)
                    ax.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

                # Write legend and plot
                ax.legend(loc='best')

                # Set grid on
                ax.grid(True)

        elif self.to_plot == 'together':
            fig = plt.figure()
            to_plot = 'mices and elephants'
            #fig.suptitle("CDF of flow completion times", fontsize=20)
            plt.xlabel("Completion time (s) [{0}]".format(to_plot), fontsize=16)
            plt.ylabel("Percentage of flows", fontsize=16)
            plt.ylim([0, 1.05])

            if ideal:
                # Print expected first
                algo = 'ideal'
                delays = self.algo_to_delays[self.algos[0]]
                expecteds = [vs['expected'] for f, vs in delays.iteritems()]

                # Returns values corresponding to each bin and bins
                values, bins = np.histogram(expecteds, bins=10000)

                # Compute CDF
                cumulative = np.cumsum(values, dtype=float)

                # Normalize wrt number of flows
                cumulative /= float(len(expecteds))

                # Plot it!
                label = 'Ideal'
                color, linestyle, linewidth, _ = self.get_algo_style(label)

                plt.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

            for algo in self.algos:
                delays = self.algo_to_delays[algo]
                measured = [vs['measured'] for f, vs in delays.iteritems()]

                # Compute histogram
                values, bins = np.histogram(measured, bins=10000)

                # Compute CDF of the histogram heights
                cumulative = np.cumsum(values, dtype=float)

                # Normalize wrt number of flows
                cumulative /= float(len(measured))

                # Plot it!
                label = algo
                color, linestyle, linewidth, _ = self.get_algo_style(algo)
                plt.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

            # Write legend and plot
            plt.legend(loc='best')

            # Set grid on
            plt.grid(True)

        plt.tight_layout()
        plt.show()
        plot_filename_png = os.path.join(self.parent_folder, '{0}.png'.format(self.plot_name))
        plot_filename_pdf = os.path.join(self.parent_folder, '{0}.pdf'.format(self.plot_name))
        plt.savefig(plot_filename_png)
        plt.savefig(plot_filename_pdf)

    def plot_delay_distribution(self, algorithm):
        delays = self.algo_to_delays[algorithm]
        dys = delays.values()
        plt.hist(dys, 200)
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Declare expected arguments
    parser.add_argument('--algo_list', nargs='+', help='List of measurement files to compare', type=str, required=True)
    parser.add_argument('--parent_folder', help='Folder in which the experiment folders reside in', type=str, default='./results/delay/')
    parser.add_argument('--to_plot', choices=['all', 'together', 'mice', 'elephant'], help="What to plot?", required=True)
    parser.add_argument('--plot_name', help="Name of the final plot", type=str, default='')
    # Parse arguments
    args = parser.parse_args()

    # Start object and load measurement files
    ac = DelaysComparator(algo_list=args.algo_list,
                          parent_folder=args.parent_folder,
                          to_plot=args.to_plot,
                          plot_name=args.plot_name)
    ac.plot_delay_distribution_comparison()