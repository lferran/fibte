import os
import numpy as np
import matplotlib.pyplot as plt


algo_styles = {'ideal':  {'color': 'grey', 'linestyle':'-.-', 'linewidth': 2.0},
               'non-blocking': {'color': 'black', 'linestyle':'--', 'linewidth': 3.0},
               'ecmp': {'color': 'mediumturquoise', 'linestyle':'-', 'linewidth': 2.0},
               'mice-dag-shifter': {'color':'darkgoldenrod', 'linestyle':'-', 'linewidth': 2.0},
               'elephant-dag-shifter-best': {'color':'firebrick', 'linestyle':'-', 'linewidth': 2.0},
               'elephant-dag-shifter-sample': {'color': 'orengered', 'linestyle': '-', 'linewidth': 2.0},
               'full-dag-shifter-best': {'color': 'black', 'linestyle': '-', 'linewidth': 2.0},
               'full-dag-shifter-sample': {'color': 'pink', 'linestyle': '-', 'linewidth': 2.0},
               }

class DelaysComparator(object):
    def __init__(self, k=4, algo_list=[]):
        # FatTree parameter
        self.k = k

        # List of files for different algorithm measurements
        self.algo_list = algo_list

        # Results folder
        self.delay_dir = os.path.join(os.path.dirname(__file__), 'results/delay/')

        # Load them
        self.algo_to_delays = self.load_measurements()

    def extractAlgoNameFromFilename(self, filename):
        if 'ecmp' in filename:
            return 'ecmp'
        elif 'elephant-dag-shifter' in filename:
            if 'True' in filename:
                return 'elephant-dag-shifter-sample'
            else:
                return 'elephant-dag-shifter-best'
        elif 'mice-dag-shifter' in filename:
            return 'mice-dag-shifter-best'

        elif 'full-dag-shifter' in filename:
            if 'True' in filename:
                return 'full-dag-shifter-sample'
            else:
                return 'full-dag-shifter-best'

    def load_measurements(self):
        """
        Given a list of filenames, return a dictionary keyed by algorithm.
        Each value is an ordered list of tuples (time, measurement).
        Each measurement is a dictionary with the correspondign edges

        :param file_list:
        :return:
        """
        algo_to_measurements = {}

        for algo in self.algo_list:

            # Extract algorithm name
            foldername = self.delay_dir + algo + '/'

#            algo = self.extractAlgoNameFromFilename(filename)

#            if algo == '': algo = 'None'

            # Extract measurements
            delays = self.read_delays(foldername)

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
            (src, sport, dst, dport) = flowfile.split('_')
            flow = (src, sport, dst, dport)
            with open(folder+flowfile, 'r') as f:
                expected = float(f.readline().strip('\n').split(' ')[1])
                starttime_ms = float(f.readline().strip('\n'))
                try:
                    endtime_ms = float(f.readline().strip('\n'))
                except:
                    print("Found flow that couldn't be completely sent!")
                    continue
                else:
                    flows_to_delay[flow] = {'expected': expected, 'measured': (endtime_ms - starttime_ms)/1000.0}

        return flows_to_delay

    def plot_delay_distribution_comparison(self):
        fig = plt.figure(figsize=(80, 20))
        fig.suptitle("CDF of mice completion times", fontsize=20)
        plt.xlabel("Completion time (s)", fontsize=16)
        plt.ylabel("Percentage of flows", fontsize=16)
        plt.ylim([0, 1.05])

        # Print expected first
        algo = 'ideal'
        delays = self.algo_to_delays[self.algo_list[0]]
        expecteds = [vs['expected'] - 1 for f, vs in delays.iteritems()]

        # Returns values corresponding to each bin and bins
        values, bins = np.histogram(expecteds, range=(0, 20), bins=10000)

        # Compute CDF
        cumulative = np.cumsum(values, dtype=float)

        # Normalize wrt number of flows
        cumulative /= float(len(expecteds))

        # Plot it!
        label = 'ideal'
        color = algo_styles[label]['color']
        linestyle = algo_styles[label]['linestyle']
        linewidth = algo_styles[label]['linewidth']
        plt.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

        for i, algo in enumerate(self.algo_list):
            import ipdb; ipdb.set_trace()
            delays = self.algo_to_delays[algo]
            measured = [vs['measured'] for f, vs in delays.iteritems()]
            values, bins = np.histogram(measured, range=(0, 20), bins=10000)

            # Compute CDF
            cumulative = np.cumsum(values, dtype=float)

            # Normalize wrt number of flows
            cumulative /= float(len(measured))

            # Plot it!
            color = algo_styles[algo]['color']
            linestyle = algo_styles[algo]['linestyle']
            linewidth = algo_styles[algo]['linewidth']
            label = algo

            plt.plot(bins[:-1], cumulative, c=color, linestyle=linestyle, linewidth=linewidth, label=label)

        # Write legend and plot
        plt.legend(loc='best')

        # Set grid on
        plt.grid(True)
        plt.show()

    def plot_delay_distribution(self, algorithm):
        delays = self.algo_to_delays[algorithm]
        dys = delays.values()
        plt.hist(dys, 200)
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Declare expected arguments
    parser.add_argument('-k', help='Fat Tree parameter', type=int, default=4)
    parser.add_argument('--algo_list', nargs='+', help='List of measurement files to compare', type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Start object and load measurement files
    ac = DelaysComparator(k=args.k, algo_list=args.algo_list)
    #ac.plot_delay_distribution('ecmp')
    #ac.plot_delay_distribution('mice-dag-shifter')
    ac.plot_delay_distribution_comparison()

    import ipdb; ipdb.set_trace()