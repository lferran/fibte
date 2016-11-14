import os
import numpy as np
import matplotlib.pyplot as plt

from fibte.monitoring import algo_styles

class DelaysComparator(object):
    def __init__(self, algo_list=[]):
        # Results folder
        self.delay_dir = os.path.join(os.path.dirname(__file__), 'results/delay/')

        # List of files for different algorithm measurements
        self.algo_list = algo_list
        self.algos = [a if self.delay_dir not in a else a.split('/')[-1] if a.split('/')[-1] else a.split('/')[-2] for a in self.algo_list]
        self.algo_dirs = [a if self.delay_dir in a else os.path.join(self.delay_dir, a) for a in self.algo_list]
        self.algo_to_dir = {v: self.algo_dirs[i] for i, v in enumerate(self.algos)}

        # Load them
        self.algo_to_delays = self.load_measurements()

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
            (src, sport, dst, dport) = flowfile.split('_')
            flow = (src, sport, dst, dport)
            with open(os.path.join(folder, flowfile), 'r') as f:
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

    def plot_delay_distribution_comparison(self, ideal=True):
        """"""
        fig = plt.figure(figsize=(80, 20))
        #fig.suptitle("CDF of flow completion times", fontsize=20)
        plt.xlabel("Completion time (s)", fontsize=16)
        plt.ylabel("Percentage of flows", fontsize=16)
        plt.ylim([0, 1.05])

        if ideal:
            # Print expected first
            algo = 'ideal'
            delays = self.algo_to_delays[self.algos[0]]
            expecteds = [vs['expected'] - 1 for f, vs in delays.iteritems()]

            # Returns values corresponding to each bin and bins
            values, bins = np.histogram(expecteds, bins=10000)

            # Compute CDF
            cumulative = np.cumsum(values, dtype=float)

            # Normalize wrt number of flows
            cumulative /= float(len(expecteds))

            # Plot it!
            label = 'Ideal'
            color = algo_styles[label]['color']
            linestyle = algo_styles[label]['linestyle']
            linewidth = algo_styles[label]['linewidth']

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
    parser.add_argument('--algo_list', nargs='+', help='List of measurement files to compare', type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Start object and load measurement files
    ac = DelaysComparator(algo_list=args.algo_list)
    ac.plot_delay_distribution_comparison()