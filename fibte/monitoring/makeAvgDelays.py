import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import operator as o
import matplotlib.cm as cm
from fibte.misc.colorPlotConfig import ColorPlotConfig


class AvgCompletionTimes(object):
    def __init__(self, test_folder='', to_plot='all', plot_name=''):
        # What to plot: mices/elephants, all?
        self.to_plot = to_plot

        # Extra folder where algo folders reside under
        if test_folder.split('/')[-1] != 'delay':
            self.test_folder = os.path.join(test_folder, 'delay')
        else:
            self.test_folder = os.path.join(test_folder)

        # Load them
        self.measurements = self.load_measurements()
        self.plot_name = plot_name

        # Get object for color config
        self.colorConfig = ColorPlotConfig()

    def load_measurements(self):
        """
        """
        # Get pattern folders
        pattern_to_dir = {d: os.path.join(self.test_folder, d) for d in os.listdir(self.test_folder)
                          if os.path.isdir(os.path.join(self.test_folder, d))}
        measurements = {}
        for pattern, dir in pattern_to_dir.iteritems():
            algos_to_dir = {d: os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))}
            measurements[pattern] = {}
            for algo, adir in algos_to_dir.iteritems():
                delays = self.read_delays(adir)

                # mice average
                mice = [delay['measured'] for delay in delays.itervalues() if delay['type'] == 'mice']
                if mice:
                    mice_avg = np.asarray(mice).mean()
                else:
                    mice_avg = 0

                # elephant average
                elep = [delay['measured'] for delay in delays.itervalues() if delay['type'] != 'mice']
                if elep:
                    eleph_avgs = np.asarray(elep).mean()
                else:
                    eleph_avgs = 0

                # add it into measuerements
                measurements[pattern][algo] = {'mice': mice_avg, 'elephant': eleph_avgs}
        return measurements

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

    def createMatrix(self, algo_list, pattern_list):
        # Create matrix from experiment data dictionary: (algo, pattern, value)
        matrix = []
        for pattern, pdata in self.measurements.iteritems():
            if pattern_list:
                if pattern in pattern_list:
                    for algo, palgo in pdata.iteritems():
                        if algo_list:
                            if algo in algo_list:
                                if self.to_plot in ['mice', 'elephant']:
                                    matrix.append([algo, pattern, palgo[self.to_plot]])

                                elif self.to_plot == 'together':
                                    value = palgo.get('mice', 0) + palgo.get('elephant', 0)
                                    matrix.append([algo, pattern, value])
                        else:
                            if self.to_plot in ['mice', 'elephant']:
                                matrix.append((algo, pattern, palgo[self.to_plot]))
                            elif self.to_plot == 'together':
                                value = palgo.get('mice', 0) + palgo.get('elephant', 0)
                                matrix.append([algo, pattern, value])
            else:
                for algo, palgo in pdata.iteritems():
                    if algo_list:
                        if algo in algo_list:
                            if self.to_plot in ['mice', 'elephant']:
                                matrix.append([algo, pattern, palgo[self.to_plot]])

                            elif self.to_plot == 'together':
                                value = palgo.get('mice', 0) + palgo.get('elephant', 0)
                                matrix.append([algo, pattern, value])
                    else:
                        if self.to_plot in ['mice', 'elephant']:
                            matrix.append((algo, pattern, palgo[self.to_plot]))
                        elif self.to_plot == 'together':
                            value = palgo.get('mice', 0) + palgo.get('elephant', 0)
                            matrix.append([algo, pattern, value])

        # Convert into np array
        matrix = np.asarray(matrix)
        return matrix

    def plotData(self, algo_list=[], pattern_list=[],):
        """"""
        matrix = self.createMatrix(algo_list, pattern_list)

        # Start figure
        fig = plt.figure(figsize=(17.5, 7))

        # Set title
        # fig.suptitle("TCP total completion times", fontsize=20, weight='bold')
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2.5)
        ax.spines["left"].set_linewidth(2.5)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # ax.set_xlabel("Traffic pattern", size='x-large', weight='bold')
        ax.set_ylabel("Average delay [s]", size='x-large', weight='bold')
        plt.tick_params(axis="both", which="both", bottom="on", top="off", labelbottom="on", left="off", right="off",
                        labelleft="on")
        plt.xticks(fontsize=18, weight="bold")
        plt.yticks(fontsize=18, weight="bold")

        # Generate bar plot
        self._barplot(ax, matrix, log=False)

        # Set fontsize and weight of axis ticks
        plt.xticks(fontsize=16, weight="bold")
        plt.yticks(fontsize=16, weight="bold")

        add_bottom_legend = True
        if add_bottom_legend:
            # Add a legend
            handles, labels = ax.get_legend_handles_labels()

            # Put a legend below current axis
            ax.legend(handles[:], labels[:], loc='upper left', shadow=True, fontsize='x-large', ncol=len(matrix))

        plt.grid(True)
        ax.grid(zorder=4)
        plt.tight_layout()
        fig.subplots_adjust(left=0.08, right=0.97)
        if self.plot_name:
            filename = '{0}.pdf'.format(self.plot_name)
        else:
            filename = 'avgCompletionTimes.pdf'
        filename = os.path.join(self.test_folder, filename)
        plt.savefig(filename)
        print ("*** Plot saved --> {0}".format(filename))

    def _barplot(self, ax, matrix, log=False):
        '''
        Create a barchart for data across different patterns with
        multiple algos for each category.

        @param ax: The plotting axes from matplotlib.
        @param matrix: The data set as an (n, 3) numpy array
        '''
        # Aggregate the conditions and the categories according to their
        # mean values

        # Aggregate the algos and the patterns according to their mean values
        algos = [(c, np.mean(matrix[matrix[:, 0] == c][:, 2].astype(float))) for c in np.unique(matrix[:, 0])]
        patterns = [(c, np.mean(matrix[matrix[:, 1] == c][:, 2].astype(float))) for c in np.unique(matrix[:, 1])]

        # Sort the algos, patterns and data so that the bars in
        # the plot will be ordered by category and condition
        algos = [c[0] for c in sorted(algos, key=o.itemgetter(1))]
        patterns = [c[0] for c in sorted(patterns, key=o.itemgetter(1))]

        # Extract the completion time values
        matrix = np.array(sorted(matrix, key=lambda x: patterns.index(x[1])))

        # Set the space between each set of bars
        space = 0.2
        n = len(algos)
        width = (1 - space) / (len(algos))

        # Create a set of bars at each position
        for i, algo in enumerate(algos):
            indeces = range(1, len(patterns) + 1)
            vals = matrix[matrix[:, 0] == algo][:, 2].astype(np.float)
            pos = [j - (1 - space) / 2. + i * width for j in indeces]
            color = self.colorConfig.getColor(algo)

            if not log:
                ax.bar(pos, vals, width=width, label=algo, color=color, alpha=1, zorder=40, edgecolor="none", linewidth=0)
            else:
                ax.bar(pos, vals, width=width, label=algo, color=color, alpha=1, zorder=40, log=1,
                       edgecolor="none", linewidth=0)

        # Set the x-axis tick labels to be equal to the patterns
        ax.set_xticks(indeces)
        ax.set_xticklabels(patterns)

        plt.setp(plt.xticks()[1], rotation=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Declare expected arguments

    parser.add_argument('--algo_list', nargs='+', help='List of measurement files to compare', type=str, default=[])
    parser.add_argument('--pattern_list', nargs='+', help='List of measurement files to compare', type=str, default=[])
    parser.add_argument('--test_folder', help='Folder in which the experiment folders reside in', type=str, default='./results/delay/')
    parser.add_argument('--to_plot', choices=['elephant', 'mice', 'together'], help='What to plot', type=str, default='mice')
    parser.add_argument('--plot_name', help="Name of the final plot", type=str, default='')

    # Parse arguments
    args = parser.parse_args()

    # Start object and load measurement files
    ac = AvgCompletionTimes(test_folder=args.test_folder, to_plot=args.to_plot, plot_name=args.plot_name)
    ac.plotData(args.algo_list, args.pattern_list)