import matplotlib.pyplot as plt
import json
import os
import numpy as np
import operator as o

from fibte import tmp_files, db_topo, LINK_BANDWIDTH
from fibte.monitoring import algo_styles
from fibte.misc.topology_graph import TopologyGraph

class AlgorithmsComparator(object):
    def __init__(self, k=4, file_list=[]):
        # FatTree parameter
        self.k = k

        # List of files for different algorithm measurements
        self.file_list = file_list

        # Results folder
        self.results_folder = os.path.join(os.path.dirname(__file__), 'results/')
        self.throughput_dir = os.path.join(self.results_folder, 'throughput/')
        self.delay_dir = os.path.join(self.results_folder, 'delay/')

    def loadTopo(self):
        self.topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))

    def extractAlgoNameFromFilename(self, filename):
        if 'ecmp' in filename:
            return 'ECMP'
        elif 'elephant-dag-shifter' in filename:
            if 'True' in filename:
                return 'Elephant-DAG-Shifter-Sample'
            else:
                return 'Elephant-DAG-Shifter-Best'
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

        for filename in self.file_list:
            # Extract algorithm name
            if not self.throughput_dir in filename:
                filename = self.throughput_dir + filename

            algo = self.extractAlgoNameFromFilename(filename)

            if algo == '': algo = 'None'

            # Extract measurements
            measurements = self.read_aggregation_traffic_file(filename)

            # Add it to dict
            algo_to_measurements[algo] = measurements

        # Align and crop data so that they are comparable
        algo_to_measurements = self.align_data(algo_to_measurements)
        algo_to_measurements = self.crop_data(algo_to_measurements)

        return algo_to_measurements

    def read_aggregation_traffic_file(self, filename):
        """Returns a dictionary keyed by measuring time, containing
        the link loads readouts for the aggregation core links at
        specified times.
        """
        f = open(filename, 'r')
        lines = f.readlines()

        # Dict: measurement time -> link loads
        samples = []
        for line in lines:
            measurement = json.loads(line.strip('\n'))

            # Extract measurement time
            m_time = measurement.pop('time')

            # Insert it in samples
            samples.append((m_time, measurement))

        return samples

    def read_in_out_traffic_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        intf = []
        outf = []
        for line in lines:
            inn, outt = line.strip('\n').split('\t')

            # Insert it in samples
            intf.append(float(inn))
            outf.append(float(outt))

        return intf, outf

    def _get_core_positions(self):
        n_cores = (self.k**2)/4
        core_base = "r_c{0}"
        core_positions = {}
        plot_position = (n_cores, 1, 1)

        for i in range(n_cores):
            router_name = core_base.format(i)
            core_positions[router_name] = plot_position
            (a,b,c) = plot_position
            plot_position = (a, b, c+1)

        return core_positions

    def _get_first_change_index(self, list_of_measurements):
        """
        """
        initial_index = 0
        found = False
        for (t, measurements) in list_of_measurements:
            for a, others in measurements.iteritems():
                for o, load in others.iteritems():
                    if load['in'] > 0.005 or load['out'] > 0.005:
                        found = True
                        return initial_index

            # Increment index
            initial_index = initial_index + 1

        if not found:
            return 0

    def _get_last_change_index(self, list_of_measurements):
        found = False
        r_list_of_measurements = list_of_measurements[:]
        r_list_of_measurements.reverse()
        initial_index = 0
        for (t, measurements) in r_list_of_measurements:
            for a, others in measurements.iteritems():
                for o, load in others.iteritems():
                    if load['in'] > 0.002 or load['out'] > 0.002:
                        found = True
                        return len(list_of_measurements) - initial_index - 1 + 10

            # Increment index
            initial_index = initial_index + 1

        if not found:
            return len(list_of_measurements) - 1

    def align_data(self, algo_to_measurements):
        algos = algo_to_measurements.keys()

        earliest_flow_index = {}

        # Get the index of the earliest flow observed for each algo
        for algo, measurements in algo_to_measurements.iteritems():
            # Get the index of the earliest change in load for any router
            earliest_index = self._get_first_change_index(measurements)
            earliest_flow_index[algo] = earliest_index

        # Get the minimum
        earliest_algo = min(earliest_flow_index, key=earliest_flow_index.get)
        earliest_index = earliest_flow_index[earliest_algo]

        # Shift the others
        new_data_to_plot = {}
        for algo in algos:
            measurements = algo_to_measurements[algo]
            if algo != earliest_algo:
                index_diff = earliest_flow_index[algo] - earliest_index
                # Shift it by the difference
                new_data_to_plot[algo] = measurements[index_diff:]
            else:
                new_data_to_plot[algo] = measurements

        return new_data_to_plot

    def crop_data(self, algo_to_measurements):
        """Crops data from the end"""

        new_algo_to_measurements = {}

        # Get the index of the earliest flow observed for each algo
        for algo, measurements in algo_to_measurements.iteritems():
            # Get the index of the earliest change in load for any router
            crop_index = self._get_last_change_index(measurements)
            if crop_index < len(measurements):
                new_algo_to_measurements[algo] = measurements[:crop_index]
            else:
                new_algo_to_measurements[algo] = measurements
        return new_algo_to_measurements

    def crop_array(self, array):
        return [a for a in array if a < 0.1]

    def get_in_out_metric(self, measurement):
        tin = []
        tout = []
        for link_loads in measurement:
            ttin = sum([data['in'] for connections in link_loads[1].values() for other, data in connections.iteritems() if 'e' in other])
            ttout = sum([data['out'] for connections in link_loads[1].values() for other, data in connections.iteritems() if 'e' in other])
            tin.append(ttin)
            tout.append(ttout)

        # Divide it for the max
        #max_traffic = (LINK_BANDWIDTH*(self.k**2))/1e3
        tout = np.asarray(tout)#/(self.k**2)
        tin = np.asarray(tin)#/(self.k**2)
        return tin, tout

    def plot_in_out_abs_traffic(self):
        """
        Plots the total input traffic leaving the hosts compared to the total traffic
        entering the hosts

        :return:
        """
        # Load them
        self.algo_to_measurements = self.load_measurements()

        data_to_plot = {}
        fig = plt.figure(figsize=(80, 20))
        fig.suptitle("Input Vs Output traffic at the edge for different LB strategies", fontsize=20)

        for algo, measurements in self.algo_to_measurements.iteritems():
            (tin, tout) = self.get_in_out_metric(measurements)
            data_to_plot[algo] = (tin, tout)

        i = 1
        ncols = 1
        nrows = len(data_to_plot.keys())
        for algo, (tin, tout) in data_to_plot.iteritems():
            # Add subplot
            sub = fig.add_subplot(nrows, ncols, i)
            # Set the title
            sub.set_title("{0}".format(algo.upper()))
            sub.grid(True)

            # Plot in and out traffic
            sub.plot(tin, color='g', linestyle='-', label='Input traffic', linewidth=2.0)
            sub.plot(tout, color='r', linestyle='-', label='Output traffic', linewidth=2.0)

            # Set labels and limits
            sub.set_xlabel("Time (s)")
            sub.set_ylabel("Total Fat-Tree load")
            sub.set_ylim([0, 16.5])

            # Increment subplot index
            i = i + 1

        # Write legend and plot
        plt.legend(loc='best')
        # Set grid on
        plt.grid(True)
        plt.show()

    def plot_in_out_traffic(self):
        """
        Plots the total input traffic leaving the hosts compared to the total traffic
        entering the hosts

        :return:
        """
        # Load them
        self.algo_to_measurements = self.load_measurements()

        data_to_plot = {}

        fig = plt.figure(figsize=(80, 20))
        fig.suptitle("Output/Input traffic ratio at the edge for different LB strategies", fontsize=20)
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("OUT/IN ratio", fontsize=16)
        plt.ylim([0, 1.05])

        for algo, measurements in self.algo_to_measurements.iteritems():
            (tin, tout) = self.get_in_out_metric(measurements)
            data_to_plot[algo] = (tin, tout)

        for algo, (tin, tout) in data_to_plot.iteritems():
            # Compute ratio
            ratio = np.asarray([min(1, tout[index]/value) for index, value in enumerate(tin)])

            #Fetch colors
            color = algo_styles[algo]['color']
            linestyle = algo_styles[algo]['linestyle']

            # Plot in and out traffic
            plt.plot(ratio, color=color, linestyle=linestyle, label=algo, linewidth=2.0)

        # Write legend and plot
        plt.legend(loc='best')
        # Set grid on
        plt.grid(True)
        plt.show()

    def get_node_loads(self, node, measurements, upwards=True):
        """Fetches upward/downward loads for a specific node
        """

        # Store loads for each edge here
        edge_loads_over_time = {}

        neighborNodes = []

        isCore = self.topology.isCoreRouter(node)
        isEdge = self.topology.isEdgeRouter(node)
        isAggr = self.topology.isAggregationRouter(node)
        if isCore or isEdge:
            # Connected aggregation routers
            neighborNodes = [ar for ar, others in measurements[0][1].iteritems() if node in others.keys()]

        elif isAggr:
            # Connected core and edge routers
            neighborNodes = measurements[0][1][node].keys()

        # Populate results dict
        for neighbor in neighborNodes:
            edge_loads_over_time[neighbor] = []

        # Iterate measurements
        for (t, measurement) in measurements:
            # Fetch load per each neighbor
            for neighbor in neighborNodes:
                # Initialize load
                load = 0

                if isEdge or isCore:
                    if upwards == True:
                        load = measurement[neighbor][node]['in']
                    else:
                        load = measurement[neighbor][node]['out']

                elif isAggr:
                    if upwards == True:
                        if self.topology.isCoreRouter(neighbor):
                            load = measurement[node][neighbor]['out']
                        else:
                            load = measurement[node][neighbor]['in']
                    else:
                        if self.topology.isCoreRouter(neighbor):
                            load = measurement[node][neighbor]['in']
                        else:
                            load = measurement[node][neighbor]['out']

                # Add load
                edge_loads_over_time[neighbor].append(load)

        # Convert it to np array
        for neighbor in edge_loads_over_time.keys():
            loads = np.asarray(edge_loads_over_time[neighbor])
            edge_loads_over_time[neighbor] = loads

        return edge_loads_over_time

    def plot_node_traffic(self, node, upwards=True):
        """
        Plots the upwards/downwards traffic observed at node for each of its links.

        For instance:
          node=h_0_0, upwards=True: would print all traffic observed traffic in h_0_0 -> r_0_e0 only edge

          node=r_c
        """
        # Load topology
        try:
            self.loadTopo()
        except:
            print("Topology file not present!")
            return

        # Store result here
        data_to_plot = {}

        # Collect data per each algorithm
        for algo, measurements in self.algo_to_measurements.iteritems():
            edge_loads_over_time = self.get_node_loads(node, measurements, upwards)
            data_to_plot[algo] = edge_loads_over_time

        # Cound how many neighbors the node has
        node_neigbors = data_to_plot[data_to_plot.keys()[0]].keys()

        # Start the plotting
        fig = plt.figure(figsize=(80, 20))
        nrows = len(node_neigbors)
        ncols = 1

        # Set main title
        if upwards == True: direction = "Upwards"
        else: direction = "Downwards"
        fig.suptitle("{0} traffic at node {1}".format(direction, node), fontsize=20)

        # Iterate each neighbor
        i = 1
        for neighbor in node_neigbors:
            # Create a new subplot
            sub = fig.add_subplot(nrows, ncols, i)
            sub.grid(True)

            # Set title of subplot
            sub.set_title("Edge with {1}".format(node, neighbor))

            # Set axis titles -- only to lower graph
            if i == nrows:
                sub.set_xlabel("Time (s)", fontsize=16)
                sub.set_ylabel("Link load (percent)", fontsize=16)

            # Itereate algos
            for algo in data_to_plot.keys():
                color = algo_styles[algo]['color']
                linestyle = algo_styles[algo]['linestyle']
                sub.plot(data_to_plot[algo][neighbor], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

            # Set limit
            sub.set_ylim([0, 1])

            # Increment subplot index
            i += 1

        # Set the legend
        plt.legend(loc='best')
        # Set grid on
        plt.grid(True)

        plt.show()

    def _barplot(self, ax, dpoints, log=False):
        '''
        Create a barchart for data across different patterns with
        multiple algos for each category.

        @param ax: The plotting axes from matplotlib.
        @param dpoints: The data set as an (n, 3) numpy array
        '''
        # Aggregate the algos and the patterns according to their mean values
        algos = [(c, np.mean(dpoints[dpoints[:, 0] == c][:, 2].astype(float))) for c in np.unique(dpoints[:, 0])]
        patterns = [(c, np.mean(dpoints[dpoints[:, 1] == c][:, 2].astype(float))) for c in np.unique(dpoints[:, 1])]

        # Sort the algos, patterns and data so that the bars in
        # the plot will be ordered by category and condition
        algos = [c[0] for c in sorted(algos, key=o.itemgetter(1))]
        patterns = [c[0] for c in sorted(patterns, key=o.itemgetter(1))]

        # Extract the completion time values
        dpoints = np.array(sorted(dpoints, key=lambda x: patterns.index(x[1])))

        # Set the space between each set of bars
        space = 0.25
        n = len(algos)
        width = (1 - space) / (len(algos))

        # Create a set of bars at each position
        for i, algo in enumerate(algos):
            indeces = range(1, len(patterns) + 1)
            vals = dpoints[dpoints[:, 0] == algo][:, 2].astype(np.float)
            pos = [j - (1 - space) / 2. + i * width for j in indeces]
            if not log:
                ax.bar(pos, vals, width=width, label=algo, color=algo_styles[algo]['color'], alpha=1, zorder=40, edgecolor="none", linewidth=0)
            else:
                ax.bar(pos, vals, width=width, label=algo, color=algo_styles[algo]['color'], alpha=1, zorder=40, log=1,
                       edgecolor="none", linewidth=0)

        # Set the x-axis tick labels to be equal to the patterns
        ax.set_xticks(indeces)
        ax.set_xticklabels(patterns)

        plt.setp(plt.xticks()[1], rotation=0)

    def parse_experiment_data(self, experiment_folder):
        results = {}
        if not self.results_folder in experiment_folder:
            experiment_folder =  os.path.join(self.results_folder, experiment_folder)

        for root, dir, files in os.walk(experiment_folder):
            # We are in first level
            if root == experiment_folder:
                patterns = dir[:]
                results.update({pattern: {} for pattern in patterns})
            else:
                root_t = root.split('/')

                # We are in third level
                if len(root_t) == 4 and files:
                    pattern = root_t[-2]
                    algo = root_t[-1]
                    data = files[0]
                    inn, outt = self.read_in_out_traffic_file(os.path.join(root, data))
                    avg_out = np.asarray(outt)
                    results[pattern][algo] = avg_out.mean()
                else:
                    continue

        results = [{p: {a: v for a,v in data.iteritems()}} for p, data in results.iteritems()]
        return results

    def plot_average_bisection_bw(self, experiment_folder=''):
        if not experiment_folder:
            experiment_data = [{'Random': {'Elephant-DAG-Shifter-Best':90, 'ECMP': 60, 'Non-Blocking': 128}},
                               {'Staggered': {'Elephant-DAG-Shifter-Best': 100, 'ECMP': 80, 'Non-Blocking': 128}},
                               {'Stride4': {'Elephant-DAG-Shifter-Best': 110, 'ECMP': 65, 'Non-Blocking': 128}},
                               {'Stride8': {'Elephant-DAG-Shifter-Best': 128, 'ECMP': 70, 'Non-Blocking': 128}},
                               {'Bijection': {'Elephant-DAG-Shifter-Best': 110, 'ECMP': 80, 'Non-Blocking': 128}},
                               ]
        else:
            experiment_data = self.parse_experiment_data(experiment_folder)

        # Create matrix from experiment data dictionary: (algo, pattern, value)
        matrix = []
        for result_data in experiment_data:
            if result_data:
                # Extract pattern name
                pattern = result_data.keys()[0]

                # Get results of that pattern
                pattern_results = result_data.pop(pattern)

                # Iterate them
                for algo, value in pattern_results.iteritems():
                    # Append to matrix
                    matrix.append([algo, pattern, value])

        # Convert into np array
        matrix = np.asarray(matrix)

        # Start figure
        fig = plt.figure(figsize=(35, 10))

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
        ax.set_ylabel("Average bisection bandwidth (Mbps)", size='x-large', weight='bold')
        plt.tick_params(axis="both", which="both", bottom="on", top="off", labelbottom="on", left="off", right="off",
                        labelleft="on")
        plt.xticks(fontsize=14, weight="bold")
        plt.yticks(fontsize=14, weight="bold")

        # Generate bar plot
        self._barplot(ax, matrix, log=False)

        # Set fontsize and weight of axis ticks
        plt.xticks(fontsize=14, weight="bold")
        plt.yticks(fontsize=14, weight="bold")

        add_bottom_legend = True
        if add_bottom_legend:
            # Add a legend
            handles, labels = ax.get_legend_handles_labels()
            # Put a legend below current axis
            ax.legend(handles[:], labels[:], loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True,
                      fontsize='x-large', ncol=len(algo_styles))

        plt.grid(True)
        ax.grid(zorder=4)
        # plt.tight_layout()
        # plt.show()
        fig.subplots_adjust(left=0.08, right=0.97)
        filename = 'avg_bb'
        plt.savefig(experiment_folder + filename, format="pdf")
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Declare expected arguments
    parser.add_argument('-k', help='Fat Tree parameter', type=int, default=4)
    parser.add_argument('--file_list', nargs='+', help='List of measurement files to compare', type=str, required=False)

    parser.add_argument('--node', help="Plot traffic observed at node links only. e.g: h_0_0", type=str, default=None)
    parser.add_argument('--downwards', action="store_true", default=False)
    parser.add_argument('--upwards', action="store_true", default=True)

    parser.add_argument('--in_out', help="Plot ratio input/output traffic", action="store_true", default=False)
    parser.add_argument('--in_out_abs', help="Plot input traffic against output traffic", action="store_true", default=False)
    parser.add_argument('--avg_bb', help="Average bisection bandwidth (useful for TCP traffic)", action="store_true", default=False)
    parser.add_argument('--experiment', help='Experiment folder')
    parser.add_argument('--all', help="Plot all metrics", action="store_true", default=False)

    # Parse arguments
    args = parser.parse_args()

    # Start object and load measurement files
    ac = AlgorithmsComparator(k=args.k, file_list=args.file_list)

    if args.all:
        ac.plot_in_out_traffic()
        ac.plot_in_out_abs_traffic()
        ac.plot_average_bisection_bw()

    else:
        # Act according to presented arguments
        if args.node:
            if args.downwards:
                ac.plot_node_traffic(args.node, upwards=not(args.downwards))
            else:
                ac.plot_node_traffic(args.node, upwards=args.upwards)

        if args.in_out:
            ac.plot_in_out_traffic()

        if args.in_out_abs:
            ac.plot_in_out_abs_traffic()

        if args.avg_bb:
            if not args.experiment:
                print "Experiment folder needed!"
                experiment_folder = raw_input('*** Introduce experiment folder [experiment1]: ') or 'experiment1'

            else:
                experiment_folder = args.experiment

            ac.plot_average_bisection_bw(experiment_folder)
