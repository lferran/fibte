import json
import os
from fibte.misc.topology_graph import TopologyGraph
import numpy as np
from cycler import cycler
from fibte import CFG, LINK_BANDWIDTH

import matplotlib.pyplot as plt

tmp_files = CFG.get("DEFAULT", "tmp_files")
db_topo = CFG.get("DEFAULT", "db_topo")

algo_styles = {'ecmp':
                   {'color': 'red', 'linestyle':'-'},

               'random-dags':
                   {'color':'orange', 'linestyle':'-'},

               'best-ranked-core':
                   {'color':'green', 'linestyle':'-'},

               'random-core':
                   {'color':'blue', 'linestyle':'-'},
               }

class AlgorithmsComparator(object):
    def __init__(self, k=4, file_list=[]):
        # FatTree parameter
        self.k = k

        # List of files for different algorithm measurements
        self.file_list = file_list

        self.topology = TopologyGraph(getIfindexes=True, db=os.path.join(tmp_files, db_topo))

        # Load them
        self.algo_to_measurements = self.load_measurements()

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
            algo = filename.strip('txt').strip('.').split('_')[-1]
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

    def plot_core_input_traffic(self):
        """
        Plots the input traffic for all core routers along time.
        :return:
        """
        # Number of cores
        n_cores = (self.k**2)/4

        # Create the figure first
        fig = plt.figure(figsize=(80, 20))
        #fig.subplots_adjust(bottom=0.025, left=0.025, top=0.975, right=0.975)
        fig.suptitle("Core input traffic", fontsize=20)

        # Start the subplot posistions
        subplot_positions = []
        for i in range(1, n_cores+1):
            subplot_positions.append((4, 1, i))

        coreRouters = self.topology.getCoreRouters()
        core_positions = self._get_core_positions()
        aggrRouters = self.topology.getAggregationRouters()

        # Accumulate data to plot here
        data_to_plot = {}

        # Iterate all core routers and collect data
        for cr in coreRouters:

            # Create inner dict
            data_to_plot[cr] = {}

            # Collect all data to plot -- sum all input traffic from all aggr ---
            for algo, measurements in self.algo_to_measurements.iteritems():
                # Loads over time
                loads_to_cr = []

                # Set time relatively to starting time
                initial_time = measurements[0][0]

                # Extract data from measurements
                for (t, link_loads) in measurements:
                    # Load to cr in that time t
                    total_load_to_cr = 0
                    for ar in aggrRouters:
                        if link_loads[ar].has_key(cr):
                            total_load_to_cr += link_loads[ar][cr]['out']

                    # Append load at that time
                    loads_to_cr.append((t - initial_time, total_load_to_cr))

                # Convert it to np array
                loads_arr = np.asarray([l for t, l in loads_to_cr])/(self.k)

                # Put it into dict
                data_to_plot[cr][algo] = loads_arr

        algos = data_to_plot[data_to_plot.keys()[0]].keys()

        for nrows, ncols, plot_number in subplot_positions:
            sub = fig.add_subplot(nrows, ncols, plot_number)
            sub.grid(True)
            sub.set_ylim([0,1])
            #if plot_number != 4:
            #    sub.set_xticks([])

            # Get core router with that position
            cr = [router for router, position in core_positions.iteritems() if position[2] == plot_number][0]
            #print "Current core router: {0}".format(cr)

            # Plot it
            for algo in algos:
                # Fetch load
                loads_to_cr = data_to_plot[cr][algo]
                sub.set_title("{0}".format(cr))
                sub.plot(loads_to_cr, label=algo, c=algo_styles[algo]['color'], ls=algo_styles[algo]['linestyle'], linewidth=2.0)

        # Locate the legend
        plt.legend(loc='best')

        # Set grid on
        plt.grid(True)

        #Show plot
        plt.show()

    def get_in_out_metric(self, measurement):
        coreRouters = self.topology.getCoreRouters()
        edgeRouters = self.topology.getEdgeRouters()
        aggrRouters = self.topology.getAggregationRouters()

        tin = []
        tout = []
        for link_loads in measurement:
            ttin = sum([data['in'] for connections in link_loads[1].values() for other, data in connections.iteritems() if other in edgeRouters])
            ttout = sum([data['out'] for connections in link_loads[1].values() for other, data in connections.iteritems() if other in edgeRouters])
            tin.append(ttin)
            tout.append(ttout)

        # Divide it for the max
        #max_traffic = (LINK_BANDWIDTH*(self.k**2))/1e3
        tout = np.asarray(tout)/(self.k**2)
        tin = np.asarray(tin)/(self.k**2)
        return tin, tout

    def plot_in_out_abs_traffic(self):
        """
        Plots the total input traffic leaving the hosts compared to the total traffic
        entering the hosts

        :return:
        """
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
            sub.set_ylim([0, 1])

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
        data_to_plot = {}

        fig = plt.figure(figsize=(80, 20))
        fig.suptitle("Output/Input traffic ratio at the edge for different LB strategies", fontsize=20)
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("OUT/IN ratio", fontsize=16)
        plt.ylim([0, 1])

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

    def get_average_core_load(self, measurements):
        coreRouters = self.topology.getCoreRouters()

        upwards = []
        downwards = []

        for t, link_loads in measurements:
            # Fetch upward and downward load values for that time
            core_upwards_loads = [link_loads[a][o]['out'] for a, other in link_loads.iteritems() for o in other if o in coreRouters]
            core_downwards_loads = [link_loads[a][o]['in'] for a, other in link_loads.iteritems() for o in other if o in coreRouters]

            # Convert it to an array
            cul = np.asarray(core_upwards_loads)
            cdl = np.asarray(core_downwards_loads)

            # Append it to lists
            upwards.append(cul)
            downwards.append(cdl)

        # Convert it to average and std
        upwards_avg = [u.mean() for u in upwards]
        upwards_std = [u.std() for u in upwards]
        downwards_avg = [u.mean() for u in downwards]
        downwards_std = [u.std() for u in downwards]

        return ({'avg': upwards_avg, 'std': upwards_std}, {'avg': downwards_avg, 'std': downwards_std})

    def plot_average_core_loads(self):
        """
        Plots the average core load for upwards and downwards traffic
        :return:
        """
        # Fetch data to plot
        data_to_plot = {}
        for algo, measurements in self.algo_to_measurements.iteritems():
            (upwards, downwards) = self.get_average_core_load(measurements)
            data_to_plot[algo] = {'up': upwards, 'down': downwards}

        #import ipdb; ipdb.set_trace()

        subplot_positions = [(2,2,1), (2,2,2), (2,2,3), (2,2,4)]

        # Create figure
        fig = plt.figure(figsize=(80, 20))
        fig.suptitle("Average core layer load for different LB strategies", fontsize=20)

        # Iterate subplots
        for (row, col, index) in subplot_positions:

            # Create a new subplot
            sub = fig.add_subplot(row, col, index)
            sub.grid(True)

            # Upper left corner
            if index == 1:
                sub.set_title("Upward traffic average load", fontsize=16)
                for algo in data_to_plot.keys():
                    #import ipdb; ipdb.set_trace()
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['up']['avg'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Average load", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

            elif index == 2:
                sub.set_title("Downward traffic average load", fontsize=16)
                for algo in data_to_plot.keys():
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['down']['avg'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Average load", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

            elif index == 3:
                sub.set_title("Upward traffic Std.", fontsize=16)
                for algo in data_to_plot.keys():
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['up']['std'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Load Std.", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

            elif index == 4:
                sub.set_title("Downward traffic Std.", fontsize=16)
                for algo in data_to_plot.keys():
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['down']['std'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Load Std.", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

        # Write legend and plot
        plt.legend(loc='best')
        # Set grid on
        plt.grid(True)
        plt.show()

    def get_max_core_load_diff(self, measurements):
        coreRouters = self.topology.getCoreRouters()

        upwards_diff = []
        downwards_diff = []

        for t, link_loads in measurements:
            # Fetch upward and downward load values for that time
            core_upwards_loads = [link_loads[a][o]['out'] for a, other in link_loads.iteritems() for o in other if o in coreRouters]
            core_downwards_loads = [link_loads[a][o]['in'] for a, other in link_loads.iteritems() for o in other if o in coreRouters]

            # Convert it to an array
            cul = np.asarray(core_upwards_loads)
            cdl = np.asarray(core_downwards_loads)

            # Append it to lists
            upwards_diff.append(max(cul) - min(cul))
            downwards_diff.append(max(cdl) - min(cdl))

        # Convert it to array
        upwards_diff = np.asarray(upwards_diff)
        downwards_diff = np.asarray(downwards_diff)

        return (upwards_diff, downwards_diff)

    def plot_max_core_load_diff(self):
        # Fetch data to plot
        data_to_plot = {}
        for algo, measurements in self.algo_to_measurements.iteritems():
            (upwards, downwards) = self.get_max_core_load_diff(measurements)
            data_to_plot[algo] = {'up': upwards, 'down': downwards}

        # import ipdb; ipdb.set_trace()
        subplot_positions = [(2, 1, 1), (2, 1, 2)]

        # Create figure
        fig = plt.figure(figsize=(80, 20))
        fig.suptitle("Difference between most and least loaded core router", fontsize=20)

        # Iterate subplots
        for (row, col, index) in subplot_positions:

            # Create a new subplot
            sub = fig.add_subplot(row, col, index)
            sub.grid(True)

            # Upper left corner
            if index == 1:
                sub.set_title("Upward traffic", fontsize=16)
                for algo in data_to_plot.keys():
                    # import ipdb; ipdb.set_trace()
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['up'], color=color, linestyle=linestyle, label=algo,
                             linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Load difference", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

            elif index == 2:
                sub.set_title("Downward traffic", fontsize=16)
                for algo in data_to_plot.keys():
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['down'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Load difference", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

        # Write legend and plot
        plt.legend(loc='best')
        # Set grid on
        plt.grid(True)
        plt.show()

    def get_average_core_diff(self, measurements):
        coreRouters = self.topology.getCoreRouters()

        upwards_diff = []
        downwards_diff = []

        for t, link_loads in measurements:
            # Fetch upward and downward load values for that time
            core_upwards_loads = [link_loads[a][o]['out'] for a, other in link_loads.iteritems() for o in other if o in coreRouters]
            core_downwards_loads = [link_loads[a][o]['in'] for a, other in link_loads.iteritems() for o in other if o in coreRouters]

            # Convert it to an array
            cul = np.asarray(core_upwards_loads)
            cdl = np.asarray(core_downwards_loads)

            # Compute average difference
            cul_diff = []
            cdl_diff = []

            for index, cu in enumerate(cul):
                for index2, cu2 in enumerate(cul):
                    if index != index2:
                        cul_diff.append(abs(cu - cu2))

            for j, cd in enumerate(cdl):
                for j2, cd2 in enumerate(cdl):
                    if j != j2:
                        cdl_diff.append(abs(cd - cd2))

            # Convert to array
            cul_diff = np.asarray(cul_diff)
            cdl_diff = np.asarray(cdl_diff)

            # Append if
            upwards_diff.append(cul_diff)
            downwards_diff.append(cdl_diff)

        # Exctract mean and std
        upwards_avg = [u.mean() for u in upwards_diff]
        upwards_std = [u.std() for u in upwards_diff]
        downwards_avg = [u.mean() for u in downwards_diff]
        downwards_std = [u.std() for u in downwards_diff]

        # Convert it to array
        upwards_diff = np.asarray(upwards_diff)
        downwards_diff = np.asarray(downwards_diff)

        return ({'avg': upwards_avg, 'std': upwards_std}, {'avg': downwards_avg, 'std': downwards_std})

    def plot_average_core_diff(self):
        """
        Plots the average core load for upwards and downwards traffic
        :return:
        """
        # Fetch data to plot
        data_to_plot = {}
        for algo, measurements in self.algo_to_measurements.iteritems():
            (upwards, downwards) = self.get_average_core_diff(measurements)
            data_to_plot[algo] = {'up': upwards, 'down': downwards}

        subplot_positions = [(2,2,1), (2,2,2), (2,2,3), (2,2,4)]

        # Create figure
        fig = plt.figure(figsize=(80, 20))
        fig.suptitle("Average difference in load for core layer routers", fontsize=20)

        # Iterate subplots
        for (row, col, index) in subplot_positions:

            # Create a new subplot
            sub = fig.add_subplot(row, col, index)
            sub.grid(True)

            # Upper left corner
            if index == 1:
                sub.set_title("Upward traffic average load difference", fontsize=16)
                for algo in data_to_plot.keys():
                    #import ipdb; ipdb.set_trace()
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['up']['avg'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Average load difference", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

            elif index == 2:
                sub.set_title("Downward traffic average load difference", fontsize=16)
                for algo in data_to_plot.keys():
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['down']['avg'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Average load difference", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

            elif index == 3:
                sub.set_title("Upward traffic load difference Std.", fontsize=16)
                for algo in data_to_plot.keys():
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['up']['std'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Load diff Std.", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

            elif index == 4:
                sub.set_title("Downward traffic load difference Std.", fontsize=16)
                for algo in data_to_plot.keys():
                    color = algo_styles[algo]['color']
                    linestyle = algo_styles[algo]['linestyle']
                    # plot
                    sub.plot(data_to_plot[algo]['down']['std'], color=color, linestyle=linestyle, label=algo, linewidth=2.0)

                    # set axis labels
                    sub.set_ylabel("Load diff Std.", fontsize=16)
                    sub.set_xlabel("Time (s)", fontsize=16)

                    # set limit
                    sub.set_ylim([0, 1])

        # Write legend and plot
        plt.legend(loc='best')
        # Set grid on
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Declare expected arguments
    parser.add_argument('-k', help='Fat Tree parameter', type=int, default=4)
    parser.add_argument('--file_list', nargs='+', help='List of measurement files to compare', type=str, required=True)

    parser.add_argument('--node', help="Plot traffic observed at node links only. e.g: h_0_0", type=str, default=None)
    parser.add_argument('--downwards', action="store_true", default=False)
    parser.add_argument('--upwards', action="store_true", default=True)

    parser.add_argument('--in_out', help="Plot ratio input/output traffic", action="store_true", default=False)
    parser.add_argument('--in_out_abs', help="Plot input traffic against output traffic", action="store_true", default=False)

    parser.add_argument('--core_input_traffic', help="Plot traffic arriving at core layer", action="store_true", default=False)

    parser.add_argument('--average_core_load', help="Plot average core layer load", action="store_true", default=False)
    parser.add_argument('--max_diff_core_load', help="Plot maximum difference between core loads", action="store_true", default=False)
    parser.add_argument('--average_diff_core_load', help="Plot average difference between core loads", action="store_true", default=False)

    parser.add_argument('--all', help="Plot all metrics", action="store_true", default=False)

    # Parse arguments
    args = parser.parse_args()

    # Start object and load measurement files
    ac = AlgorithmsComparator(k=args.k, file_list=args.file_list)

    if args.all:
        ac.plot_in_out_traffic()
        ac.plot_in_out_abs_traffic()
        ac.plot_core_input_traffic()
        ac.plot_average_core_loads()
        ac.plot_max_core_load_diff()
        ac.plot_average_core_diff()

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

        if args.core_input_traffic:
            ac.plot_core_input_traffic()

        if args.average_core_load:
            ac.plot_average_core_loads()

        if args.max_diff_core_load:
            ac.plot_max_core_load_diff()

        if args.average_diff_core_load:
            ac.plot_average_core_diff()