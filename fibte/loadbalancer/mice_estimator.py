from fibte.misc.dc_graph import DCGraph, DCDag
import numpy as np
import scipy.stats as stats
from fibte import  LINK_BANDWIDTH
import threading
import random
import logging
from fibte.logger import log

class MiceEstimatorThread(threading.Thread):
    def __init__(self, orders_queue, results_queue,
                 mice_distributions, mice_distributions_lock,
                 capacities_graph, capacities_lock,
                 dags, dags_lock, samples=100, *args, **kwargs):

        super(MiceEstimatorThread, self).__init__(*args, **kwargs)

        # Get params
        self.mice_dbs_lock = mice_distributions_lock
        self.mice_dbs = mice_distributions
        self.caps_graph = capacities_graph
        self.caps_lock = capacities_lock
        self.dags = dags
        self.prefixes = self.dags.keys()
        self.dags_lock = dags_lock

        # Number of samples to use
        self.samples = samples

        # Where to receive orders from
        self.orders_queue = orders_queue

        # Queue where to store the results
        self.results_queue = results_queue

        # Set debug level
        log.setLevel(logging.DEBUG)

    def propagatePrefixNoise(self, prefix, dc_graph, i):
        """Propagates the sampled average load of the hosts towards prefix over
        the network graph"""

        # Fetch current prefix dag
        with self.dags_lock:
            dag = self.dags[prefix]['dag']
            gw = self.dags[prefix]['gateway']

        # Iterate all other edge routers
        for er in dc_graph.edge_routers_iter():
            if er != gw:
                # Collect sum of loads from connected prefixes
                pxs = dc_graph.get_connected_destination_prefixes(er)

                # Take random samples for each source prefix connected to the edge router
                er_load = sum([random.choice(self.mice_dbs[prefix][px]['avg']) for px in pxs if prefix in self.mice_dbs and px in self.mice_dbs[prefix]])

                # Propagate it
                edges = self.propagate_sample(dag=dag, source=er, target=gw, load=er_load)

                # Iterate edges and sum pertinent load
                for (u, v, load) in edges:
                    dc_graph[u][v]['loads'][i] += load

        # Return graph
        return dc_graph

    def propagateAllPrefixes(self, dcg, i):
        """Propagates all noise distributions"""
        for prefix in self.prefixes:
            dcg = self.propagatePrefixNoise(prefix, dcg, i)
        return dcg

    def takePropagationSamples(self, dcg):
        """Propagates all noise distributions over the dc_graph"""

        # Insert arrays to accumulate load samples
        for (a, b, data) in dcg.edges_iter(data=True):
            data['loads'] = np.zeros(self.samples)

        with self.mice_dbs_lock:
            # Take as many samples!
            for i in range(self.samples):
                dcg = self.propagateAllPrefixes(dcg, i)

        return dcg

    def _totalCongestionProbability(self, threshold=0.3):
        """
        This way of computing the congestion probability is wrong, because the
        individual link's congestion probability are dependant on each other.
        """
        # Create copy of dc_graph first
        with self.caps_lock:
            caps_graph = self.caps_graph.copy()

        # Propagate current mice distributions
        caps_graph = self.takePropagationSamples(caps_graph)

        # Accumulate congestion probabilites
        cps = []
        for (u, v, data) in caps_graph.edges_iter(data=True):
            if data.has_key('loads'):
                loads = data['loads']
                cp = self.linkCongestionProbability(loads, threshold)
                cps.append(cp)
            else:
                import ipdb; ipdb.set_trace()

        return self.union_congestion_probability(cps)

    def totalCongestionProbability(self, threshold=0.3):
        """
        Propagate distribution samples and then, check for at least one
        congested link in all the network at every sample
        """
        # Compute maximum allowed bandwidth usage
        max_load = LINK_BANDWIDTH * threshold

        # Create copy of dc_graph first
        with self.caps_lock:
            caps_graph = self.caps_graph.copy()

        # Propagate current mice distributions
        caps_graph = self.takePropagationSamples(caps_graph)

        samples_with_congestion = 0
        for i in range(self.samples):
            for (u, v, data) in caps_graph.edges_iter(data=True):
                # Get the currently used capacity on the edge
                used_capacity = data.get('elephants_capacity', 0)

                if data.has_key('loads'):
                    # Exctract ith sample
                    load = data['loads'][i]

                    # Check for congestion
                    congestion = (load + used_capacity) >= max_load
                    if congestion:
                        samples_with_congestion += 1
                        break
                else:
                    import ipdb; ipdb.set_trace()

        # Return ratio of congested samples
        congestion_probability = samples_with_congestion/float(self.samples)

        return congestion_probability

    def linkCongestionProbability(self, loads, threshold):
        """Computes the probability of the samples for a given link to be
        over the specified threshold.
        Returns the ratio of load samples over the threshold
        """
        total_samples = float(len(loads))
        congested_samples = len([l for l in loads if l > LINK_BANDWIDTH*threshold])
        return congested_samples/total_samples

    @staticmethod
    def union_congestion_probability(cps):
        """
        Apply function found in: http://lethalman.blogspot.ch/2011/08/probability-of-union-of-independent.html

        IMPORTANT ASSUMPTION: We assume here that the individual link congestion
                              probability are independent from each other...
                              QUESTION: IS THAT RIGHT?
        """
        tmp = reduce(lambda x,y: x*y, map(lambda x: 1-x, cps))
        final_cp = 1 - tmp
        return final_cp

    @staticmethod
    def get_edges_from_path(path):
        return zip(path[:-1], path[1:])

    @staticmethod
    def propagate_sample(dag, source, target, load):
        if source == target:
            return []
        else:
            # Get successors from DAG
            successors = dag.successors(source)

            # Compute new load
            new_load = load / float(len(successors))
            edges = []
            for succ in successors:
                edges += [(source, succ, new_load)] + MiceEstimatorThread.propagate_sample(dag, succ, target, new_load)
            return edges

    def run(self):
        while True:
            try:
                order = self.orders_queue.get(block=True)
            except:
                exit(0)

            # Checked received order
            order_type = order['type']
            if order_type == 'compute_congestion_probability':
                tcp = self.totalCongestionProbability(threshold=order['threshold'])
                self.results_queue.put(tcp)

            elif order_type == 'terminate':
                log.info("Self-shutting down...")
                break

            else:
                log.info(order)
                continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--threshold',
                        help='Congestion threshold (as a percentage of link bandwidth)',
                        type=float,
                        default=0.5)

    parser.add_argument('-k', '--k', help='Fat-Tree parameter', type=int, default=4)
    args = parser.parse_args()

    # Both DCGraph and DCDags will be given by the LBController
    dc_graph = DCGraph(k=args.k)

    # For the moment, the DCDags will be generated here, although in practice,
    # they need to be transmitted from the LBController too
    me = MiceEstimator(k=args.k, dc_graph=dc_graph, dst_dags=None)
    cp = me.totalCongestionProbability(threshold=args.threshold)
    import ipdb; ipdb.set_trace()
