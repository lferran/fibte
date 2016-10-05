from fibte.misc.dc_graph import DCGraph, DCDag
import numpy as np
import scipy.stats as stats
from fibte import  LINK_BANDWIDTH
import threading
import random
import logging
from fibte.logger import log
import copy
import time

# Decorator function for timing purposes
def time_func(function):
    def wrapper(*args,**kwargs):
        t = time.time()
        res = function(*args,**kwargs)
        log.debug("{0} took {1}s to execute".format(function.func_name, time.time()-t))
        return res
    return wrapper

class MiceEstimatorThread(threading.Thread):
    def __init__(self, sbmanager, orders_queue, results_queue,
                 mice_distributions, mice_distributions_lock,
                 capacities_graph, capacities_lock,
                 dags, samples=100, *args, **kwargs):
        # Superclass __init__()
        super(MiceEstimatorThread, self).__init__(*args, **kwargs)

        # Get params
        self.sbmanager = sbmanager           # Fibbign southbound manager instance
        self.mice_dbs = mice_distributions   # Mice flow average level distributions
        self.mice_dbs_lock = mice_distributions_lock
        self.caps_graph = capacities_graph   # Capacities left by the elephant flows
        self.caps_lock = capacities_lock
        self.dags = dags                     # Current Mice traffic dags
        self.prefixes = self.dags.keys()

        # Here we store the propagated mice loads
        self.propagated_mice_levels = self.caps_graph.copy()

        # Here we store link probabilites
        self.link_probs_graph = self.caps_graph.copy()
        for (u, v, data) in self.link_probs_graph.edges_iter(data=True):
            # Initially all links are chosen (assuming ECMP on all possible paths)
            for px in self.prefixes:
                data[px] = {'probability': 1, 'changed': False}

        # Number of samples to use
        self.samples = samples

        # Where to receive orders from
        self.orders_queue = orders_queue

        # Queue where to store the results
        self.results_queue = results_queue

        # Congestion probability threshold
        self.max_mice_cong_prob = 0.5

        # Link load from which we consider congestion
        self.congestion_threshold = 0.95

        # Set debug level
        log.setLevel(logging.DEBUG)

    def propagatePrefixNoise(self, prefix, i):
        """Propagates the sampled average load of the hosts towards prefix over
        the network graph"""
        # Fetch current prefix dag
        dag = self.dags[prefix]['dag']
        gw = self.dags[prefix]['gateway']

        # Iterate all other edge routers
        for er in self.propagated_mice_levels.edge_routers_iter():
            if er != gw:
                # Collect sum of loads from connected prefixes
                pxs = self.propagated_mice_levels.get_connected_destination_prefixes(er)

                # Take random samples for each source prefix connected to the edge router
                er_load = sum([random.choice(self.mice_dbs[prefix][px]['avg']) for px in pxs if prefix in self.mice_dbs and px in self.mice_dbs[prefix]])

                # Propagate it
                edges = self.propagate_sample(dag=dag, source=er, target=gw, load=er_load)

                # Iterate edges and sum pertinent load
                for (u, v, load) in edges:
                    self.propagated_mice_levels[u][v]['loads'][i] += load

    def propagateAllPrefixes(self, i):
        """Propagates all noise distributions"""
        for prefix in self.prefixes:
            self.propagatePrefixNoise(prefix, i)

    def takePropagationSamples(self):
        """Propagates all noise distributions over the dc_graph"""
        # Insert arrays to accumulate load samples
        for (a, b, data) in self.propagated_mice_levels.edges_iter(data=True):
            data['loads'] = np.zeros(self.samples)

        with self.mice_dbs_lock:
            # Take as many samples!
            for i in range(self.samples):
                self.propagateAllPrefixes(i)

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

    @time_func
    def totalCongestionProbability(self, caps_graph, threshold=0.3):
        """
        Propagate distribution samples and then, check for at least one
        congested link in all the network at every sample

        Assumes that noise level has been propagated in self.propagated_noise_level
        """
        # Compute maximum allowed bandwidth usage
        max_load = LINK_BANDWIDTH * threshold

        samples_with_congestion = 0

        for i in range(self.samples):
            for (u, v, data) in caps_graph.edges_iter(data=True):
                # Get the currently used capacity on the edge
                used_capacity = data.get('elephants_capacity', 0)
                mice_level = self.propagated_mice_levels[u][v]['loads'][i]

                # Check for congestion
                congestion = (mice_level + used_capacity) >= max_load
                if congestion:
                        samples_with_congestion += 1
                        break

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

    def choose_random_dag(self, prefix, link_probabilities):
        """Given the individual link probabilities, this
        generates a random DCDag

        Returns a brand new DCDag
        """
        dc_dag = self.link_probs_graph.get_random_dag_from_probability_dcgraph(prefix, link_probabilities)
        return dc_dag

    def get_link_probabilities(self, prefix):
        """
        """
        link_probs = {}
        for (u, v, data) in self.link_probs_graph.edges_iter(data=True):
            if prefix in data.keys():
                link_probs[(u,v)] = data[prefix]
            else:
                log.error("Something weird is happening...")
                import ipdb; ipdb.set_trace()

        return link_probs

    def set_probabilities_unchanged(self, prefix):
        """Iterate prob graph and set changed = False for specific prefix"""
        action = [data[prefix].update({'changed': False}) for (u, v, data) in self.link_probs_graph.edges_iter(data=True)]

    def get_prefixes_using_link(self, src, dst):
        """"""
        srctype = self.link_probs_graph.get_router_type(src)
        dsttype = self.link_probs_graph.get_router_type(dst)
        link_direction = self.link_probs_graph.get_edge_data(src, dst)['direction']

        if link_direction == 'uplink':
            if srctype == 'edge' and dsttype == 'aggregation':
                # Need to modify all DAGS except those for destinations connected to u (edge router)
                destinations = [px for px in self.prefixes if self.link_probs_graph.get_destination_prefix_gateway(px) != src]

            elif srctype == 'aggregation' and dsttype == 'core':
                # Need to modify all DAGS except those for destinations connected in pod of aggregation router
                aggr_pod = self.link_probs_graph.get_router_pod(src)

                # Filter out destinations that are not in that pods
                destinations = [px for px in self.prefixes if self.link_probs_graph.get_destination_prefix_pod(px) != aggr_pod]

            else:
                log.error("Wrong data")
                import ipdb;ipdb.set_trace()
                raise ValueError("wrong link")
        else:
            if srctype == 'aggregation' and dsttype == 'edge':
                # Need to modify only dags towards destinations connected to edge router
                destinations = [px for px in self.prefixes if self.link_probs_graph.get_destination_prefix_gateway(px) == dst]

            elif srctype == 'core' and dsttype == 'aggregation':
                # Need to modify only dags to destinations under pod connected to aggregation
                aggr_pod = self.link_probs_graph.get_router_pod(dst)

                # Filter destinations in that pod
                destinations = [px for px in self.prefixes if self.link_probs_graph.get_destination_prefix_pod(px) == aggr_pod]

            else:
                log.error("Wrong data")
                import ipdb;ipdb.set_trace()
                raise ValueError("wrong link")

        return destinations

    def modify_link_probabilities(self, src, dst):
        """Modifies the probabilities to be chosen for the corresponding
        predecessor links, given a congested link (src, dst)
        """
        # Get destination prefixes that use this link
        prefixes_to_modify = self.get_prefixes_using_link(src, dst)

        srctype = self.link_probs_graph.get_router_type(src)
        dsttype = self.link_probs_graph.get_router_type(dst)
        link_direction = self.link_probs_graph.get_edge_data(src, dst)['direction']

        if link_direction == 'uplink':
            if srctype == 'edge' and dsttype == 'aggregation':
                # Need to reduce probability of that edge to be chosen

                # Iterate prefixes to modify
                for px in prefixes_to_modify:
                    # Number of possible outgoing edges
                    n_succ = len(self.link_probs_graph.successors(src))

                    to_reduce = 1.0 / n_succ
                    self.link_probs_graph[src][dst][px]['probability'] -= to_reduce
                    self.link_probs_graph[src][dst][px]['changed'] = True

            elif srctype == 'aggregation' and dsttype == 'core':
                # Need to reduce the probability of that link being chosen and the predecessor edges too,
                # so that not so much traffic is sent through it, and other cores are chosen from that pod

                n_succ = len(self.link_probs_graph.successors(src))
                to_reduce_that_link =  1.0 / n_succ

                # Get source pod
                src_pod = self.link_probs_graph.get_router_pod(src)

                # Compute links to reduce probability
                links_to_reduce = [(er, src) for er in self.link_probs_graph.edge_routers() if self.link_probs_graph.get_router_pod(er) == src_pod]

                # Iterate prefixes to modify
                for px in prefixes_to_modify:
                    # Reduce that link itself first
                    self.link_probs_graph[src][dst][px]['probability'] -= to_reduce_that_link
                    self.link_probs_graph[src][dst][px]['changed'] = True

                    # Then others in source pod
                    for (u,v) in links_to_reduce:
                        self.link_probs_graph[u][v][px]['probability'] -= to_reduce_that_link
                        self.link_probs_graph[u][v][px]['changed'] = True

            else:
                log.error("Wrong data")
                import ipdb; ipdb.set_trace()

        else:
            if srctype == 'aggregation' and dsttype == 'edge':
                # Need to reduce probability of other edge routers to same aggregation router,
                # and from aggregation routers in other pods to the same core routers connected to src

                # Get pod of edge router
                dst_pod = self.link_probs_graph.get_router_pod(dst)

                # Then get all other links from edge routers in same pod to that source aggregation router
                links_in_pod = [(er, src) for er in self.link_probs_graph.edge_routers()
                                      if self.link_probs_graph.get_router_pod(er) == dst_pod and er != dst]

                # Get connected core routers
                connected_cores = [d for d in self.link_probs_graph.predecessors(src) if self.link_probs_graph.is_core(d)]

                # Get all other links from aggregation routers outside the pod connected the cores above
                links_outside_pod = [(aggr, cr) for cr in connected_cores for aggr in self.link_probs_graph.predecessors(cr)
                                     if self.link_probs_graph.get_router_pod(aggr) != dst_pod]

                # Iterate prefixes to modify
                for px in prefixes_to_modify:

                    to_reduce = 1.0 / len(prefixes_to_modify)
                    for (u, v) in links_in_pod:
                        self.link_probs_graph[u][v][px]['probability'] -= to_reduce
                        self.link_probs_graph[u][v][px]['changed'] = True

                    to_reduce = 1.0 / len(prefixes_to_modify)
                    for (u, v) in links_outside_pod:
                        self.link_probs_graph[u][v][px]['probability'] -= to_reduce
                        self.link_probs_graph[u][v][px]['changed'] = True

            elif srctype == 'core' and dsttype == 'aggregation':
                # Need to reduce the probability of links from other pods to that core route
                # Get pod of edge router
                dst_pod = self.link_probs_graph.get_router_pod(dst)

                # Get all other links from aggregation routers outside the pod connected the cores above
                links_from_other_pods = [(aggr, src) for aggr in self.link_probs_graph.predecessors(src)
                                     if self.link_probs_graph.get_router_pod(aggr) != dst_pod]

                # Iterate prefixes to modify
                for px in prefixes_to_modify:
                    to_reduce = 1.0 / len(prefixes_to_modify)
                    for (u, v) in links_from_other_pods:
                        self.link_probs_graph[u][v][px]['probability'] -= to_reduce
                        self.link_probs_graph[u][v][px]['changed'] = True

            else:
                log.error("Wrong data")
                import ipdb;ipdb.set_trace()

    def link_probabilities_changed(self, link_probs):
        """Given likn probabilities as a dict of dicts of dicts:
        {src -> {dst -> {'probability: 1, 'changed': False}}}
        Returns true iif there is at least one link that changed == True
        """
        return any([data['changed'] == True for ((u, v), data) in link_probs.iteritems()])

    @time_func
    def modify_dags(self, caps_graph):
        """
        Adapts mice traffic to capacities left by elephant flows
        :return:
        """
        # Accumulate here the links with higher congestion probability
        congested_links = []
        for (u, v, data) in self.propagated_mice_levels.edges_iter(data=True):
            elephant_capacity = caps_graph[u][v].get('elephants_capacity', 0)
            loads = data['loads'] + elephant_capacity
            linkCongProb = self.linkCongestionProbability(loads, threshold=self.congestion_threshold)
            if linkCongProb >= self.max_mice_cong_prob:
               congested_links.append((u, v))

        # Modify link probabilities in the dags for each link
        for (u, v) in congested_links:
            self.modify_link_probabilities(u,v)

        # Choose random dangs for each prefix
        new_dags = {}
        for prefix in self.prefixes:
            link_probabilities = self.get_link_probabilities(prefix)
            if self.link_probabilities_changed(link_probabilities):
                # Generate new random dag
                new_random_dag = self.choose_random_dag(prefix, link_probabilities)
                new_dags[prefix] = new_random_dag
                self.dags[prefix]['dag'] = new_random_dag

                # Reset changed = False
                self.set_probabilities_unchanged(prefix)

        # Apply new dags all at same time
        self.sbmanager.add_dag_requirements_from(new_dags)

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

            elif order_type == 'propagate_new_distributions':
                # Create copy of dc_graph first
                with self.caps_lock:
                    caps_graph = self.caps_graph.copy()

                # Take new samples based on new distributions
                self.takePropagationSamples()

                # Compute congestion probability on propagated mice levels
                congProb = self.totalCongestionProbability(caps_graph, threshold=self.congestion_threshold)

                log.info("Mice congestion probability: {0} \t --threshold = {1}".format(congProb, self.max_mice_cong_prob))

                # If congestion is over the threshold
                if congProb >= self.max_mice_cong_prob:
                    log.info("Modifying mice dags...")
                    self.modify_dags(caps_graph)

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
