from fibte.misc.dc_graph import DCGraph, DCDag
import numpy as np
from fibte import  LINK_BANDWIDTH
import threading
import random
import logging
from fibte.logger import log
import copy
import time
import json

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
                 dags, q_server, samples=10, *args, **kwargs):
        # Superclass __init__()
        super(MiceEstimatorThread, self).__init__(*args, **kwargs)

        # Get params
        self.sbmanager = sbmanager           # Fibbign southbound manager instance
        self.mice_dbs = mice_distributions   # Mice flow average level distributions
        self.mice_dbs_lock = mice_distributions_lock
        self.caps_graph = capacities_graph   # Capacities left by the elephant flows
        self.caps_lock = capacities_lock
        self.dags = dags                     # Current Mice traffic dags
        self.initial_dags = copy.deepcopy(self.dags) # Keep copy of initial dags
        self.prefixes = self.initial_dags.keys()
        self.q_server = q_server

        # Get k
        self.k = self.caps_graph.k

        # Here we store the propagated mice loads
        self.propagated_mice_levels = self.caps_graph.copy()

        # Number of samples to use
        self.samples = samples

        # Where to receive orders from
        self.orders_queue = orders_queue

        # Queue where to store the results
        self.results_queue = results_queue

        # Create link probabilities graph
        self.link_probs_graph = self._start_link_probabilities_graph()

        # Congestion probability threshold
        self.max_mice_cong_prob = 0.5

        # Link load from which we consider congestion
        self.congestion_threshold = 0.95

        self.load_router_names()

        # Set debug level
        #log.setLevel(logging.DEBUG)

    @staticmethod
    def get_links_from_path(path):
        return zip(path[:-1], path[1:])

    def _sendMainThreadToSleep(self, seconds):
        """Makes the main thread jump to the sleep mode
        **ONLY FOR DEBUGGING PURPOSES!
        """
        self.q_server.put(json.dumps({'type': 'sleep', 'data': seconds}))

    def load_router_names(self):
        """FOR DEBUGGING PURPOSES ONLY"""
        self.r_0_e0 = self.caps_graph.get_router_from_position('edge', 0,0)
        self.r_0_e1 = self.caps_graph.get_router_from_position('edge', 1,0)

        self.r_1_e0 = self.caps_graph.get_router_from_position('edge', 0,1)
        self.r_1_e1 = self.caps_graph.get_router_from_position('edge', 1,1)

        self.r_2_e0 = self.caps_graph.get_router_from_position('edge', 0,2)
        self.r_2_e1 = self.caps_graph.get_router_from_position('edge', 1,2)

        self.r_3_e0 = self.caps_graph.get_router_from_position('edge', 0,3)
        self.r_3_e1 = self.caps_graph.get_router_from_position('edge', 1,3)

        self.r_0_a0 = self.caps_graph.get_router_from_position('aggregation', 0, 0)
        self.r_0_a1 = self.caps_graph.get_router_from_position('aggregation', 1, 0)

        self.r_1_a0 = self.caps_graph.get_router_from_position('aggregation', 0, 1)
        self.r_1_a1 = self.caps_graph.get_router_from_position('aggregation', 1, 1)

        self.r_2_a0 = self.caps_graph.get_router_from_position('aggregation', 0, 2)
        self.r_2_a1 = self.caps_graph.get_router_from_position('aggregation', 1, 2)

        self.r_3_a0 = self.caps_graph.get_router_from_position('aggregation', 0, 3)
        self.r_3_a1 = self.caps_graph.get_router_from_position('aggregation', 1, 3)

        self.r_c0 = self.caps_graph.get_router_from_position('core', 0)
        self.r_c1 = self.caps_graph.get_router_from_position('core', 1)
        self.r_c2 = self.caps_graph.get_router_from_position('core', 2)
        self.r_c3 = self.caps_graph.get_router_from_position('core', 3)

    def print_stuff(self, stuff):
        """
        ***DEBUGGING PURPOSES ONLY
        """
        return self.caps_graph.print_stuff(stuff)

    def get_dependant_probability(self, link, dst_prefix):
        """Given a link, returns its dependant probability:
        the one given by its successor edges towards destination
        prefix"""
        slinks = self.get_successor_links_on_direction(link, dst_prefix)

        # No dependencies (aggr -> edge link)
        if not slinks:
            return 1

        else:
            # Compute average of dependant links
            dependant_probability = 0.0
            for edge in slinks:
                [src, dst] = edge
                if dst_prefix in self.link_probs_graph[src][dst]:
                    dependant_probability += self.link_probs_graph[src][dst][dst_prefix]['final_probability']
                else:
                    import ipdb;ipdb.set_trace()

            return dependant_probability/float(len(slinks))

    def compute_final_probability(self, link, dst_prefix, mpr=None):
        """Takes a link, fetches its own current probability (from mice rate passing),
        fetches its dependant probability, and computes the final one"""
        if not mpr:
            # Get own mpr
            src, dst = link
            mpr = self.link_probs_graph[src][dst]['mpr']

        # Get dependant probability
        dep_prob = self.get_dependant_probability(link, dst_prefix)
        return dep_prob * mpr

    def _start_link_probabilities_graph(self):
        """Fills up a DCGraph with the probabilities of the edges
        for each of the prefixes using these edges
        """
        # Here we store link probabilites
        link_probs_graph = self.caps_graph.copy()

        for (u, v, data) in link_probs_graph.edges_iter(data=True):
            # Remove elephant capacity
            if data.has_key('elephants_capacity'):
                data.pop('elephants_capacity')

            # Set the mpr
            data.update({'mpr': 1})

            pxs = self.get_prefixes_using_link(u, v, graph=link_probs_graph)
            # Initially all links are chosen (assuming ECMP on all possible paths)
            for px in pxs:
                if not data:
                    data = {}
                # Get edges that it depends on
                data[px] = {'final_probability': 1, 'changed': False}

        # Return it
        return link_probs_graph

    def get_predecessor_links_on_direction(self, link, dst_prefix):
        """
        Given a link with a certain direction, returns the predecessor
        links on which the probability of the given link depends on,
        if we follow the same direction of the parent link
        """
        (src, dst) = link
        predecessor_links = [(other_router, src) for other_router in self.initial_dags[dst_prefix]['dag'].predecessors(src)]

        return predecessor_links

    def get_successor_links_on_direction(self, link, dst_prefix):
        """On the direction towards dst_prefix, return the
         successor links coming after link"""
        (src, dst) = link
        dag = self.initial_dags[dst_prefix]['dag']
        successors = dag.successors(dst)
        return [(dst, s) for s in successors if not dag.is_destination_prefix(s)]

    def choose_random_dag(self, prefix, link_probabilities):
        """Given the individual link probabilities, this
        generates a random DCDag
        Returns a brand new DCDag
        """
        dc_dag = self.link_probs_graph.get_random_dag_from_probability_dcgraph(prefix, link_probabilities)
        return dc_dag

    def get_link_probabilities(self, prefix):
        """
        Returns a dictionary keyed by links in the prefix dag, pointing to the
        current probabilities of the links.
        """
        # Accumulate result here
        link_probs = {}

        # Get initial dag for prefix
        dag = self.initial_dags[prefix]['dag'].copy()

        # Iter its edges
        for (u, v, data) in dag.edges_iter(data=True):
            if prefix in self.link_probs_graph[u][v].keys():
                link_probs[(u,v)] = {mykey: self.link_probs_graph[u][v][prefix][mykey] for mykey in ['changed', 'final_probability']}
            else:
                log.error("Something weird is happening...")
                self._sendMainThreadToSleep(3000)
                import ipdb;ipdb.set_trace()
        return link_probs

    def set_probabilities_unchanged(self, prefix):
        """Iterate prob graph and set changed = False for specific prefix"""
        action = [data[prefix].update({'changed': False}) for (u, v, data) in self.link_probs_graph.edges_iter(data=True) if prefix in data]

    def get_prefixes_using_link(self, src, dst, graph=None):
        """
        Given a link in the DCGraph, returns the destination prefixes
        towards which there can be traffic going through that link
        """
        if not graph:
            graph = self.link_probs_graph

        srctype = graph.get_router_type(src)
        dsttype = graph.get_router_type(dst)
        link_direction = graph.get_edge_data(src, dst)['direction']

        if link_direction == 'uplink':
            if srctype == 'edge' and dsttype == 'aggregation':
                # Need to modify all DAGS except those for destinations connected to u (edge router)
                destinations = [px for px in self.prefixes if graph.get_destination_prefix_gateway(px) != src]

            elif srctype == 'aggregation' and dsttype == 'core':
                # Need to modify all DAGS except those for destinations connected in pod of aggregation router
                aggr_pod = graph.get_router_pod(src)

                # Filter out destinations that are not in that pods
                destinations = [px for px in self.prefixes if graph.get_destination_prefix_pod(px) != aggr_pod]

            else:
                log.error("Wrong data")
                self._sendMainThreadToSleep(3000)
                import ipdb;ipdb.set_trace()
                raise ValueError("wrong link")
        else:
            if srctype == 'aggregation' and dsttype == 'edge':
                # Need to modify only dags towards destinations connected to edge router
                destinations = [px for px in self.prefixes if graph.get_destination_prefix_gateway(px) == dst]

            elif srctype == 'core' and dsttype == 'aggregation':
                # Need to modify only dags to destinations under pod connected to aggregation
                aggr_pod = graph.get_router_pod(dst)

                # Filter destinations in that pod
                destinations = [px for px in self.prefixes if graph.get_destination_prefix_pod(px) == aggr_pod]

            else:
                log.error("Wrong data")
                self._sendMainThreadToSleep(3000)
                import ipdb;ipdb.set_trace()
                raise ValueError("wrong link")

        return destinations

    def link_probabilities_changed(self, link_probs):
        """Given likn probabilities as a dict of dicts of dicts:
        {src -> {dst -> {'probability: 1, 'changed': False}}}
        Returns true iif there is at least one link that changed == True
        """
        return any([data['changed'] == True for ((u, v), data) in link_probs.iteritems()])

    def compute_link_passing_rate(self, remaining_capacity, incoming_load, n_dests):
        """
        Compute how much of the incoming traffic to those destinations
        can actually be sent through without congestion
        """
        if not incoming_load:
            return 1

        rate = min(1, remaining_capacity/incoming_load)
        rate_dests = int(n_dests * rate)/float(n_dests)
        return rate_dests

    def get_link_mice_load(self, src, dst):
        """
        Returns the sum of the loads in the link directed to destinations
        """
        #return random.choice(self.propagated_mice_levels[src][dst]['loads'])
        if 'loads' in self.propagated_mice_levels[src][dst].keys():
            return self.propagated_mice_levels[src][dst]['loads'].mean()
        else:
            return 0

    def get_remaining_elephant_capacity(self, src, dst):
        with self.caps_lock:
            return max(0, LINK_BANDWIDTH - self.caps_graph[src][dst]['elephants_capacity'])

    def modify_link_probabilities(self, link):
        # Extract src and dst
        [src, dst] = link

        # Get destinations using link
        dests = self.get_prefixes_using_link(src, dst)

        # Get all load incoming to that link, directed to those destinations
        mice_load = self.get_link_mice_load(src, dst)

        # Get remaining capacity after elephants
        remaining_capacity = self.get_remaining_elephant_capacity(src, dst)

        # Compute individual link probability
        newMPR = self.compute_link_passing_rate(remaining_capacity, mice_load, len(dests))

        # Update its current probability
        self.link_probs_graph[src][dst]['mpr'] = newMPR

        for px in dests:
            # Get old one
            old_probability = self.link_probs_graph[src][dst][px]['final_probability']

            # Compute new one
            new_probability = self.compute_final_probability(link=link, dst_prefix=px, mpr=newMPR)

            # Update it only if it changed
            if old_probability != new_probability:
                self.link_probs_graph[src][dst][px]['final_probability'] = new_probability
                self.link_probs_graph[src][dst][px]['changed'] = True

                # Propagate
                self.propagate_to_predecessor_links(link, dest_px=px)

    def propagate_to_predecessor_links(self, link, dest_px):
        """Once compute a new link probability, it has to be propagated to all
        its predecessor edges towards dst_prefixes"""
        # Get edge source and destination
        (src, dst) = link

        # Get predecessor edges
        predecessor_links = self.get_predecessor_links_on_direction(link, dst_prefix=dest_px)

        # If there are indeed predecessor links
        if predecessor_links:
            # Iterate them and propagate
            for plink in predecessor_links:
                [psrc, pdst] = plink

                # Recompute updated final probability
                self.link_probs_graph[psrc][pdst][dest_px]['final_probability'] = self.compute_final_probability(plink, dst_prefix=dest_px)
                self.link_probs_graph[psrc][pdst][dest_px]['changed'] = True

                # Propagate on predecessor links of each link
                self.propagate_to_predecessor_links(link=plink, dest_px=dest_px)
            else:
                return

    @time_func
    def modify_path_probabilities(self, path):
        # We traverse the path inversely, because of dependencies between
        # predecessor edges
        links_on_path = self.get_links_from_path(path)
        links_on_path.reverse()

        # Compute new probabilities for the links first
        for link in links_on_path:
            self.modify_link_probabilities(link)

    @time_func
    def plotAllNewDags(self, new_dags):
        for px, dag in new_dags.iteritems():
            dag.plot('images/{0}_{1}.png'.format(px.replace('/', '|'), time.time()))

    @time_func
    def adapt_mice_dags(self, path):
        """Compute new link probabilities for the path where an elephant flow
        has been added or deleted, and generates new random DAGs"""

        # Modify the probabilities of the links in the path
        self.modify_path_probabilities(path)

        # Generate new random DAGs for destinations that probabilities changed
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

        if new_dags:
            # Apply new dags all at same time
            self.sbmanager.add_dag_requirements_from(new_dags)

            # We must re-propagate the mice loads!
            #TODO: self.propagate again


            # Plot them!
            #self.plotAllNewDags(new_dags)

    def run(self):
        while True:
            try:
                order = self.orders_queue.get(block=True)
            except:
                exit(0)

            # Checked received order
            order_type = order['type']

            if order_type == 'compute_mice_congestion_probability':
                # Create copy of dc_graph first
                with self.caps_lock:
                    caps_graph = self.caps_graph.copy()

                # Compute it
                congProb = self.totalCongestionProbability(caps_graph, threshold=self.congestion_threshold)

                # Return result to main thread
                self.results_queue.put(congProb)

                # Log a bit
                log.info("Mice congestion probability: {0} \t --threshold = {1}".format(congProb, self.max_mice_cong_prob))

                # If congestion is over the threshold
                if congProb >= self.max_mice_cong_prob:
                    log.info("Modifying mice DAGs to avoid most congested links...")
                    self.avoid_most_congested_links(caps_graph)
                # TODO: Increase probabilities of not-loaded paths such that they are also chosen

            elif order_type == 'propagate_new_distributions':
                # Take new samples based on new distributions
                self.takePropagationSamples()

            elif order_type == 'adapt_mice_to_elephants':
                # Extract path first
                flow_path = order['path']
                self.adapt_mice_dags(path=flow_path)

            elif order_type == 'terminate':
                log.info("Self-shutting down...")
                break

            else:
                log.info(order)
                continue

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
                log.debug(er_load)

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

    ## Other unused functions functions #################################

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

    def avoid_most_congested_links(self, caps_graph):
        """
        Modifies mice dags such that the most congested links are avoided
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
            self.modify_link_probabilities(u, v)

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
    me = MiceEstimatorThread(k=args.k, dc_graph=dc_graph, dst_dags=None)
    cp = me.totalCongestionProbability(threshold=args.threshold)
    import ipdb; ipdb.set_trace()
