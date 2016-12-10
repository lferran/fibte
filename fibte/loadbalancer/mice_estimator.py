from fibte.misc.dc_graph import DCGraph, DCDag
from fibte.misc.flowEstimation import EstimateDemands, EstimateDemandError
import numpy as np
from fibte import  LINK_BANDWIDTH
import threading
import random
import logging
from fibte.logger import log
import copy
import time
import json
from fibte.misc import ipalias as ipalias

# Decorator function for timing purposes
def time_func(function):
    def wrapper(*args,**kwargs):
        t = time.time()
        res = function(*args,**kwargs)
        log.debug("{0} took {1}s to execute".format(function.func_name, time.time()-t))
        return res
    return wrapper

class MiceEstimatorThread(threading.Thread):
    def __init__(self, active, sbmanager, sbmanagerLock, orders_queue, flowpath_queue, capacities_graph, dags, q_server, *args, **kwargs):
        # Superclass __init__()
        super(MiceEstimatorThread, self).__init__(*args, **kwargs)

        # Wake up every:
        self.sleep_interval_s = 10

        # Get params
        self.active = active
        self.sbm = sbmanager           # Fibbign southbound manager instance
        self.sbmLock = sbmanagerLock
        self.caps_graph = capacities_graph   # Capacities left by the elephant flows
        self.dags = dags                     # Current Mice traffic dags
        self.initial_dags = copy.deepcopy(self.dags) # Keep copy of initial dags
        self.prefixes = self.initial_dags.keys()
        self.q_server = q_server
        # Queue where flow->path events are placed
        self.flowpath_queue = flowpath_queue

        # Get k
        self.k = self.caps_graph.k

        # Here we store the propagated mice loads
        self.propagated_mice_levels = self.caps_graph.copy()

        # Where to receive orders from
        self.orders_queue = orders_queue

        # Create link probabilities graph
        self.link_probs_graph = self._start_link_probabilities_graph()

        self.load_router_names()

        # We assume that 10% of the link bandwidth accounts for mice flows.
        # We assume that the mice traffic sent by a host is evenly distributed for all other destinations
        self.mice_load = 0.1 * LINK_BANDWIDTH
        self.n_hosts = (self.k**3/4)
        self.avg_host_to_host_load = (self.mice_load) / (self.n_hosts - 1)
        self.avg_load_to_other_er_host = self.avg_host_to_host_load * ((self.k/2 - 1)/float(self.n_hosts))

        # Keep track of flow demands
        self.demands = EstimateDemands()

        # Empty flow to path and edges data structures
        self.flows2paths = {}
        self.edges2flows = {(a, b): {} for (a, b) in self.propagated_mice_levels.edges()}

        # Set debug level
        log.setLevel(logging.DEBUG)

    @staticmethod
    def flowToKey(flow):
        """Fastest way to create a dictionary key out of a dictionary
        """
        return tuple(sorted(flow.items()))

    @staticmethod
    def keyToFlow(key):
        """Given a flow key, returns the flow as a dictionary"""
        return dict(key)

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
        self.r0e0 = self.caps_graph.get_router_from_position('edge', 0,0)
        self.r0e1 = self.caps_graph.get_router_from_position('edge', 1,0)

        self.r1e0 = self.caps_graph.get_router_from_position('edge', 0,1)
        self.r1e1 = self.caps_graph.get_router_from_position('edge', 1,1)

        self.r2e0 = self.caps_graph.get_router_from_position('edge', 0,2)
        self.r2e1 = self.caps_graph.get_router_from_position('edge', 1,2)

        self.r3e0 = self.caps_graph.get_router_from_position('edge', 0,3)
        self.r3e1 = self.caps_graph.get_router_from_position('edge', 1,3)

        self.r0a0 = self.caps_graph.get_router_from_position('aggregation', 0, 0)
        self.r0a1 = self.caps_graph.get_router_from_position('aggregation', 1, 0)

        self.r1a0 = self.caps_graph.get_router_from_position('aggregation', 0, 1)
        self.r1a1 = self.caps_graph.get_router_from_position('aggregation', 1, 1)

        self.r2a0 = self.caps_graph.get_router_from_position('aggregation', 0, 2)
        self.r2a1 = self.caps_graph.get_router_from_position('aggregation', 1, 2)

        self.r3a0 = self.caps_graph.get_router_from_position('aggregation', 0, 3)
        self.r3a1 = self.caps_graph.get_router_from_position('aggregation', 1, 3)

        self.rc0 = self.caps_graph.get_router_from_position('core', 0)
        self.rc1 = self.caps_graph.get_router_from_position('core', 1)
        self.rc2 = self.caps_graph.get_router_from_position('core', 2)
        self.rc3 = self.caps_graph.get_router_from_position('core', 3)

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
                    log.error("Something wrong is happening")
                    self._sendMainThreadToSleep(2000)
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
        if 'load' in self.propagated_mice_levels[src][dst].keys():
            return self.propagated_mice_levels[src][dst]['load']
        else:
            return 0

    def get_elephant_load(self, src, dst):
        """Returns the current elephant load of the (src, dst) link
        """
        if (src, dst) in self.edges2flows.keys():
            # Iterate flows on that link and return the sum of demands
            sumDemand = sum([self.demands.getDemand(fkey) for fkey in self.edges2flows[(src, dst)].iterkeys()])
            return sumDemand * LINK_BANDWIDTH

        else:
            log.error("This link is not in the edges2flows dict")
            raise ValueError

    def get_remaining_capacity(self, src, dst):
        elephant_load = self.get_elephant_load(src, dst)
        return max(0, LINK_BANDWIDTH - elephant_load)

    def modify_path_probabilities(self, path):
        # We traverse the path inversely, because of dependencies between
        # predecessor edges
        links_on_path = self.get_links_from_path(path)
        links_on_path.reverse()

        # Compute new probabilities for the links first
        for link in links_on_path:
            self.modify_link_probabilities(link)

    def modify_link_probabilities(self, link):
        # Extract src and dst
        [src, dst] = link

        # Get destinations using link
        dests = self.get_prefixes_using_link(src, dst)

        # Get all load incoming to that link, directed to those destinations
        mice_load = self.get_link_mice_load(src, dst)

        # Get remaining capacity after elephants
        remaining_capacity = self.get_remaining_capacity(src, dst)

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

    @time_func
    def modify_link_probabilities_from(self, modified_links):
        """Modify link probabilities from list of links"""
        log.info("Updating probabilities of {0} links".format(len(modified_links)))
        for link in modified_links:
            self.modify_link_probabilities(link)

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
                old_probability = self.link_probs_graph[psrc][pdst][dest_px]['final_probability']
                new_probability = self.compute_final_probability(plink, dst_prefix=dest_px)
                if old_probability != new_probability:
                    self.link_probs_graph[psrc][pdst][dest_px]['final_probability'] = new_probability
                    self.link_probs_graph[psrc][pdst][dest_px]['changed'] = True

                    # Propagate on predecessor links of each link
                    self.propagate_to_predecessor_links(link=plink, dest_px=dest_px)
                else:
                    return
        else:
            return

    @time_func
    def plotAllNewDags(self, new_dags):
        for px, dag in new_dags.iteritems():
            dag.plot('images/{0}_{1}.png'.format(px.replace('/', '|'), time.time()))

    def getLinkProbability(self, src, dst, px):
        return self.link_probs_graph[src][dst][px]

    def adaptMiceDAGs(self, modified_paths):
        """
        Given a list of paths where the elephant load has been modified:

        - Recompute link probabilities & propagate them
        - Choose new DAGs randomly
        - Set them!
        """
        # Avoids link duplicates
        modified_paths = {tuple(p) for p in modified_paths}
        modified_links = {l for p in modified_paths for l in self.get_links_from_path(p)}
        modified_links = list(modified_links)

        # Sort them
        atoe = [(a, b) for (a, b) in modified_links if self.caps_graph.is_aggregation(a) and self.caps_graph.is_edge(b)]
        ctoa = [(a, b) for (a, b) in modified_links if self.caps_graph.is_core(a) and self.caps_graph.is_aggregation(b)]
        atoc = [(a, b) for (a, b) in modified_links if self.caps_graph.is_aggregation(a) and self.caps_graph.is_core(b)]
        etoa = [(a, b) for (a, b) in modified_links if self.caps_graph.is_edge(a) and self.caps_graph.is_aggregation(b)]
        modified_links = atoe + ctoa + atoc + etoa

        # Update link probabilities
        self.modify_link_probabilities_from(modified_links)

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
            with self.sbmLock:
                self.sbm.add_dag_requirements_from(new_dags)

            # Plot them!
            #self.plotAllNewDags(new_dags)

            dags_changed = True
            return dags_changed
        else:
            dags_changed = False
            return dags_changed

    def addFlowToPath(self, flow, path):
        """Adds a new flow to path datastructures"""
        # Get key from flow
        fkey = self.flowToKey(flow)

        # Update flows2paths
        if fkey not in self.flows2paths.keys():
            self.flows2paths[fkey] = {'path': path, 'to_update': False}
        else:
            raise ValueError("Weird: flow was already in the data structure")

        # Upadte links too
        for link in self.get_links_from_path(path):
            if link in self.edges2flows.keys():
                self.edges2flows[link][fkey] = {'flow': flow, 'to_update': False}
            else:
                raise ValueError("Weird: flow was already in the data structure")

    def delFlowFromPath(self, flow):
        """Removes an ongoing flow from path"""
        # Get key from flow
        fkey = self.flowToKey(flow)

        # Update data structures
        if fkey in self.flows2paths.keys():
            old_path = self.flows2paths[fkey]['path']
            self.flows2paths.pop(fkey)
        else:
            raise ValueError("Flow wasn't in data structure")

        for link in self.get_links_from_path(old_path):
            if link in self.edges2flows.keys():
                if fkey in self.edges2flows[link]:
                    self.edges2flows[link].pop(fkey)
                else:
                    log.error("Flow wasn't in data structure")
                    return
            else:
                log.error("Weird: link not in the data structure")
                return

    def processFlowToPathEvents(self, to_process):
        """
        :param to_process: list of tuples indicating flow->path changes: (flow, path, 'add'/'del')
        :return:
        """
        modified_paths = []

        # Add or remove flows first from paths
        for flow, path, action in to_process:
            if action == 'add':
                self.demands.addFlow(flow)
                self.addFlowToPath(flow, path)
                modified_paths.append(path)

            elif action == 'del':
                self.demands.delFlow(flow)
                self.delFlowFromPath(flow)
                modified_paths.append(path)

            else:
                log.error("Wrong action to process")
                continue

        # Re-estimate flow demands
        self.demands.estimateDemandsAll()

        # Return the list of paths for which some elephant flows have changed
        return modified_paths

    def run(self):
        """"""
        # Fill up the loads according to current dags
        self.propagateLoadsOnDags()

        # Some initial values
        loop_forever = True
        sleep_time = self.sleep_interval_s
        now = None

        while loop_forever:
            if now:
                elapsed_time = time.time() - now
                log.info("Elapsed time: {0}".format(elapsed_time))
                # Time left to sleep
                sleep_time = self.sleep_interval_s - elapsed_time

            # Wait for sleep_interval
            start = time.time()
            while (time.time() - start) < sleep_time:
                if not self.orders_queue.empty():
                    order = self.orders_queue.get_nowait()
                    if order['type'] == 'terminate':
                        log.info("MiceEstimatorThread: <TERMINATE> event received. Shutting down")
                        loop_forever = False
                        break
                    else:
                        log.warning("<UNKNOWN> event received {0}".format(order))
                        continue
                time.sleep(0.2)

            # After sleeping, collect flow->path events
            if loop_forever and self.active:
                # Read all queue and accumulate
                to_process = []
                while not self.flowpath_queue.empty():
                    to_process.append(self.flowpath_queue.get_nowait())

                # Start new crono
                now = time.time()

                if to_process:
                    # Process all flow to path events
                    modified_paths = self.processFlowToPathEvents(to_process)

                    # Shift mice DAGs to adapt to spare capacity
                    dags_changed = self.adaptMiceDAGs(modified_paths)

                    if dags_changed:
                        # Propagate mice loads on new DAGs
                        self.propagateLoadsOnDags()

        log.info("Leaving")
        return

    def propagate_sample(self, dag, source, target, load):
        if source == target:
            return []
        else:
            # Get successors from DAG
            successors = dag.successors(source)

            # Compute new load
            new_load = load / float(len(successors))
            edges = []
            for succ in successors:
                edges += [(source, succ, new_load)] + self.propagate_sample(dag, succ, target, new_load)
            return edges

    @time_func
    def propagateLoadsOnDags(self):
        """Propagates the sampled average load of the hosts towards prefix over
        the network graph"""
        # Insert arrays to accumulate load samples
        for (a, b, data) in self.propagated_mice_levels.edges_iter(data=True):
            data['load'] = 0.0

        for prefix in self.prefixes:
            # Fetch current prefix dag
            dag = self.dags[prefix]['dag']
            gw = self.dags[prefix]['gateway']

            # Iterate all other edge routers
            for er in self.propagated_mice_levels.edge_routers_iter():
                if er != gw:
                    # Obtain the total load going from er to the upper lyer
                    er_load = self.avg_host_to_host_load * (self.k/2)

                    # Propagate it
                    edges = self.propagate_sample(dag=dag, source=er, target=gw, load=er_load)

                    # Iterate edges and sum pertinent load
                    for (u, v, load) in edges:
                        self.propagated_mice_levels[u][v]['load'] += load


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
    import ipdb; ipdb.set_trace()
