from fibte.misc.dc_graph import DCGraph, DCDag
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from fibte import  LINK_BANDWIDTH


class MiceEstimator(object):
    def __init__(self, k, dc_graph, dst_dags=None, samples=100):
        # Set k-paramenter
        self.k = k

        # Number of samples to use
        self.samples = samples

        # Here, the capacities
        self.dc_graph = dc_graph


        self.AVERAGE_PER_DST_NOISE_LEVEL = LINK_BANDWIDTH*0.2/((self.k**3)/4)
        self.VARIANGE_PER_DST_NOISE_LEVEL = self.AVERAGE_PER_DST_NOISE_LEVEL*0.2
        #print self.AVERAGE_PER_DST_NOISE_LEVEL, self.VARIANGE_PER_DST_NOISE_LEVEL
        #print "Each host should receive around: {0}".format(self.AVERAGE_PER_DST_NOISE_LEVEL*(((self.k**3)/4)-1))

        # Start the initial dags for each destination
        if dst_dags:
            self.dags = dst_dags
        else:
            self.dags = self._createInitialDags()
            self.prefixes = self.dags.keys()

        # Crete fake distributions: in theory this
        # will be given by the thread that estimates average
        # distributions
        self.micepdfs = self._createFakeNoiseDistributions()

    def _createInitialDags(self):
        dags = {}
        network_prefixes = self.dc_graph.topo.getInitialNetworkPrefixes()
        for nwpx in network_prefixes:
            dag = self.dc_graph.get_default_ospf_dag(prefix=nwpx)
            #TODO: We should add the prefix - edge router links here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            gw = dag.get_gateway()
            dags[nwpx] = {'gateway': gw, 'dag': dag}

        return dags

    def _createFakeNoiseDistributions(self):
        """{receiver :-> {sender: [loads], sender:[]} """
        mice_pdfs = {px: {} for px in self.prefixes}
        for px in self.prefixes:
            for px2 in self.prefixes:
                if px2 != px:
                    mice_pdfs[px][px2] = np.random.normal(self.AVERAGE_PER_DST_NOISE_LEVEL, self.VARIANGE_PER_DST_NOISE_LEVEL, self.samples)
        return mice_pdfs

    def propagatePrefixNoise(self, prefix, dc_graph, i):
        """Propagates the sampled average load of the hosts towards prefix over
        the network graph"""
        # Fetch current prefix dag
        dag = self.dags[prefix]['dag']

        # Get gw
        gw = dag.get_gateway()

        # Iterate all other edge routers
        for er in self.dc_graph.edge_routers_iter():
            if er != gw:
                # Collect sum of loads from connected prefixes
                pxs = dc_graph.get_connected_destination_prefixes(er)
                er_load = sum([self.micepdfs[prefix][px][i] for px in pxs])

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

        # Take as many samples!
        for i in range(self.samples):
            dcg = self.propagateAllPrefixes(dcg, i)

        return dcg

    def totalCongestionProbability(self, threshold=0.3):
        # Create copy of dc_dag
        dcg = self.dc_graph.copy()

        # Propagate current mice distributions
        dcg = self.takePropagationSamples(dcg)

        # Accumulate congestion probabilites
        cps = []
        for (u, v, data) in dcg.edges_iter(data=True):
            if data.has_key('loads'):
                loads = data['loads']
                cp = self.linkCongestionProbability(loads, threshold)
                cps.append(cp)
            else:
                import ipdb; ipdb.set_trace()

        return self.union_congestion_probability(cps)

    def linkCongestionProbability(self, loads, threshold):
        total_samples = float(len(loads))
        congested_samples = len([l for l in loads if l > LINK_BANDWIDTH*threshold])
        return congested_samples/total_samples

    @staticmethod
    def union_congestion_probability(cps):
        """Apply function found in: http://lethalman.blogspot.ch/2011/08/probability-of-union-of-independent.html
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
                edges += [(source, succ, new_load)] + MiceEstimator.propagate_sample(dag, succ, target, new_load)
            return edges


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
