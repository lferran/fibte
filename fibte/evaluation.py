import subprocess
import time
import os.path
import threading
import os

from os import listdir
from os.path import isfile, join

from fibte.misc.unixSockets import UnixServer, UnixClient
from fibte.trafficgen.udpTrafficGeneratorBase import SAVED_TRAFFIC_DIR

class FlowServers(object):
    def __init__(self):
        pass

    def start(self):
        subprocess.call("python start_flow_servers.py --ip_alias", shell=True)
        time.sleep(10)

    def stop_ncs(self):
        subprocess.call("killall nc", shell=True)

    def stop(self):
        subprocess.call("sudo kill -9 $(ps aux | grep 'flowServer.py' | awk '{print $2}')", shell=True)
        self.stop_ncs()

    def running(self):
        ps_cmd = ['ps', 'aux']
        grep_cmd = ['grep', 'flowServer']
        awk_cmd = ['awk', '{print $2}']
        ps = subprocess.Popen(ps_cmd, stdout=subprocess.PIPE)
        grep = subprocess.Popen(grep_cmd, stdin=ps.stdout, stdout=subprocess.PIPE)
        awk_cmd = subprocess.Popen(awk_cmd, stdin=grep.stdout, stdout=subprocess.PIPE)
        ps.stdout.close()
        grep.stdout.close()
        out, err = awk_cmd.communicate()
        pids = out.split('\n')
        if len(pids) > 2:
            return True
        else:
            return False

    def restart(self):
        self.stop()
        self.start()

class LoadBalancer(object):
    def __init__(self):
        pass

    def start(self, algorithm, sample=None):
        algorithm = algorithm.lower()
        if algorithm == 'elephant-dag-shifter' and sample == True:
            subprocess.call("python loadbalancer.py --algorithm {0} --sample &".format(algorithm), shell=True)
        else:
            subprocess.call("python loadbalancer.py --algorithm {0} &".format(algorithm), shell=True)

    def stop(self):
        subprocess.call("kill -9 $(ps aux | grep 'loadbalancer.py' | awk '{print $2}')", shell=True)

    def running(self):
        ps_cmd = ['ps', 'aux']
        grep_cmd = ['grep', 'loadbalancer.py']
        awk_cmd = ['awk', '{print $2}']

        ps = subprocess.Popen(ps_cmd, stdout=subprocess.PIPE)
        grep = subprocess.Popen(grep_cmd, stdin=ps.stdout, stdout=subprocess.PIPE)
        awk_cmd = subprocess.Popen(awk_cmd, stdin=grep.stdout, stdout=subprocess.PIPE)

        ps.stdout.close()
        grep.stdout.close()

        out, err = awk_cmd.communicate()
        pids = out.split('\n')
        if len(pids) > 2:
            return True
        else:
            return False

class Network(object):
    def __init__(self):
        self.utils = Utils()

    def start(self, fair_queues=False):
        if fair_queues:
            subprocess.call("python network_example.py -k 4 --ip_alias --fair_queues &", shell=True)
        else:
            subprocess.call("python network_example.py -k 4 --ip_alias &", shell=True)
        time.sleep(30)

    def stop(self):
        fibpids = self.utils.pgrep('fibbingnode')
        counterpids = self.utils.pgrep('collectCounter')
        nwpids = self.utils.pgrep('python network_example')
        allpids = fibpids + counterpids + nwpids
        for pid in allpids:
            if pid:
                subprocess.call("kill -9 {0}".format(pid), shell=True)
        time.sleep(2)
        self.clean()

    def clean(self):
        subprocess.call("killall zebra ospfd", shell=True)
        subprocess.call("rm /tmp/* ; rm /var/run/netns/*", shell=True)

    def running(self):
        ps_cmd = ['ps', 'aux']
        grep_cmd = ['grep', 'network_example']
        awk_cmd = ['awk', '{print $2}']
        ps = subprocess.Popen(ps_cmd, stdout=subprocess.PIPE)
        grep = subprocess.Popen(grep_cmd, stdin=ps.stdout, stdout=subprocess.PIPE)
        awk_cmd = subprocess.Popen(awk_cmd, stdin=grep.stdout, stdout=subprocess.PIPE)
        ps.stdout.close()
        grep.stdout.close()
        out, err = awk_cmd.communicate()
        pids = out.split('\n')
        if len(pids) > 1:
            return True
        else:
            return False

class Traffic(object):
    def __init__(self):
        self.saved_traffic_dir = SAVED_TRAFFIC_DIR

    def file_exists(self, filename):
        return os.path.isfile(filename)

    def get_pattern_args_filename(self, pattern, pattern_args):
        """Assumes pattern_args is a dict"""
        if pattern == 'random':
            return None

        elif pattern == 'staggered':
            sameEdge = pattern_args.get('sameEdge')
            samePod = pattern_args.get('samePod')
            return "se{0}sp{1}".format(sameEdge, samePod)

        elif pattern == 'bijection':
            return None

        elif pattern == 'stride':
            i = pattern_args.get('i')
            return "i{0}".format(i)

    def get_filename(self, pattern, pattern_args, n_elephants, mice_avg, duration):
        """Return filename sample pattern"""
        pattern_args_fn = self.get_pattern_args_filename(pattern, pattern_args)
        filename = '{0}'.format(self.saved_traffic_dir)
        anames = ['tgf_tcp', '{0}', pattern_args_fn, 'nelep{1}','mavg{2}' 't{3}', 'ts{4}']
        filename += '_'.join([a for a in anames if a != None])
        filename = filename.format(pattern, str(n_elephants), str(mice_avg).replace('.', ','), duration, 1)
        filename += '.traffic'
        return filename

    def start(self, pattern, pattern_args, n_elephants, mice_avg, duration):
        traffic_file = self.get_filename(pattern, pattern_args, n_elephants, mice_avg, duration)
        if self.file_exists(traffic_file):
            tg_cmd = "python trafficgen/tcpElephantFiller.py --load_traffic {0}".format(traffic_file)
        else:
            if pattern_args:
                    tg_cmd = "python trafficgen/tcpElephantFiller.py --pattern {0} --pattern_args \"{1}\" --n_elephants {2} --mice_avg {3} --time {4} --save_traffic"
                    tg_cmd = tg_cmd.format(pattern, pattern_args, n_elephants, mice_avg, duration)
            else:
                tg_cmd = "python trafficgen/tcpElephantFiller.py --pattern {0} --n_elephants {1} --mice_avg {2} --time {3} --save_traffic"
                tg_cmd = tg_cmd.format(pattern, n_elephants, mice_avg, duration)

        # Call command
        subprocess.call(tg_cmd, shell=True)

    def stop(self):
        tg_cmd = "python trafficgen/tcpElephantFiller.py --terminate"
        subprocess.call(tg_cmd, shell=True)

class Plot(object):
    def __init__(self):
        self.script = 'monitoring/makeDelays.py'
        pass

    def plot(self, parent_folder, algo_list, to_plot, plot_name, ratio=False, difference=False):
        if not ratio and not difference:
            plt_cmd = "python {4} --parent_folder {0} --algo_list {1} --to_plot {2} --plot_name {3}"
        elif ratio:
            plt_cmd = "python {4} --parent_folder {0} --algo_list {1} --to_plot {2} --plot_name {3} --ratio"
        elif difference:
            plt_cmd = "python {4} --parent_folder {0} --algo_list {1} --to_plot {2} --plot_name {3} --difference"
        else:
            raise ValueError("Ratio and Difference not allowed at the same time")
        plt_cmd = plt_cmd.format(parent_folder, ' '.join(algo_list), to_plot, plot_name, self.script)
        subprocess.call(plt_cmd, shell=True)

class Utils(object):
    def __init__(self):
        self.fibte_dir = os.getcwd()
        self.trafficgen_dir = os.path.join(self.fibte_dir, 'trafficgen')
        self.monitoring_dir = os.path.join(self.fibte_dir, 'monitoring')
        self.results_dir = os.path.join(self.monitoring_dir, 'results')
        self.throughput_dir = os.path.join(self.results_dir, 'throughput')
        self.delay_dir = os.path.join(self.results_dir, 'delay')
        self.root_dir = os.path.join('/', 'root')

    def file_exists(self, path_to_file):
        return os.path.isfile(path_to_file)

    def dir_exists(self, path_to_dir):
        return os.path.isdir(path_to_dir)

    def mv(self, src, dst):
        subprocess.call('mv {0} {1}'.format(src, dst), shell=True)

    def mkdir(self, dir, p=None):
        if p:
            subprocess.call("mkdir -p {0}".format(dir), shell=True)
        else:
            subprocess.call("mkdir {0}".format(dir), shell=True)

    def has_mice_delays(self, folder):
        for i in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, i)) and 'mice_' in i:
                return True

    def has_elephant_delays(self, folder):
        for i in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, i)) and 'elep_' in i:
                return True

    def pgrep(self, regex):
        ps_cmd = ['ps', 'aux']
        grep_cmd = ['grep', regex]
        awk_cmd = ['awk', '{print $2}']
        ps = subprocess.Popen(ps_cmd, stdout=subprocess.PIPE)
        grep = subprocess.Popen(grep_cmd, stdin=ps.stdout, stdout=subprocess.PIPE)
        awk_cmd = subprocess.Popen(awk_cmd, stdin=grep.stdout, stdout=subprocess.PIPE)
        ps.stdout.close()
        grep.stdout.close()
        out, err = awk_cmd.communicate()
        pids = out.split('\n')
        return pids

    def join(self, a, b):
        return os.path.join(a, b)

class Test(object):
    def __init__(self):
        self.name = 'default-test'
        self.tests = self.load_tests()
        self.index = 0
        self.max_index = len(self.tests) - 1
        self.utils = Utils()

    def mkdir_test(self, parent_dir):
        testdir = os.path.join(parent_dir, self.name)
        delaydir = os.path.join(testdir, 'delay')
        self.utils.mkdir(delaydir, p=True)
        return delaydir

    def load_tests(self):
        tests = []
        n_elephants = 16
        mice_avg = 6
        duration = 100
        patterns = [('stride', {'i': 4}),
                    ('stride', {'i': 2}),
                    ('random', None),
                    ('staggered', {'sameEdge': 0.2, 'samePod': 0.5}),
                    ('staggered', {'sameEdge': 0.2, 'samePod': 0.3}),
                    ('bijection', None)]

        algos = [('ecmp', None),
                 ('elephant-dag-shifter', None),
                 ('elephant-dag-shifter', '--sample'),
                 ('mice-dag-shifter', None)]

        for (pattern, pargs) in patterns:
            for (algo, aargs) in algos:
                d = {'pattern': pattern, 'pattern_args': pargs,
                'n_elephants': n_elephants, 'mice_avg': mice_avg,
                'duration': duration}
                d.update({'algorithm': algo, 'algorithm_args': aargs})
                tests.append(d)
        return tests

    def __iter__(self):
        return self

    def next(self):
        if self.index > self.max_index:
            raise StopIteration
        else:
            self.index += 1
            return self.tests[self.index - 1]

class ElephantFlowsTest(Test):
    def __init__(self):
        super(ElephantFlowsTest, self).__init__()
        self.name = 'elephant-flows-test'

    def load_tests(self):
        # Define here your simple test
        tests = []
        n_elephants = [16, 32]
        mice_avg = 0.0
        duration = 100
        fair_queues = [True, False]
        #fair_queues = [False]

        patterns = [
            ('stride', {'i': 2}),
            ('stride', {'i': 4}),
            ('random', None),
            ('bijection', None),
            ('staggered', {'sameEdge': 0.2, 'samePod': 0.3}),
            ('staggered', {'sameEdge': 0.2, 'samePod': 0.5}),
            ]

        algos = [
            ('ecmp', None),
            ('elephant-dag-shifter', None),
            ('elephant-dag-shifter', '--sample'),
        ]

        # Iterate traffic pattern
        for (pattern, pargs) in patterns:
            for n_ele in n_elephants:
                for (algo, aargs) in algos:
                    for fq in fair_queues:
                        d = {'fair_queues': fq,
                             'pattern': pattern,
                             'pattern_args': pargs,
                             'n_elephants': n_ele,
                             'mice_avg': mice_avg,
                             'duration': duration,
                             'algorithm': algo,
                             'algorithm_args': aargs}
                        tests.append(d)
        return tests

class MiceFlowsTest(Test):
    def __init__(self):
        super(MiceFlowsTest, self).__init__()
        self.name = 'mice-flows-test'

    def load_tests(self):
        # Define here your simple test
        tests = []
        n_elephants = [16, 32]
        mice_avg = 4
        duration = 100
        fair_queues = [True, False]

        patterns = [
            ('stride', {'i': 2}),
            ('random', None),
            ('bijection', None),
            ('staggered', {'sameEdge': 0.2, 'samePod': 0.3}),
            ]

        algos = [
            ('ecmp', None),
            ('elephant-dag-shifter', None),
            ('mice-dag-shifter', None),
            ]

        # Iterate traffic pattern
        for (pattern, pargs) in patterns:
            for n_ele in n_elephants:
                for algo, aargs in algos:
                    for fq in fair_queues:
                        d = {'fair_queues': fq,
                             'pattern': pattern,
                             'pattern_args': pargs,
                             'n_elephants': n_ele,
                             'mice_avg': mice_avg,
                             'duration': duration,
                             'algorithm': algo,
                             'algorithm_args': aargs}
                        tests.append(d)
        return tests

class Evaluation(object):
    def __init__(self):
        self.flowServers = FlowServers()
        self.network = Network()
        self.loadBalancer = LoadBalancer()
        self.traffic = Traffic()
        self.utils = Utils()
        self.plot = Plot()

        # Define here the tests we want to run
        self.tests = [
            #MiceFlowsTest(),
            #ElephantFlowsTest(),
            ]
        self.results_dir = self.utils.join(self.utils.root_dir, 'evaluation_results')
        self.serverSocket = UnixServer("/tmp/evaluationServer")
        self.ownClient = UnixClient("/tmp/evaluationServer")
        self.ownStopTrafficTimer = None

    def startEnvironment(self, fair_queues=False):
        self.network.start(fair_queues)
        self.flowServers.start()

        self.serverSocket = UnixServer("/tmp/evaluationServer")
        self.ownClient = UnixClient("/tmp/evaluationServer")

    def stopEnvironment(self):
        self.network.stop()
        self.flowServers.stop()

    def restartEnvironment(self, fair_queues=False):
        self.stopEnvironment()
        self.startEnvironment(fair_queues)

    def ownStopTraffic(self):
        self.ownClient.send("auto-terminate")

    def waitForTrafficToFinish(self):
        # Start maximum waiting time timer
        self.ownStopTrafficTimer = threading.Timer(60*10, self.ownStopTraffic)

        # Blocking read to the serverSocket
        st = time.time()
        print("*** WAITING FOR TRAFFIC TO FINISH " + "*"*60)
        finished = self.serverSocket.receive()
        if finished == 'finished':
            if self.ownStopTrafficTimer and self.ownStopTrafficTimer.is_alive():
                self.ownStopTrafficTimer.cancel()
            print("*** TRAFFIC FINISHED SUCCESSFULLY after {0}seconds".format(time.time() - st) + "*"*60)
            return
        else:
            print("*** ERROR ON FINISHING TRAFFIC after {0}seconds".format(time.time() - st) + "*"*60)
            return

    def killall(self):
        if self.network.running():
            self.traffic.stop()
            self.loadBalancer.stop()
        self.stopEnvironment()

    def startLoadBalancer(self, sample):
        """Gets the parameters from the sample and starts the load balancer"""
        algo = sample.get('algorithm')
        aargs = sample.get('algorithm_args')
        self.loadBalancer.start(algo, aargs)

    def startTraffic(self, sample):
        """Gets parameters from the sample and starts the traffic generator"""
        # Start traffic
        pattern = sample.get('pattern')
        pargs = sample.get('pattern_args')
        n_elephants = sample.get('n_elephants')
        mice_avg = sample.get('mice_avg')
        duration = sample.get('duration')
        self.traffic.start(pattern, pargs, n_elephants, mice_avg, duration)

    def createPatternFolder(self, sample, delaydir):
        """Creates, if it doesn't exist already, a folder for the specified traffic pattern
        """
        # Parses arguments from sample
        pattern = sample.get('pattern')
        pargs = sample.get('pattern_args')
        n_elephants = sample.get('n_elephants')
        mice_avg = sample.get('mice_avg')
        duration = sample.get('duration')

        # Create directory name
        if pargs:
            sampledirname = "{0}_{1}_nelep{2}_avmice{3}_{4}s".format(pattern, str(pargs).replace(' ', ''),
                                                                     str(n_elephants),
                                                                     str(mice_avg).replace('.', ','), duration)
        else:
            sampledirname = "{0}_nelep{1}_avmice{2}_{3}s".format(pattern, str(n_elephants),
                                                                 str(mice_avg).replace('.', ','),
                                                                 duration)
        # Join it with the delay parent directory
        sampledir = os.path.join(delaydir, sampledirname)
        if not self.utils.dir_exists(sampledir):
            self.utils.mkdir(sampledir, p=True)


        # Return it
        return sampledir

    def createAlgoFolder(self, sample, patterndir):
        """Given the traffic pattern directory, creates a specific
        folder for the algorithm specified in the sample
        """
        # Parse parameters
        algo = sample.get('algorithm')
        aargs = sample.get('algorithm_args')
        fair_queues = sample.get('fair_queues')

        # Create folder for algorithm
        algoname = algo
        if algo == 'elephant-dag-shifter':
            if aargs:
                algoname += "_{0}".format('sampled')
            else:
                algoname += "_{0}".format('best')

        # Add queue type
        if fair_queues:
            queue = 'fairQueues'
        else:
            queue = 'pFifo'
        algoname = algoname + "_{0}".format(queue)

        # Create folder
        algodir = os.path.join(patterndir, algoname)
        self.utils.mkdir(algodir, p=True)

        # Return it
        return algodir

    def moveResults(self, algodir, logfile=None):
        """Collects all completion times from the default results folder and moves it to the
        corresponding pattern/algo evaaluation results folder"""

        # Move all flow completion times to delay
        if self.utils.has_mice_delays(self.utils.delay_dir):
            self.utils.mv(os.path.join(self.utils.delay_dir, 'mice_*'), algodir)

        if self.utils.has_elephant_delays(self.utils.delay_dir):
            self.utils.mv(os.path.join(self.utils.delay_dir, 'elep_*'), algodir)

        if logfile:
            self.utils.mv(logfile, algodir)

    def plotTest(self, test, patterns=None, algos=None):
#        import ipdb; ipdb.set_trace()
        if test not in listdir(self.results_dir):
            print("ERROR: {0} not in {1}".format(test, self.results_dir))

        testdir = self.utils.join(self.results_dir, test)
        testdir = self.utils.join(testdir, 'delay')
        if patterns:
            pattern_list = [d for d in listdir(testdir) if os.path.isdir(self.utils.join(testdir, d)) and d in patterns]
        else:
            pattern_list = [d for d in listdir(testdir) if os.path.isdir(self.utils.join(testdir, d))]
        if 'mice' in test:
            to_plot = 'all'
            # Iterate patterns
            for pattern in pattern_list:
                patterndir = self.utils.join(testdir, pattern)
                if algos:
                    algo_list = [d for d in listdir(patterndir) if os.path.isdir(self.utils.join(patterndir, d)) and d in algos]
                else:
                    algo_list = [d for d in listdir(patterndir) if os.path.isdir(self.utils.join(patterndir, d))]
                parent_folder = patterndir
                plot_name = "{0}_allCompared".format(pattern)
                self.plot.plot(parent_folder, algo_list, to_plot, plot_name)

                if 'mice' in test:
                    algo_list = ['ecmp_pFifo', 'ecmp_fairQueues']
                    parent_folder = patterndir
                    plot_name = "{0}__ecmpFIFO_vs_ecmpFairQueues".format(pattern)
                    self.plot.plot(parent_folder, algo_list, to_plot, plot_name)

        elif 'elephant' in test:
            to_plot = 'elephant'
            # Iterate patterns
            for pattern in pattern_list:
                patterndir = self.utils.join(testdir, pattern)
                if algos:
                    algo_list = [d for d in listdir(patterndir) if os.path.isdir(self.utils.join(patterndir, d)) and d in algos]
                else:
                    algo_list = [d for d in listdir(patterndir) if os.path.isdir(self.utils.join(patterndir, d))]

                parent_folder = patterndir
                plot_name = "{0}_allCompared".format(pattern)
                self.plot.plot(parent_folder, algo_list, to_plot, plot_name,)
                self.plot.plot(parent_folder, algo_list, to_plot, plot_name, ratio=True)
                self.plot.plot(parent_folder, algo_list, to_plot, plot_name, difference=True)

        else:
            print("*** ERROR: elephant or mice should be in the name of the test")

    def emptyDelayDir(self):
        subprocess.call("rm {0}".format(os.path.join(self.utils.delay_dir, 'mice_*')), shell=True)
        subprocess.call("rm {0}".format(os.path.join(self.utils.delay_dir, 'elep_*')), shell=True)

    def run(self, from_index=None):
        # Check if start index was given
        start_index = 0 if not from_index else from_index
        current_index = 0

        # Get all tests with its samples
        tests = {t: list(t) for t in self.tests}

        # How many tests?
        n_tests = len(tests.keys())

        # Count total samples
        n_samples = sum([len(s) for s in tests.itervalues()])

        start_time = time.time()
        try:
            for testindex, test in enumerate(tests.iterkeys()):
                print("*** Starting test {0} {1}/{2}".format(test.name, testindex, n_tests)+"*"*60 )

                # Create folder for results
                delaydir = test.mkdir_test(parent_dir=self.results_dir)

                # Get test samples
                samples = tests[test]
                for sample in samples:
                    if current_index >= start_index:
                        # Restart network and flowServers
                        fair_queues = sample.get('fair_queues')
                        self.restartEnvironment(fair_queues)

                        # Create dir for this pattern
                        patterndir = self.createPatternFolder(sample, delaydir)

                        # Create dif for this algorithm
                        algodir = self.createAlgoFolder(sample, patterndir)

                        print("*** Starting sample ({1}/{2}): {0}".format(sample, current_index, n_samples))

                        # Start LB and Traffic
                        self.startLoadBalancer(sample)
                        self.startTraffic(sample)

                        # Wait for some time
                        self.waitForTrafficToFinish()

                        # Stop traffic
                        self.killall()

                        # Move results to specified folder
                        self.moveResults(algodir)
                    else:
                        print("*** Skipping sample ({1}/{2}): {0}".format(sample, current_index, n_samples))
                    # Increment sample index
                    current_index += 1

        except KeyboardInterrupt:
            print("*** CTRL-C catched! at test: {0} sample index: {1}/{2}".format(test.name, current_index, n_samples)+"*"*60)
            print("*** Sample args: {0}".format(sample))

        finally:
            self.traffic.stop()
            self.loadBalancer.stop()
            self.stopEnvironment()
            self.emptyDelayDir()
            print("*** Finishing evaulation after {0} minutes".format((time.time() - start_index) / 60.0) + "*" * 60)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--run', help='Run evaluations', action="store_true", default=False)
    parser.add_argument('--from_index', help='Start at specific sample', type=int, default=None)
    parser.add_argument('--plot_test', help="Make plots of given test", type=str, default='')
    parser.add_argument('--patterns', nargs='+', help="Specify patterns that you want to plot", type=str, default=None)
    parser.add_argument('--algos', nargs='+', help="Specify algorithms that you want to plot", type=str, default=None)

    args = parser.parse_args()

    # Start Evaluation object
    ev = Evaluation()

    #import ipdb; ipdb.set_trace()

    if args.run:
        # Kill all that's going on
        ev.killall()

        # Run evaluation tests
        ev.run(from_index=args.from_index)

    elif args.plot_test:
        # make plots for the given test
        ev.plotTest(args.plot_test, patterns=args.patterns, algos=args.algos)

    else:
        print("Nothing to do!")