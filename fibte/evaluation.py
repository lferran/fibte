import subprocess
import time
import os.path
import os
import shutil
import ast

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
        subprocess.call("kill -9 $(ps aux | grep 'flowServer' | awk '{print $2}')", shell=True)
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

    def start(self):
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

class Utils(object):
    def __init__(self):
        self.fibte_dir = os.getcwd()
        self.trafficgen_dir = os.path.join(self.fibte_dir, 'trafficgen')
        self.monitoring_dir = os.path.join(self.fibte_dir, 'monitoring')
        self.results_dir = os.path.join(self.monitoring_dir, 'results')
        self.throughput_dir = os.path.join(self.results_dir, 'throughput')
        self.delay_dir = os.path.join(self.results_dir, 'delay')

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
        if self.index >= self.max_index:
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
        n_elephants = [16, 32, 64]
        mice_avg = 0.0
        duration = 100
        patterns = [
            ('stride', {'i': 4}),
            #('random', None),
            ]

        algos = [
            ('ecmp', None),
            ('elephant-dag-shifter', None),
            #('elephant-dag-shifter', '--sample'),
            #('mice-dag-shifter', None),
            ]

        for (pattern, pargs) in patterns:
            for n_ele in n_elephants:
                for (algo, aargs) in algos:
                    d = {'pattern': pattern, 'pattern_args': pargs,
                    'n_elephants': n_ele, 'mice_avg': mice_avg,
                    'duration': duration}
                    d.update({'algorithm': algo, 'algorithm_args': aargs})
                    tests.append(d)
        return tests

class Evaluation(object):
    def __init__(self):
        self.flowServers = FlowServers()
        self.network = Network()
        self.loadBalancer = LoadBalancer()
        self.traffic = Traffic()
        self.utils = Utils()

        # Define here the tests we want to run
        self.tests = [ElephantFlowsTest()]

    def startEnvironment(self):
        self.network.start()
        self.flowServers.start()

    def stopEnvironment(self):
        self.network.stop()
        self.flowServers.stop()

    def restartEnvironment(self):
        self.stopEnvironment()
        self.startEnvironment()

    def killall(self):
        if self.network.running():
            self.traffic.stop()
            self.loadBalancer.stop()
        self.stopEnvironment()

    def run(self):
        # Run all tests
        try:
            evaluation_results_dir = os.path.join(self.utils.fibte_dir, 'evaluation_results/')
            for test in self.tests:
                print("*** Starting test: {0}".format(test.name))

                # Create folder for results
                delaydir = test.mkdir_test(parent_dir=evaluation_results_dir)

                for sample in test:
                    # Restart network and flowServers
                    self.restartEnvironment()
                    # Start loadbalancer
                    algo = sample.get('algorithm')
                    aargs = sample.get('algorithm_args')
                    self.loadBalancer.start(algo, aargs)

                    # Start traffic
                    pattern = sample.get('pattern')
                    pargs = sample.get('pattern_args')
                    n_elephants = sample.get('n_elephants')
                    mice_avg = sample.get('mice_avg')
                    duration = sample.get('duration')
                    self.traffic.start(pattern, pargs, n_elephants, mice_avg, duration)

                    # Create dir for this sample
                    sampledirname = "{0}_{1}_nelep{2}_avmice{3}_{4}s".format(pattern, str(pargs).replace(' ', ''),
                                                                             str(n_elephants),
                                                                             str(mice_avg).replace('.', ','),
                                                                             duration)
                    sampledir = os.path.join(delaydir, sampledirname)
                    if not self.utils.dir_exists(sampledir):
                        self.utils.mkdir(sampledir, p=True)

                    # Create folder for algorithm
                    if aargs:
                        algoname = "{0}_{1}".format(algo, 'True')
                    else:
                        algoname = "{0}".format(algo)

                    algodir = os.path.join(sampledir, algoname)
                    self.utils.mkdir(algodir, p=True)

                    # Wait for some time
                    print("*** Sleeping until traffic simulation finishes...")
                    time.sleep(duration * 2.5)

                    # Stop traffic
                    self.killall()

                    # Move all flow completion times to delay
                    if self.utils.has_mice_delays(self.utils.delay_dir):
                        self.utils.mv(os.path.join(self.utils.delay_dir, 'mice_*'), algodir)

                    if self.utils.has_elephant_delays(self.utils.delay_dir):
                        self.utils.mv(os.path.join(self.utils.delay_dir, 'elep_*'), algodir)

        except KeyboardInterrupt:
            print("*** CTRL-C catched!")
        finally:
            print("*** Finishing evaulation")
            self.traffic.stop()
            self.loadBalancer.stop()
            self.stopEnvironment()

if __name__ == '__main__':
    # start_loadBalancer('elephant-dag-shifter')
    # is_loadbalancer_running()
    # kill_loadBalancer()
    # is_loadbalancer_running()

    import ipdb;ipdb.set_trace()
    ev = Evaluation()
    ev.killall()
    ev.run()

