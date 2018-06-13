# epanet toolkit
import json

import numpy as np
import pandas as pd
from epanettools.epanet2 import *
from epanettools.epanettools import *
from plotly import tools
from plotly.graph_objs import *
# plotting imports
from plotly.offline import plot

# extending the EPANetSimulation class to ease acces to
# simulation routines
class ENSim(EPANetSimulation):
    EN_INIT = 10

    def __init__(self, network_file, pdd=False):
        # careful when using pdd = true for residues simulations
        self.json_sim = {}
        super().__init__(network_file, pdd)

    def set_emitter(self, node_index, emitter_val):
        if self.network.nodes[node_index].node_type is EN_JUNCTION:
            ENSim._getncheck(self.ENsetnodevalue(node_index, EN_EMITTER, emitter_val))

    def set_basedemand(self, node_index, demand_val):
        if self.network.nodes[node_index].node_type is EN_JUNCTION:
            ENSim._getncheck(self.ENsetnodevalue(node_index, EN_BASEDEMAND, demand_val))

    def set_emitters(self, emitter_info=None):

        if emitter_info is None:
            # if arg is none reset emitter values
            for node_index in self.network.nodes:
                self.set_emitter(node_index, 0)
        else:

            for node_index, emitter_val in emitter_info:
                self.set_emitter(node_index, emitter_val)

    def get_nodes_data(self, data_query, emitter=(1, 0)):

        no_nodes = ENSim._getncheck(self.ENgetcount(EN_NODECOUNT)) - ENSim._getncheck(self.ENgetcount(EN_TANKCOUNT))
        t_step = 1
        node_values = {}

        for queries in data_query:
            node_values[queries] = [[] for _ in range(no_nodes)]

        node_values["EMITTER_NODE"] = emitter[0]
        node_values["EMITTER_VAL"] = emitter[1]

        # initialize network for hydraulic process

        ENSim._getncheck(self.ENinitH(ENSim.EN_INIT))

        while t_step > 0:

            self.ENrunH()

            for node_index in range(1, no_nodes + 1):
                for query_type in data_query:
                    ret_val = ENSim._getncheck(self.ENgetnodevalue(node_index, eval(query_type)))
                    node_values[query_type][node_index - 1].append(ret_val)

            t_step = ENSim._getncheck(self.ENnextH())

        for key in node_values:
            node_values[key] = np.transpose(node_values[key]).tolist()

        return node_values

    def get_links_data(self, data_query, emitter=(1, 0)):

        no_links = self.ENgetcount(EN_LINKCOUNT)[1]
        t_step = 1
        link_values = {}

        for queries in data_query:
            link_values[queries] = [[] for _ in range(no_links)]

        link_values["EMITTER_NODE"] = emitter[0]
        link_values["EMITTER_VAL"] = emitter[1]

        # initialize network for hydraulic process

        ENSim._getncheck(self.ENinitH(ENSim.EN_INIT))

        while t_step > 0:
            ENSim._getncheck(self.ENrunH())

            for link_index in range(1, no_links + 1):
                for query_type in data_query:
                    ret_val = ENSim._getncheck(self.ENgetlinkvalue(link_index, eval(query_type)))
                    link_values[query_type][link_index - 1].append(ret_val)

            t_step = ENSim._getncheck(self.ENnextH())

        for key in link_values:
            link_values[key] = np.transpose(link_values[key]).tolist()

        return link_values

    def query_network(self, sim_dict):
        """
        :param sim_dict: a dict containing info about the network
        has the form
        {
            simulation_name : "name",
            simulation_type: "H" or "Q"
            emitter_values : [ (node_index, emitter_value) ]
            query : {
                nodes : [ "EN_PRESSURE"
                ]
                links : [ "EN_VELOCITY"
                ]
            }
        }
        :return: JSON with required data
        output_json format:
        {
            SIM_NAME = simulation-name
            NODE_VALUES = [
                {
                    "EN_PRESSURE" : [ [values for each node]]
                    "EMITTER_VAL" :
                    "EMITTER_NODE" :
                }
            ]

        }

        """

        # for the moment i'll treat only hydraulic simulations :)

        # initialize network simulaton

        ENSim._getncheck(self.ENopenH())

        # initialize session
        ENSim._getncheck(self.ENinitH(ENSim.EN_INIT))

        node_query = False
        link_query = False
        simulations = False

        # check json for querried data

        # node info:
        try:
            if sim_dict["query"]["nodes"]:
                node_query = True
        except KeyError:
            node_query = False

        # link info
        try:
            if sim_dict["query"]["links"]:
                link_query = True
        except KeyError:
            link_query = False

        # emitter info:
        try:
            simulations = sim_dict["emitter_values"]
        except KeyError:
            simulations = False

        if simulations:
            node_values = []
            link_values = []

            # in order to plot residues we need to simulate the case where there is no leakage

            self.set_emitters()
            if node_query:
                node_values.append(
                    self.get_nodes_data(sim_dict["query"]["nodes"]))

            if link_query:
                link_values.append(
                    self.get_links_data(sim_dict["query"]["links"]))

            for node_index, emitter_value in simulations:
                print("Simulating emitter in node no {} with value {}".format(node_index, emitter_value))

                self.set_emitter(node_index, emitter_value)

                if node_query:
                    node_values.append(
                        self.get_nodes_data(sim_dict["query"]["nodes"], emitter=(node_index, emitter_value)))

                if link_query:
                    link_values.append(
                        self.get_links_data(sim_dict["query"]["links"], emitter=(node_index, emitter_value)))

                # reset emitter values everywhere in network
                self.set_emitters()

        else:

            if node_query:
                node_values = [self.get_nodes_data(sim_dict["query"]["nodes"])]
            else:
                node_values = []

            if link_query:
                link_values = [self.get_links_data(sim_dict["query"]["links"])]
            else:
                link_values = []

        ENSim._getncheck(self.ENcloseH())
        ENSim._getncheck(self.ENclose())

        self.__init__(self.OriginalInputFileName)
        self.json_sim = {
            "SIM_NAME": sim_dict["simulation_name"],
            "NODE_VALUES": node_values,
            "LINK_VALUES": link_values
        }
        return self.json_sim

    def get_time_step(self, pattern_id=1):
        """
        returns the time_step of the network in minutes
        :param pattern_id:
        :return:
        """
        return (24 * 60) / ENSim._getncheck(self.ENgetpatternlen(pattern_id))

    def plot(self, json_data, residues=False):
        """
        utility function used to plot data from network simulations
        WIP
        :param json_data:
        :return:
        """
        values = json_data["NODE_VALUES"]
        date_range = pd.date_range('1/1/2018', periods=97, freq='15min')

        if residues:
            # consider the first value of the JSON as the reference
            ref = values[0]
            for emitter in values:
                trace = []
                data = np.transpose(emitter["EN_PRESSURE"]) - np.transpose(ref["EN_PRESSURE"])


                for node_index, vals in enumerate(data):
                    trace.append(Scatter(
                        x=date_range,
                        y=vals,
                        name="node{}".format(node_index+1)
                    )
                    )
                layout = dict(
                    title = "Ressidues with emitter in node {}, val = {}".format(emitter["EMITTER_NODE"], emitter["EMITTER_VAL"])
                )

                fig = dict(data=trace, layout=layout)
                plot(fig, filename= "Plot_node{}val{}".format(emitter["EMITTER_NODE"], emitter["EMITTER_VAL"]))

        else:
            for emitter in values:
                trace = []
                data = np.transpose(emitter["EN_PRESSURE"])
                for node_index, vals in enumerate(data):
                    trace.append(Scatter(
                        x=date_range,
                        y=vals,
                        name="node{}".format(node_index+1)
                    )
                    )
                layout = dict(
                    title = "Pressure with emitter in node {}, val = {}".format(emitter["EMITTER_NODE"], emitter["EMITTER_VAL"])
                )

                fig = dict(data=trace, layout=layout)
                plot(fig, filename= "Plot_node{}val{}".format(emitter["EMITTER_NODE"], emitter["EMITTER_VAL"]))

    def save_data(self, path=None):

        try:
            with open(path, "wt") as file:
                file.write(json.dumps(self.json_sim))
        except IOError:
            print("Could not write to file {}".format(path))

    @staticmethod
    def write_json(output_json, path):

        json_data = json.dumps(output_json)
        with open(path, "wt") as f:
            f.write(json_data)

    @staticmethod
    def _getncheck(ret_val):

        # check the return code
        if isinstance(ret_val, list):
            if ret_val[0] == 0:
                # everything OK
                return ret_val[1]
            else:
                err_msg = ENgeterror(ret_val[0], 100)
                raise EpanetError(err_msg)
        else:
            if ret_val is not 0:
                err_msg = ENgeterror(ret_val, 100)
                raise EpanetError(err_msg)


def en_check(func):
    def func_wrapper(*args):
        ret_val = func(args)

        # check the return code
        if isinstance(ret_val, list):
            if ret_val[0] == 0:
                # everything OK
                return ret_val[1]
            else:
                err_msg = ENgeterror(ret_val[0], 100)
                raise EpanetError(err_msg)
        else:
            if ret_val is not 0:
                err_msg = ENgeterror(ret_val, 100)
                raise EpanetError(err_msg)


class EpanetError(Exception):

    def __init__(self, err_msg):
        super().__init__(err_msg)


def run_simulation(network, pdd, query_dict):

    es = EPANetSimulation(network, pdd)

    print("Running {}".format(query_dict["simulation_name"]))
    ret_vals = []

    print(query_dict)
    for emitter, emitter_val in query_dict["emitter_values"]:
        print("for node {} simulating emitter_Val {}".format(emitter, emitter_val))

        # modify current network and and save inp temp file

        es.ENsetnodevalue(emitter, EN_EMITTER, emitter_val)

        es.ENsaveinpfile("temp.inp")
        print(es.ENsetnodevalue(emitter, EN_EMITTER, 0))

        e2 = EPANetSimulation("temp.inp", pdd)
        e2.ENsetnodevalue(emitter,EN_EMITTER,emitter_val)

        e2.run()

        node_vals = {}
        link_vals = {}
        for node_query in query_dict["query"]["nodes"]:
            node_vals[node_query] = []

        for link_query in query_dict["query"]["links"]:
            link_vals[link_query] = []

        for node_query in query_dict["query"]["nodes"]:
            for node in e2.network.nodes:
                node_vals[node_query].append(e2.network.nodes[node].results[eval(node_query)])

        for link_query in query_dict["query"]["links"]:
            for link in e2.network.links:
                link_vals[link_query].append(e2.network.links[link].results[eval(link_query)])


        ret_vals.append({
            "EMITTER_VAL" : emitter_val,
            "EMITTER_NODE" : emitter,
            "NODE_VALS" : np.transpose(node_vals).tolist(),
            "LINK_VALS" : np.transpose(link_vals).tolist()
        })
    return ret_vals

#TODO metoda pentru modificarea demand-ului pe nod!!!
if __name__ == '__main__':
    es = ENSim("data/hanoi.inp", pdd=False)

    test_vals = [val for val in range(32) if val % 2 == 0]
    train_vals = [val for val in range(32) if val % 2 == 1]


    intense_leak = [35, 40, 50, 60, 100]

    nodes = list(range(1, 32))

    emitter_test = [ (node, val) for node in nodes for val in test_vals]
    emitter_train = [(node, val) for node in nodes for val in train_vals]
    emitter_intense_leak = [(node, val) for node in nodes for val in intense_leak]

    train_dataset = {
        "simulation_name": "Hanoi train simulation",
        "simulation_type": "H",
        "emitter_values": emitter_train,
        "query": {

            "nodes": ["EN_PRESSURE", "EN_DEMAND"],
            "links": ["EN_VELOCITY"]
        }

    }

    test_dataset = {
        "simulation_name": "Hanoi test simulation",
        "simulation_type": "H",
        "emitter_values": emitter_test,
        "query": {

            "nodes": ["EN_PRESSURE", "EN_DEMAND"],
            "links": ["EN_VELOCITY"]
        }

    }

    test2_dataset = {

        "simulation_name": "Hanoi intense leak simulation",
        "simulation_type": "H",
        "emitter_values": emitter_intense_leak,
        "query": {

            "nodes": ["EN_PRESSURE", "EN_DEMAND"],
            "links": ["EN_VELOCITY"]
        }

    }

    data_train = es.query_network(train_dataset)
    es.save_data("train_set.json")
    data_test = es.query_network(test_dataset)
    es.save_data("test_set.json")
    data_test2 = es.query_network(test2_dataset)
    es.save_data("test2_set.json")