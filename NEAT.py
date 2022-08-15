import numpy as np
import random as rd
"""


connection_genes=[{"innovation":1, "from":0,"to":4, "state":"Enabled", "weight":3,"weight_2":"none"},
                  {"innovation":2, "from":0,"to":5, "state":"Enabled", "weight":3,"weight_2":"none"},
                  {"innovation":3, "from":1,"to":4, "state":"Enabled", "weight":3,"weight_2":"none"},
                  {"innovation":4, "from":2,"to":4, "state":"Enabled", "weight":4,"weight_2":"none"},
                  {"innovation":5, "from":4,"to":5, "state":"Enabled", "weight":5,"weight_2":"none"},
                  {"innovation":6, "from":5,"to":3, "state":"Enabled", "weight":7,"weight_2":"none"},
                  {"innovation":7, "from":4,"to":3, "state":"Enabled", "weight":6,"weight_2":"none"}
                  ]
node_genes=[{"node_num":0,"node_type":"Input","node_value":5,"activation":"None","bias":1,"bias2":"none","Layer":1.2},
            {"node_num":1,"node_type":"Input","node_value":5,"activation":"None","bias":1,"bias2":"none","Layer":1.2},
            {"node_num":2,"node_type":"Input","node_value":5,"activation":"None","bias":1,"bias2":"none","Layer":1.2},
            {"node_num":3,"node_type":"Output","node_value":0,"activation":"None","bias":1,"bias2":"none","Layer":1.2},
            {"node_num":4,"node_type":"Hidden","node_value":0,"activation":"None","bias":1,"bias2":"none","Layer":1.2},
            {"node_num":5,"node_type":"Hidden","node_value":0,"activation":"None","bias":1,"bias2":"none","Layer":1.2}]
genome=[connection_genes,node_genes]


def forward_prop(genome):
    layer = 1
    connection_genes = genome[0]
    node_genes = genome[1]
    max_layer = 0
    for i in range (len(node_genes)):
        if node_genes[i]["Layer"] > max_layer:
            max_layer = node_genes[i]["Layer"]
    while layer <= max_layer:
        for i in range (len(node_genes)):
            new_node_value = 0
            if node_genes[i]["Layer"] == layer:
                to_node = node_genes[i]["node_num"]
                for j in range (len(connection_genes)):
                    if connection_genes[j]["to"] == to_node:
                        if connection_genes[j]["state"]=="Enabled":
                           weight_value=connection_genes[j]["weight"]
                        else:
                           weight_value=0
                        new_node_value += (weight_value*node_genes[connection_genes[j]["from"]]["node_value"])
                node_genes[i]["node_value"] = activate(i,new_node_value+node_genes[i]["bias"])
        layer += 1
    return genome

"""
class NEAT:
    def __init__(self,
                 n_genomes=100,
                 n_inputs=3,
                 n_outputs=2,
                 w_low=-1,
                 w_high=1,
                 b_low=-1,
                 b_high=1,
                 max_generations=200,
                 target_species=5,
                 c1=1,
                 c2=1,
                 c3=0.4,
                 compatability_threshold=3,
                 mutate_add_node_prob=0.2,
                 mutate_add_weight_prob=0.2,
                 mutate_weight_shift_prob=0.2,
                 mutate_weight_random_prob=0.2,
                 mutate_weight_enable_disable_prob=0.2,
                 mutate_shift_bias_prob=0.1,
                 mutate_random_bias_prob=0.1,
                 activation="Relu",
                 output_activation="Sigmoid"
                 ):
          self.n_genomes=n_genomes
          self.n_inputs=n_inputs
          self.n_outputs=n_outputs
          self.w_low=w_low
          self.w_high=w_high
          self.b_low=b_low
          self.b_high=b_high
          self.max_generations=max_generations
          self.target_species=target_species
          self.c1=c1
          self.c2=c2
          self.c3=c3
          self.compatability_threshold=compatability_threshold
          self.mutate_add_node_prob=mutate_add_node_prob
          self.mutate_add_weight_prob=mutate_add_weight_prob
          self.mutate_weight_shift_prob=mutate_weight_shift_prob
          self.mutate_weight_random_prob=mutate_weight_random_prob
          self.mutate_weight_enable_disable_prob=mutate_weight_enable_disable_prob
          self.mutate_shift_bias_prob=mutate_shift_bias_prob
          self.mutate_random_bias_prob=mutate_random_bias_prob
          self.activation=activation
          self.output_activation=output_activation
          self.max_innovation=0
          self.global_connections=[]
    def relu(self,Z):
        return np.maximum(0, Z)
    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s
    def activate(self,i, new_node_value):
        if node_genes[i]["activation"] == "Relu":
            return self.relu(new_node_value)
        if node_genes[i]["activation"] == "Sigmoid":
            return self.sigmoid(new_node_value)
        else:
            return new_node_value
    def initialize_genomes(self):
        self.genomes=[]
        for i in range(self.n_genomes):
            connection_genes=[]
            node_genes=[]
            for j in range(self.n_inputs):
                node_genes.append({"node_num": j,
                                   "node_type": "Input",
                                   "node_value": 0,
                                   "activation": "none",
                                   "bias": 0,
                                   "bias2": "none",
                                   "Layer": 0
                                   })
            for j in range(self.n_inputs,self.n_inputs+self.n_outputs):
                node_genes.append({"node_num": j,
                                   "node_type": "Output",
                                   "node_value": 0,
                                   "activation": self.output_activation,
                                   "bias": rd.uniform(self.b_low, self.b_high),
                                   "bias2": "none",
                                   "Layer": 1
                                   })
            genome=[connection_genes,node_genes]
            self.genomes.append(genome)
        return self.genomes
    def mutate_add_node(self,genome,mutate_add_node_prob):
        connection_genes = genome[0]
        node_genes = genome[1]
        for i in range(len(connection_genes)):
            if np.random.random() < mutate_add_node_prob and connection_genes[i]["state"] == "Enabled":
                old_innovation = connection_genes[i]
                node_genes.append({"node_num": node_genes[len(node_genes) - 1]["node_num"] + 1,
                                   "node_type": "Hidden",
                                   "node_value": 0,
                                   "activation": self.activation,
                                   "bias": rd.uniform(self.b_low, self.b_high),
                                   "bias2": "none",
                                   "Layer": 1.2
                                   })

                new_node_num = node_genes[len(node_genes) - 1]["node_num"]
                innovation = self.checkinnovation(old_innovation["from"],new_node_num)
                connection_genes.append({"innovation": innovation,
                                         "from": old_innovation["from"],
                                         "to": new_node_num,
                                         "state": "Enabled",
                                         "weight": 1,
                                         "weight_2": "none"})

                innovation = self.checkinnovation(new_node_num,old_innovation["to"])
                connection_genes.append({"innovation": self.max_innovation,
                                         "from": new_node_num,
                                         "to": old_innovation["to"],
                                         "state": "Enabled",
                                         "weight": old_innovation["weight"],
                                         "weight_2": "none"})
                genome = [connection_genes, node_genes]
                genome = self.define_layers(genome)
                connection_genes[i]["state"] = "Disabled"
        return genome
    def mutate_add_weight(self,genome, mutate_add_weight_prob, low, high):
        connection_genes = genome[0]
        node_genes = genome[1]
        layers = []
        maxlayer = 0
        ## Find the highest layer
        for i in range(len(node_genes)):
            if node_genes[i]["Layer"] > maxlayer:
                maxlayer = node_genes[i]["Layer"]
        ## Sort the nodes into layers
        for i in range(maxlayer + 1):
            layers_helper = []
            for j in range(len(node_genes)):
                if node_genes[j]["Layer"] == i:
                    layers_helper.append(node_genes[j]["node_num"])
            layers.append(layers_helper)
        potential_connections = []
        for i in range(len(layers) - 1):
            current_layer_nodes = layers[i]
            for j in range(i + 1, len(layers)):
                nextlayer_nodes=layers[j]
                for node in current_layer_nodes:
                    for node2 in nextlayer_nodes:
                        potential_connections.append([node,node2])
        existing_connections = []
        for i in range(len(connection_genes)):
            existing_connections.append([connection_genes[i]["from"], connection_genes[i]["to"]])
        new_potential_connections = []
        for i in range(len(potential_connections)):
            append = True
            for j in range(len(existing_connections)):
                if potential_connections[i] == existing_connections[j]:
                    append = False
            if append:
                new_potential_connections.append(potential_connections[i])
        for i in range(len(new_potential_connections)):
            if np.random.random() < mutate_add_weight_prob:
                innovation=self.checkinnovation(new_potential_connections[i][0],new_potential_connections[i][1])
                connection_genes.append({"innovation":innovation,
                                         "from": new_potential_connections[i][0],
                                         "to": new_potential_connections[i][1],
                                         "state": "Enabled",
                                         "weight": rd.uniform(low, high),
                                         "weight_2": "none"
                                         })
                genome = [connection_genes, node_genes]
                genome = self.define_layers(genome)
        return genome
    def mutate_weight_shift(self,genome, mutate_weight_shift_prob, low, high):
        connection_genes=genome[0]
        node_genes=genome[1]
        if len(connection_genes)!=0:
            for i in range(len(connection_genes)):
                if connection_genes[i]["weight_2"] == "none":
                    connection_genes[i]["weight_2"] = rd.uniform(low, high)
            for i in range(len(connection_genes)):
                if np.random.random() < mutate_weight_shift_prob:
                    w1holder = connection_genes[i]["weight"]
                    connection_genes[i]["weight"] = connection_genes[i]["weight_2"]
                    connection_genes[i]["weight_2"] = w1holder
        genome=[connection_genes,node_genes]
        return genome
    def mutate_weight_random(self,genome, mutate_weight_random_prob, low, high):
        connection_genes = genome[0]
        for i in range(len(connection_genes)):
            if np.random.random() < mutate_weight_random_prob:
                connection_genes[i]["weight"] = rd.uniform(low, high)
        genome[0] = connection_genes
        return genome
    def mutate_weight_enable_disable(self,genome, mutate_enable_disable_prob):
        connection_genes = genome[0]
        for i in range(len(connection_genes)):
            if np.random.random() < mutate_enable_disable_prob:
                if connection_genes[i]["state"] == "Enabled":
                    connection_genes[i]["state"] = "Disabled"
                if connection_genes[i]["state"] == "Disabled":
                    connection_genes[i]["state"] = "Enabled"
        genome[0] = connection_genes
        return genome
    def mutate_bias_random(self,genome, mutate_bias_random_prob, low, high):
        node_genes = genome[1]
        for i in range(len(node_genes)):
            if np.random.random() < mutate_bias_random_prob:
                node_genes[i]["bias"] = rd.uniform(low, high)
        genome[1] = node_genes
        return genome
    def mutate_bias_shift(self,genome, mutate_bias_shift_prob, low, high):
        node_genes = genome[1]
        for i in range(len(node_genes)):
            if node_genes[i]["bias2"] == "none":
                node_genes[i]["bias2"] = rd.uniform(low, high)
        for i in range(len(node_genes)):
            if np.random.random() < mutate_bias_shift_prob:
                w1holder = node_genes[i]["bias"]
                node_genes[i]["bias"] = node_genes[i]["bias2"]
                node_genes[i]["bias2"] = w1holder
        genome[1] = node_genes
        return genome
    def define_layers(self,genome):
        connection_genes = genome[0]
        if len(connection_genes)!=0:
            node_genes = genome[1]
            for i in range(len(node_genes)):
                node_genes[i]["Layer"] = 1.2
            for i in range(len(node_genes)):
                if node_genes[i]["node_type"] == "Input":
                    node_genes[i]["Layer"] = 0
            while True:
                for i in range(len(node_genes)):
                    node_num = node_genes[i]["node_num"]
                    previous_array = []
                    for j in range(len(connection_genes)):
                        if node_num == connection_genes[j]["to"]:
                            from_node = connection_genes[j]["from"]
                            for k in range(len(node_genes)):
                                if node_genes[k]["node_num"] == from_node:
                                    previous_array.append(node_genes[k]["Layer"])
                    if 1.2 not in previous_array and previous_array != []:
                        node_genes[i]["Layer"] = max(previous_array) + 1
                for i in range(len(node_genes)):
                    if node_genes[i]["node_type"]=="Output":
                       connectionto=False
                       for j in range(len(connection_genes)):
                           if connection_genes[j]["to"]==node_genes[i]["node_num"]:
                              connectionto=True
                       if connectionto:
                          continue
                       else:
                          node_genes[i]["Layer"]=1
                done = True
                for i in range(len(node_genes)):
                    if node_genes[i]["Layer"] == 1.2:
                        done = False
                if done:
                    break
        genome = [connection_genes, node_genes]
        return genome
    def mutate(self,genome):
        genome=self.mutate_add_weight(genome,self.mutate_add_weight_prob,self.w_low,self.w_high)
        genome=self.mutate_add_node(genome,self.mutate_add_node_prob)
        genome=self.mutate_weight_shift(genome,self.mutate_weight_shift_prob,self.w_low,self.w_high)
        genome=self.mutate_weight_random(genome,self.mutate_weight_random_prob,self.w_low,self.w_high)
        genome=self.mutate_bias_random(genome,self.mutate_random_bias_prob,self.b_low,self.b_high)
        genome=self.mutate_bias_shift(genome,self.mutate_shift_bias_prob,self.b_low,self.b_high)
        genome=self.mutate_weight_enable_disable(genome, self.mutate_weight_enable_disable_prob)
        return genome
    def forward_prop(self,genome,inputs):

        layer = 1
        connection_genes = genome[0]
        node_genes = genome[1]
        max_layer = 0
        for i in range(len(node_genes)):
            if node_genes[i]["Layer"] > max_layer:
                max_layer = node_genes[i]["Layer"]
        while layer <= max_layer:
            for i in range(len(node_genes)):
                new_node_value = 0
                if node_genes[i]["Layer"] == layer:
                    to_node = node_genes[i]["node_num"]
                    for j in range(len(connection_genes)):
                        if connection_genes[j]["to"] == to_node:
                            if connection_genes[j]["state"] == "Enabled":
                                weight_value = connection_genes[j]["weight"]
                            else:
                                weight_value = 0
                            new_node_value += (weight_value * node_genes[connection_genes[j]["from"]]["node_value"])
                    node_genes[i]["node_value"] = self.activate(i, new_node_value + node_genes[i]["bias"])
            layer += 1
        output=[]
        for i in range(len(node_genes)):
            if node_genes[i]["node_type"]=="Output":
               output.append(node_genes[i]["node_value"])
        return output
    def compatability(self,genome1, genome2):
        con1 = genome1
        con2 = genome2
        W = 0
        divide = 0
        disjoint = []
        excess = []
        common = []
        innovation_common = []
        N = 0
        if len(con1) < len(con2):  # sort the parent so that the shorter one is connection_num1

            link1 = con1
            link2 = con2
            N = len(link2)
        elif len(con2) < len(con1):
            link1 = con2
            link2 = con1
            N = len(link2)
        else:

            link1 = con1
            link2 = con2
            N = len(link2)
        for i in range(
                len(link1)):  # compares shorter parents innovations numbers to longer parents and if the both have 1, append it to common
            for j in range(len(link2)):
                if link1[i]["innovation"] == link2[j]["innovation"]:
                    new_common = link1[i]
                    divide += 1
                    common.append(new_common)
                    W += (np.abs(link1[i]['weight'] - link2[j]['weight']))
        for j in range(
                len(link2)):  # checks if longer parent has any innovation numbers that are larger than the last innovation in the shorte parent
            if link2[j]["innovation"] > link1[len(link1) - 1]["innovation"]:
                excess.append(link2[j])
        for num3 in range(len(common)):
            innovation_common.append(common[num3]["innovation"])
        for i in range(len(link1)):
            if (link1[i]["innovation"] not in innovation_common) and (link1[i] not in excess):
                disjoint.append(link1[i])
        for j in range(len(link2)):
            if (link2[j]["innovation"] not in innovation_common) and (link2[j] not in excess):
                disjoint.append(link2[j])
        disjoint = sorted(disjoint, key=lambda i: i['innovation'])
        E = len(excess)
        D = len(disjoint)
        print(E)
        print(D)
        W = W / divide

        comp = (self.c1E) / N + (self.c2D) / N + self.c3 * W
        return comp
    def printconnections(self,genome):
        connection_genes=genome[0]
        for i in range(len(connection_genes)):
            print(connection_genes[i])
    def printnodes(self,genome):
        node_genes = genome[1]
        for i in range(len(node_genes)):
            print(node_genes[i])
    def checkinnovation(self,from_node,to_node):
        exists=False
        for i in range(len(self.global_connections)):
            if self.global_connections[i]["from"]==from_node and self.global_connections[i]["to"]==to_node:
               exists=True
               innovation_number=self.global_connections[i]["innovation"]
        if not exists:
           innovation_number=self.max_innovation
           self.global_connections.append({
               "innovation":self.max_innovation,
               "from":from_node,
               "to":to_node
           })
           self.max_innovation+=1
        return innovation_number










neat=NEAT()
genomes=neat.initialize_genomes()
for i in range(len(genomes)):
    neat.mutate(genomes[i])

genome0=genomes[0]
neat.printconnections(genome0)
neat.printnodes(genome0)
print("-"*30)

genome19=genomes[19]
neat.printconnections(genome19)
neat.printnodes(genome19)
print("-"*30)
genome30=genomes[30]
neat.printconnections(genome30)
neat.printnodes(genome30)
print("-"*30)
genome50=genomes[50]
neat.printconnections(genome50)
neat.printnodes(genome50)
print("-"*30)
genome81=genomes[81]
neat.printconnections(genome81)
neat.printnodes(genome81)


























