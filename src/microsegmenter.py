import numpy as np
import pandas as pd
import copy

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

class microsegmenter:
    """
    Cast the elements of an array to a given dtype a nan-safe manner.
    
    Parameters
    ----------
    model : DecisionTreeClassifier or RandomForestClassifier object
        DecisionTreeClassifier or RandomForestClassifier object that has been fit to training data
    X : DataFrame
        Data used in training of tree algorithm, main output required from the data is the column names
    y : array or series 
        data for target variable
    
    Raises
    ------
    TypeError 
        The model type is not a tree-based object
    """
    def _init_(self, model, X, y):
        # Define input variables
        self.model = model
        self.model_type = model_type = type(self.model)._name_
        self.X = X
        self.y = y

        # Define selection parameters
        self.criteria = 'lift'
        self.criteria_value = 2 
        self.min_samples = 0

        self._checks()

        # print("end of execution")

    def _checks(self):

        # Model type checks
        if isinstance(self.model, DecisionTreeClassifier) or isinstance(self.model, RandomForestClassifier):
            # print(f"{self.model_type} to be analysed for microsegments")
            pass
        else:
            raise TypeError("Model type is not DecisionTreeClassifier or RandomForestClassifier")

        # Dataframe type checks
        if isinstance(self.X, pd.DataFrame):
            # print(f"Data is a pandas dataframe")
            pass
        else:
            raise TypeError("Data is not pandas dataframe, which is needed to infer features")
        

    def parse_tree(self, tree_obj):

        # define and assign variables
        n_nodes = tree_obj.tree_.node_count
        children_left = tree_obj.tree_.children_left
        children_right = tree_obj.tree_.children_right
        feature = tree_obj.tree_.feature
        threshold = tree_obj.tree_.threshold
        value = tree_obj.tree_.value
        n_nodes_samples = tree_obj.tree_.n_node_samples
        n_classes = len(tree_obj.classes_)
        base_cr = 1
        training_features = self.X.columns
        target = self.y
        training_input = self.X

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack` so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        # For every node store its values into a dictionary
        tree_schema = {}
        for i in range(n_nodes):
            """
            TODO: Keep only binary classes 
            pass class weights as well
            find out how to pass feature names 
            """

            cw = len(target) / (n_classes*np.bincount(target))
            value_0 = int(value[i][0][0]/cw[0])
            value_1 = int(value[i][0][1]/cw[1])
            
            # Compute metrics
            class_values_ = [value_0, value_1]
            cr = round((value_1/(value_0+value_1)*100),3)

            tree_schema[i] = {
                'leaf': is_leaves[i],
                'children': {'left': children_left[i], 'right': children_right[i]},
                'samples': n_nodes_samples[i],
                # 'feature': training_input.iloc[:,feature[i]].name,
                'feature': training_features[feature[i]],
                'threshold': threshold[i],
                'values': class_values_,
                'conversion_rate': cr,
                'depth': node_depth[i]
            }

        # For each node, find out what path it took to get to the node
        for node in tree_schema:
            if node == 0: # root node
                tree_schema[node]['decision_path'] = []
            else:
                decision_path = []
                check = copy.copy(node)
                for j in reversed(range(node)):
                    if check == tree_schema[j]['children']['left']:
                        decision_path.append((j, 'left', tree_schema[j]['feature']))
                        check = j
                    elif check == tree_schema[j]['children']['right']:
                        decision_path.append((j, 'right', tree_schema[j]['feature']))
                        check = j
                decision_path.sort()
                tree_schema[node]['decision_path'] = decision_path
                tree_schema[node]['decision_path_feature'] = list(set([f[2] for f in decision_path]))


        # Compute lift and other metrics if any
        base_cr = tree_schema[0]['conversion_rate']
        for node in tree_schema:
            tree_schema[node]['lift'] = round(tree_schema[node]['conversion_rate'] / base_cr, 3)

        return tree_schema

    def select_nodes(self, schema, criteria, criteria_value, min_samples):
        # define criteria for microsegments
        selected_nodes = schema.copy()

        for node in selected_nodes:
            if (selected_nodes[node]['samples'] >= min_samples) & (selected_nodes[node][criteria] >= criteria_value):
                selected_nodes[node]['selected'] = True
            else: 
                selected_nodes[node]['selected'] = False
        return selected_nodes


    def store_decision_path(self, schema):
        decision_path = {}

        # print the decision path
        dec_symbol = {
            'left': '<=',
            'right': '>'
        }

        for node in schema:
            if schema[node]['selected']:
                decision_path[node] = f"Node {node} has conversion rate {schema[node]['conversion_rate']}%, lift: {schema[node]['lift']}, samples: {schema[node]['values']} with decision path:\n"
                for dec in schema[node]['decision_path']:
                    path_node = dec[0]
                    feature = schema[path_node]['feature']
                    threshold = schema[path_node]['threshold']
                    symbol = dec_symbol[dec[1]]
                    decision_path[node] += f"\t At node: {path_node} - {feature} {symbol} {threshold} \n"

        return decision_path

    def fit(self, **select_parameters):
        if select_parameters:
            self.criteria = select_parameters['criteria']
            self.criteria_value = select_parameters['criteria_value']
            self.min_samples = select_parameters['min_samples']
        
        criteria = self.criteria
        criteria_value = self.criteria_value
        min_samples = self.min_samples

        # check if its a dt object or a rf 
        if isinstance(self.model, DecisionTreeClassifier):
            self.tree_schema = self.parse_tree(self.model)
            self.tree_schema = self.select_nodes(self.tree_schema, criteria = criteria, criteria_value = criteria_value, min_samples = min_samples)
            self.decision_path = self.store_decision_path(self.tree_schema)
        elif isinstance(self.model, RandomForestClassifier):
            self.tree_schema = {}
            self.decision_path = {}

            for i,tree in enumerate(self.model.estimators_):
                temp_tree_schema = {}
                temp_decision_path = {}
                temp_tree_schema = self.parse_tree(tree)
                temp_tree_schema = self.select_nodes(temp_tree_schema, criteria = criteria, criteria_value = criteria_value, min_samples = min_samples)
                temp_decision_path = self.store_decision_path(temp_tree_schema)

                temp_tree_schema = {(f"{i+1}_{k}"):v for (k,v) in temp_tree_schema.items()}
                temp_decision_path = {(f"{i+1}_{k}"):v for (k,v) in temp_decision_path.items()}

                self.tree_schema.update(temp_tree_schema)
                self.decision_path.update(temp_decision_path)


    # def get_duplicate_nodes(self):
    #     for node in self.tree_schema:
            

    def print_decision_path(self):
        if isinstance(self.model, DecisionTreeClassifier):
            for i, dec in self.decision_path.items():
                print(dec) 
        elif isinstance(self.model, RandomForestClassifier):
            for i, dec in self.decision_path.items():
                print(f"Tree {i} - {dec}") 

    def describe(self):
        # Run initial calculations  
        max_lift = round(np.max([x['lift'] for i,x in self.tree_schema.items()]), 3)
        avg_lift = round(np.mean([x['lift'] for i,x in self.tree_schema.items()]), 3)
        max_cr = round(np.max([x['conversion_rate'] for i,x in self.tree_schema.items()]), 3)
        avg_cr = round(np.mean([x['conversion_rate'] for i,x in self.tree_schema.items()]), 3)

        fstring = f"Model type to be analysed: \n \t {self.model_type}\n"
        fstring += f"Selection parameters:\n \t Criteria: {self.criteria} \n \t Criteria value: {self.criteria_value} \n \t Min samples per node: {self.min_samples} \n"
        fstring += f"Nodes summary:\n \t Total nodes analysed: {len(self.tree_schema)} \n \t Nodes that are selected: {len(self.decision_path)} \n \t Max {self.criteria}: {max_lift} \n \t Avg {self.criteria}: {avg_lift} \n\t Max CR: {max_cr}% \n \t Avg CR: {avg_cr}%"

        return print(fstring)