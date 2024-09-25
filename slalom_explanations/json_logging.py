import typing as tp
import os
import json

def smart_update_json(results_dict, file_name):
    """ Update a json file. """
    isExisting = os.path.exists(file_name)
    if isExisting:
        old_res = json.load(open(file_name))
        _rec_update_layer(old_res, results_dict)
        json.dump(old_res, open(file_name,"w"))
    else:
        json.dump(results_dict, open(file_name,"w"))


def _rec_update_layer(old_dict, results_dict):
    for k in results_dict:
        if k in old_dict:
            if type(results_dict[k]) == dict and type(old_dict[k]) == dict:
                print("Recursively updating key", k)
                _rec_update_layer(old_dict[k], results_dict[k])
            else:
                old_dict[k] = results_dict[k]
        else:
            old_dict[k] = results_dict[k] ## Append new results if key does not exist.

class JSONLogger():

    def __init__(self, filename, hierarchy: tp.List[str]):
        self.filename = filename
        self.hierarchy = hierarchy
    
    def update_result(self, new_results, **kwargs):
        res_new = {}
        lvl_curr = res_new
        for lvl_name in self.hierarchy:
            lvl_curr[kwargs[lvl_name]] = {}
            lvl_prev = lvl_curr
            lvl_curr = lvl_curr[kwargs[lvl_name]]

        lvl_prev[kwargs[self.hierarchy[-1]]] = new_results
        smart_update_json(res_new, self.filename)

