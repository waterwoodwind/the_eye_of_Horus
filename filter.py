# -*- coding: utf-8 -*-

class Filter(object):
    def __init__(self):
        self.company_list = ['CA','MU','SC','ZH','TV','KY','FM','EU','BK','NX','KA','B7','BR','CX','CI','AY','OZ']

    def get_no_repeat(self, list_flt_data):
        no_repeat_list_flt_data = []
        for id in list_flt_data:
            if id not in no_repeat_list_flt_data:
                no_repeat_list_flt_data.append(id)
        return no_repeat_list_flt_data

    def get_own_company(self, list_flt_data):
        own_company_flt_list = []
        for index, item in enumerate(list_flt_data):
            if item[0][0:2] in self.company_list:
                own_company_flt_list.append(item)

        return own_company_flt_list