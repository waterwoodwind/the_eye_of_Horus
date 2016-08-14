# -*- coding: utf-8 -*-

class Filter(object):
    def __init__(self):
        self.company_list = ['CA','MU','SC','ZH','TV','KY','FM','EU','BK','NX','KA','B7','BR','CX','CI','AY','OZ']
        self.chongqing_plane = ["2612","2613","2700",
                                    "5201","5202","5203",
                                    "5214","5217","5220",
                                    "5325","5327","5329",
                                    "5390","5392","5398",
                                    "5426","5443","5477",
                                    "5486","5496","5198",
                                    "2649","1976","1956",
                                    "5803","5679","1527","1738",
                                    "5622","1942","1959",
                                    "5682","5297","5296",
                                    "5583","1768","1765",
                                    "1763","5582","1531",
                                    "6496","6497","7181",
                                   "7892"]
    def get_no_repeat(self, list_flt_data):
        no_repeat_list_flt_data = []
        for id in list_flt_data:
            if id not in no_repeat_list_flt_data:
                no_repeat_list_flt_data.append(id)
        return no_repeat_list_flt_data

    def get_own_company(self, list_flt_data):
        own_company_flt_list = []
        for index, item in enumerate(list_flt_data):
            if item[0][0:2] in self.company_list and item[1] not in self.chongqing_plane:
                own_company_flt_list.append(item)

        return own_company_flt_list

    def filter_chongqing_plane(self, list_flt_data):
        own_company_flt_list = []
        for index, item in enumerate(list_flt_data):
            if item[1] not in self.chongqing_plane:
                print item[1]
                own_company_flt_list.append(item)

        return own_company_flt_list