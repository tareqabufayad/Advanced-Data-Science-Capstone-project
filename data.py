# -*- coding: utf-8 -*-
def get_data(name):
    f = open(name, 'r')
    data = []
    for line in f.readlines():
        data.append(line.replace('\n', '').split(' '))
    f.close()
    return data


def label_data(name):
    f = open(name, 'r')
    data = []
    for line in f.readlines():
        data.append(line.replace('\n', ''))
    f.close()
    return data